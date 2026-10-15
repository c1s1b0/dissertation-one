from __future__ import annotations

import json
from threading import RLock
from typing import Any

import z3
from pydantic import BaseModel, Field

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldKind,
    FieldStatus,
    FieldValue,
    RequirementState,
)

from .rules import Rule, build_rules


class SolverResult(BaseModel):
    satisfiable: bool | None
    domains: dict[str, tuple[Any, Any]] = Field(default_factory=dict)
    unsat_core: list[str] = Field(default_factory=list)
    timed_out: bool = False


class ConstraintEngine:
    def __init__(
        self,
        schema: dict[str, FieldDefinition],
        timeout_ms: int = 2_000,
    ) -> None:
        self.schema = schema
        self.timeout_ms = timeout_ms
        self.category_codes = {
            field_id: {
                category: index
                for index, category in enumerate(definition.categories)
            }
            for field_id, definition in schema.items()
            if definition.kind == FieldKind.CATEGORICAL
        }
        self.category_labels = {
            field_id: {code: label for label, code in codes.items()}
            for field_id, codes in self.category_codes.items()
        }
        self.variables = self._build_variables()
        self.rules = build_rules()
        self._result_cache: dict[str, SolverResult] = {}
        self._satisfiability_cache: dict[str, SolverResult] = {}
        self._satisfiability_lock = RLock()
        self._base_satisfiability_solver = self._build_base_satisfiability_solver()
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_info(self) -> dict[str, int]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._result_cache),
        }

    def _state_cache_key(self, state: RequirementState) -> str:
        return json.dumps(
            state.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _cache_result(self, key: str, result: SolverResult) -> SolverResult:
        self._result_cache[key] = result.model_copy(deep=True)
        return result

    def _build_variables(self) -> dict[str, z3.ExprRef]:
        variables: dict[str, z3.ExprRef] = {}
        for field_id, definition in self.schema.items():
            if definition.kind == FieldKind.CONTINUOUS:
                variables[field_id] = z3.Real(field_id)
            elif definition.kind in {
                FieldKind.INTEGER,
                FieldKind.CATEGORICAL,
            }:
                variables[field_id] = z3.Int(field_id)
            else:
                variables[field_id] = z3.Bool(field_id)
        return variables

    def _domain_constraints(
        self, field_ids: set[str] | None = None
    ) -> list[z3.BoolRef]:
        constraints: list[z3.BoolRef] = []
        for field_id, definition in self.schema.items():
            if field_ids is not None and field_id not in field_ids:
                continue
            variable = self.variables[field_id]
            if definition.kind in {FieldKind.CONTINUOUS, FieldKind.INTEGER}:
                constraints.extend(
                    [
                        variable >= definition.lower,
                        variable <= definition.upper,
                    ]
                )
            elif definition.kind == FieldKind.CATEGORICAL:
                constraints.extend(
                    [
                        variable >= 0,
                        variable < len(definition.categories),
                    ]
                )
        return constraints

    def encode_equality(self, field_id: str, value: Any) -> z3.BoolRef:
        definition = self.schema[field_id]
        variable = self.variables[field_id]
        if definition.kind == FieldKind.CATEGORICAL:
            try:
                value = self.category_codes[field_id][str(value)]
            except KeyError as exc:
                raise ValueError(
                    f"unknown category for {field_id}: {value}"
                ) from exc
        if definition.kind == FieldKind.BOOLEAN:
            value = bool(value)
        return variable == value

    def _observation_constraints(
        self, state: RequirementState
    ) -> list[tuple[str, z3.BoolRef]]:
        ignored = {FieldStatus.HIDDEN, FieldStatus.UNKNOWN_TO_USER}
        constraints: list[tuple[str, z3.BoolRef]] = []
        for field_id, field_value in state.values.items():
            if field_id not in self.schema:
                raise KeyError(f"unknown field: {field_id}")
            if field_value.status in ignored or field_value.value is None:
                continue
            constraints.append(
                (
                    f"obs__{field_id}",
                    self.encode_equality(field_id, field_value.value),
                )
            )
        return constraints

    def _plain_constraints(
        self, state: RequirementState
    ) -> list[z3.BoolRef]:
        return [
            *self._domain_constraints(),
            *(rule.build(self.variables) for rule in self.rules),
            *(
                expression
                for _, expression in self._observation_constraints(state)
            ),
        ]

    def _build_base_satisfiability_solver(self) -> z3.Solver:
        """编译与公开状态无关的 domain/rule 约束，供纯可满足性检查复用。"""
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        solver.add(
            *self._domain_constraints(),
            *(rule.build(self.variables) for rule in self.rules),
        )
        return solver

    def _component_fields(self, field_id: str) -> set[str]:
        fields = {field_id}
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                scope = set(rule.scope)
                if fields.intersection(scope) and not scope.issubset(fields):
                    fields.update(scope)
                    changed = True
        return fields

    def _component_constraints(
        self, state: RequirementState, field_id: str
    ) -> list[z3.BoolRef]:
        fields = self._component_fields(field_id)
        return [
            *self._domain_constraints(fields),
            *(
                rule.build(self.variables)
                for rule in self.rules
                if set(rule.scope).issubset(fields)
            ),
            *(
                expression
                for name, expression in self._observation_constraints(state)
                if name.removeprefix("obs__") in fields
            ),
        ]

    def check(self, state: RequirementState) -> SolverResult:
        cache_key = self._state_cache_key(state)
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached.model_copy(deep=True)
        self._cache_misses += 1
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        solver.add(*self._domain_constraints())
        for rule in self.rules:
            solver.assert_and_track(
                rule.build(self.variables), z3.Bool(rule.rule_id)
            )
        for name, expression in self._observation_constraints(state):
            solver.assert_and_track(expression, z3.Bool(name))

        status = solver.check()
        if status == z3.unknown:
            return self._cache_result(
                cache_key, SolverResult(satisfiable=None, timed_out=True)
            )
        if status == z3.unsat:
            return self._cache_result(
                cache_key,
                SolverResult(
                    satisfiable=False,
                    unsat_core=[str(item) for item in solver.unsat_core()],
                ),
            )
        return self._cache_result(
            cache_key,
            SolverResult(
                satisfiable=True,
                domains=self._compute_domains(state),
            ),
        )

    def check_satisfiability(self, state: RequirementState) -> SolverResult:
        """Check feasibility without the expensive all-field domain search."""
        with self._satisfiability_lock:
            cache_key = self._state_cache_key(state)
            cached = self._satisfiability_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached.model_copy(deep=True)
            self._cache_misses += 1
            solver = self._base_satisfiability_solver
            solver.push()
            try:
                solver.add(
                    *(expression for _, expression in self._observation_constraints(state))
                )
                status = solver.check()
            finally:
                solver.pop()
            if status == z3.unknown:
                result = SolverResult(satisfiable=None, timed_out=True)
            else:
                result = SolverResult(satisfiable=status == z3.sat)
            self._satisfiability_cache[cache_key] = result.model_copy(deep=True)
            return result

    def _compute_domains(
        self, state: RequirementState
    ) -> dict[str, tuple[Any, Any]]:
        domains: dict[str, tuple[Any, Any]] = {}
        for field_id, definition in self.schema.items():
            constraints = self._component_constraints(state, field_id)
            if definition.kind in {FieldKind.CONTINUOUS, FieldKind.INTEGER}:
                lower = self._search_numeric_bound(
                    constraints, field_id, minimize=True
                )
                upper = self._search_numeric_bound(
                    constraints, field_id, minimize=False
                )
                if lower is not None and upper is not None:
                    domains[field_id] = (lower, upper)
            elif definition.kind == FieldKind.CATEGORICAL:
                possible = self._possible_categories(
                    constraints, field_id, definition
                )
                if possible:
                    domains[field_id] = (possible[0], possible[-1])
            else:
                possible_bool = self._possible_booleans(
                    constraints, field_id
                )
                if possible_bool:
                    domains[field_id] = (
                        possible_bool[0],
                        possible_bool[-1],
                    )
        return domains

    def _search_numeric_bound(
        self,
        constraints: list[z3.BoolRef],
        field_id: str,
        *,
        minimize: bool,
    ) -> float | None:
        definition = self.schema[field_id]
        variable = self.variables[field_id]
        lower = float(definition.lower)
        upper = float(definition.upper)
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        solver.add(*constraints)

        for _ in range(40):
            midpoint = (lower + upper) / 2.0
            solver.push()
            solver.add(
                variable <= midpoint if minimize else variable >= midpoint
            )
            status = solver.check()
            solver.pop()
            if status == z3.unknown:
                return None
            if minimize:
                if status == z3.sat:
                    upper = midpoint
                else:
                    lower = midpoint
            elif status == z3.sat:
                lower = midpoint
            else:
                upper = midpoint

        return upper if minimize else lower

    def _possible_categories(
        self,
        constraints: list[z3.BoolRef],
        field_id: str,
        definition: FieldDefinition,
    ) -> list[str]:
        possible: list[str] = []
        for category in definition.categories:
            solver = z3.Solver()
            solver.set(timeout=self.timeout_ms)
            solver.add(*constraints)
            solver.add(self.encode_equality(field_id, category))
            if solver.check() == z3.sat:
                possible.append(category)
        return possible

    def _possible_booleans(
        self,
        constraints: list[z3.BoolRef],
        field_id: str,
    ) -> list[bool]:
        possible: list[bool] = []
        for value in (False, True):
            solver = z3.Solver()
            solver.set(timeout=self.timeout_ms)
            solver.add(*constraints)
            solver.add(self.encode_equality(field_id, value))
            if solver.check() == z3.sat:
                possible.append(value)
        return possible

    def counterfactual(
        self, state: RequirementState, field_id: str, value: Any
    ) -> SolverResult:
        values = dict(state.values)
        values[field_id] = FieldValue(
            value=value,
            status=FieldStatus.OBSERVED,
            source="counterfactual",
        )
        return self.check(state.model_copy(update={"values": values}))

    def rule_graph_degree(self, field_id: str) -> float:
        counts = {
            candidate: sum(
                candidate in rule.scope for rule in self.rules
            )
            for candidate in self.schema
        }
        maximum = max(counts.values(), default=1)
        return counts[field_id] / maximum if maximum else 0.0

    def fields_for_rules(self, rule_ids: list[str]) -> set[str]:
        selected = set(rule_ids)
        return {
            field_id
            for rule in self.rules
            if rule.rule_id in selected
            for field_id in rule.scope
        }

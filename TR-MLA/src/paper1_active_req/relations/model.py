from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from paper1_active_req.domain.models import FieldDefinition, FieldKind, FieldStatus, RequirementState
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import RelationFamily, V3RelationConfig
from paper1_active_req.relations.continuous import ContinuousRelationFactor
from paper1_active_req.relations.discrete import DiscreteConditionalFactor
from paper1_active_req.relations.posterior import FieldPosterior, OutcomeBranch


_AVAILABLE_STATUSES = {FieldStatus.OBSERVED, FieldStatus.INFERRED}


@dataclass(frozen=True)
class TypedRelationalModel:
    schema: dict[str, FieldDefinition]
    relation_config: V3RelationConfig
    factors_by_target: dict[str, tuple[DiscreteConditionalFactor | ContinuousRelationFactor, ...]]
    marginal_posteriors: dict[str, FieldPosterior]

    @classmethod
    def fit(
        cls,
        training_specs: Iterable[CompleteSpec],
        schema: dict[str, FieldDefinition],
        relation_config: V3RelationConfig,
        use_relations: bool = True,
    ) -> "TypedRelationalModel":
        specs = tuple(training_specs)
        if not specs:
            raise ValueError("typed relational model requires training specs")
        factors: dict[str, list[DiscreteConditionalFactor | ContinuousRelationFactor]] = {}
        if use_relations:
            for relation_spec in relation_config.relations:
                if relation_spec.family == RelationFamily.DISCRETE_CPT:
                    factor: DiscreteConditionalFactor | ContinuousRelationFactor = DiscreteConditionalFactor.fit(
                        specs, relation_spec
                    )
                else:
                    factor = ContinuousRelationFactor.fit(
                        specs,
                        relation_spec,
                        relation_config.continuous_quantiles,
                        relation_config.continuous_interval_quantiles,
                    )
                factors.setdefault(relation_spec.target_field_id, []).append(factor)
        marginals = {
            field_id: cls._marginal_posterior(
                specs, definition, relation_config.continuous_quantiles
                , relation_config.continuous_interval_quantiles
            )
            for field_id, definition in schema.items()
        }
        return cls(
            schema=dict(schema),
            relation_config=relation_config,
            factors_by_target={field_id: tuple(items) for field_id, items in factors.items()},
            marginal_posteriors=marginals,
        )

    @staticmethod
    def _marginal_posterior(
        specs: tuple[CompleteSpec, ...],
        definition: FieldDefinition,
        quantiles: tuple[float, float, float],
        interval_quantiles: tuple[float, float],
    ) -> FieldPosterior:
        values: list[object] = []
        for spec in specs:
            item = spec.values.get(definition.field_id)
            if item is not None and item.value is not None:
                values.append(item.value)
        if not values:
            raise ValueError(f"training specs omit field {definition.field_id}")
        if definition.kind == FieldKind.CONTINUOUS:
            numeric = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"training marginal for {definition.field_id} must be finite")
            representatives = tuple(float(value) for value in np.quantile(numeric, quantiles))
            nearest_branch = np.argmin(
                np.abs(numeric[:, np.newaxis] - np.asarray(representatives)), axis=1
            )
            counts = np.bincount(nearest_branch, minlength=len(representatives))
            probabilities = tuple(
                float(value) / (len(numeric) + len(representatives))
                for value in counts + 1
            )
            branches = tuple(
                OutcomeBranch(value=value, probability=probability)
                for value, probability in zip(representatives, probabilities, strict=True)
            )
            return FieldPosterior(
                field_id=definition.field_id,
                branches=branches,
                confidence=max(branch.probability for branch in branches),
                interval_low=float(np.quantile(numeric, interval_quantiles[0])),
                interval_high=float(np.quantile(numeric, interval_quantiles[1])),
            )
        counts = Counter(values)
        total = sum(counts.values())
        branches = tuple(
            OutcomeBranch(value=value, probability=count / total)
            for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
        )
        return FieldPosterior(
            field_id=definition.field_id,
            branches=branches,
            confidence=max(branch.probability for branch in branches),
        )

    @staticmethod
    def _discrete_applicable(factor: DiscreteConditionalFactor, state: RequirementState) -> bool:
        return any(
            item is not None and item.status in _AVAILABLE_STATUSES and item.value is not None
            for item in (state.values.get(field_id) for field_id in factor.relation_spec.parent_field_ids)
        )

    def posterior(self, state: RequirementState, field_id: str) -> FieldPosterior:
        if field_id not in self.schema:
            raise KeyError(field_id)
        factors = self.factors_by_target.get(field_id, ())
        applicable: list[tuple[DiscreteConditionalFactor | ContinuousRelationFactor, FieldPosterior]] = []
        for factor in factors:
            if isinstance(factor, ContinuousRelationFactor):
                if factor.is_applicable(state):
                    applicable.append((factor, factor.posterior(state)))
            elif self._discrete_applicable(factor, state):
                applicable.append((factor, factor.posterior(state)))
        if not applicable:
            return self.marginal_posteriors[field_id]
        posteriors = [posterior for _, posterior in applicable]
        if self.schema[field_id].kind == FieldKind.CONTINUOUS:
            return self._combine_continuous(field_id, posteriors)
        return self._combine_discrete(field_id, posteriors)

    @staticmethod
    def _factor_ids(posteriors: list[FieldPosterior]) -> tuple[str, ...]:
        return tuple(
            factor_id for posterior in posteriors for factor_id in posterior.factor_ids
        )

    def _combine_discrete(self, field_id: str, posteriors: list[FieldPosterior]) -> FieldPosterior:
        products: dict[object, float] = {}
        for posterior in posteriors:
            for branch in posterior.branches:
                products[branch.value] = products.get(branch.value, 1.0) * branch.probability
        total = sum(products.values())
        if total <= 0.0:
            raise ValueError("discrete posterior combination has no probability mass")
        branches = tuple(
            OutcomeBranch(value=value, probability=probability / total)
            for value, probability in sorted(products.items(), key=lambda item: (-item[1], str(item[0])))
        )
        return FieldPosterior(
            field_id=field_id,
            branches=branches,
            confidence=max(branch.probability for branch in branches),
            factor_ids=self._factor_ids(posteriors),
        )

    def _combine_continuous(self, field_id: str, posteriors: list[FieldPosterior]) -> FieldPosterior:
        quantile_values = np.asarray(
            [[float(branch.value) for branch in posterior.branches] for posterior in posteriors],
            dtype=float,
        )
        means = np.asarray(
            [
                sum(float(branch.value) * branch.probability for branch in posterior.branches)
                for posterior in posteriors
            ],
            dtype=float,
        )
        variances = np.asarray(
            [
                sum(
                    branch.probability * (float(branch.value) - mean) ** 2
                    for branch in posterior.branches
                )
                for posterior, mean in zip(posteriors, means, strict=True)
            ],
            dtype=float,
        )
        weights = 1.0 / np.maximum(variances, 1e-12)
        combined = tuple(float(value) for value in np.average(quantile_values, axis=0, weights=weights))
        probabilities = tuple(
            float(value)
            for value in np.average(
                np.asarray(
                    [[branch.probability for branch in posterior.branches] for posterior in posteriors],
                    dtype=float,
                ),
                axis=0,
                weights=weights,
            )
        )
        branches = tuple(
            OutcomeBranch(value=value, probability=probability)
            for value, probability in zip(combined, probabilities, strict=True)
        )
        interval_lows = np.asarray(
            [float(posterior.interval_low) for posterior in posteriors], dtype=float
        )
        interval_highs = np.asarray(
            [float(posterior.interval_high) for posterior in posteriors], dtype=float
        )
        return FieldPosterior(
            field_id=field_id,
            branches=branches,
            confidence=max(branch.probability for branch in branches),
            interval_low=float(np.average(interval_lows, weights=weights)),
            interval_high=float(np.average(interval_highs, weights=weights)),
            factor_ids=self._factor_ids(posteriors),
        )

    def reliably_resolved(self, state: RequirementState) -> dict[str, object]:
        resolved: dict[str, object] = {}
        for field_id, item in state.values.items():
            if item.status not in {FieldStatus.HIDDEN, FieldStatus.INFERRED}:
                continue
            if field_id not in self.schema:
                continue
            posterior = self.posterior(state, field_id)
            tolerance = self.relation_config.engineering_tolerances.get(field_id)
            tolerance_met = (
                tolerance is not None
                and posterior.interval_low is not None
                and posterior.interval_high is not None
                and posterior.interval_high - posterior.interval_low <= tolerance
            )
            if posterior.confidence < self.relation_config.confidence_threshold and not tolerance_met:
                continue
            if posterior.interval_low is not None and posterior.interval_high is not None:
                value = min(
                    posterior.branches,
                    key=lambda branch: abs(
                        float(branch.value)
                        - (posterior.interval_low + posterior.interval_high) / 2.0
                    ),
                ).value
            else:
                value = max(posterior.branches, key=lambda branch: branch.probability).value
            resolved[field_id] = value
        return resolved

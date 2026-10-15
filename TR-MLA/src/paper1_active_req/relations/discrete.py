from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import math
from types import MappingProxyType
from typing import Iterable

import yaml

from paper1_active_req.domain.models import FieldKind, FieldStatus, RequirementState
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import RelationFamily, RelationSpec
from paper1_active_req.relations.posterior import FieldPosterior, OutcomeBranch


_AVAILABLE_STATUSES = {FieldStatus.OBSERVED, FieldStatus.INFERRED}


@dataclass(frozen=True)
class _DiscreteDomain:
    kind: FieldKind
    values: tuple[object, ...]


@lru_cache(maxsize=1)
def _frozen_discrete_domains() -> MappingProxyType:
    """Load the v3 discrete schema shipped with this Python package."""
    resource = files("paper1_active_req.relations").joinpath(
        "resources/v3_discrete_schema.yaml"
    )
    raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("fields"), list):
        raise ValueError("packaged discrete schema must contain a fields list")

    domains: dict[str, _DiscreteDomain] = {}
    for item in raw["fields"]:
        if not isinstance(item, dict):
            raise ValueError("packaged discrete schema fields must be mappings")
        field_id = item.get("field_id")
        if not isinstance(field_id, str) or field_id in domains:
            raise ValueError("packaged discrete schema field IDs must be unique")
        try:
            kind = FieldKind(item["kind"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"packaged discrete schema has invalid kind for {field_id!r}"
            ) from error
        if kind == FieldKind.CATEGORICAL:
            categories = item.get("categories")
            if (
                not isinstance(categories, list)
                or not categories
                or not all(isinstance(category, str) for category in categories)
                or len(categories) != len(set(categories))
            ):
                raise ValueError(
                    f"packaged discrete schema has invalid categories for {field_id}"
                )
            values: tuple[object, ...] = tuple(categories)
        elif kind == FieldKind.BOOLEAN:
            values = (False, True)
        else:
            raise ValueError(f"packaged schema field {field_id} is not discrete")
        domains[field_id] = _DiscreteDomain(kind=kind, values=values)
    return MappingProxyType(domains)


def _validate_schema_value(
    field_id: str, value: object, domain: _DiscreteDomain
) -> None:
    if domain.kind == FieldKind.BOOLEAN:
        valid = type(value) is bool
    else:
        valid = value in domain.values
    if not valid:
        raise ValueError(f"schema rejects value for {field_id}: {value!r}")


def _mutual_information(rows: Iterable[tuple[object, object]]) -> float:
    values = tuple(rows)
    total = len(values)
    if total == 0:
        return 0.0
    pair_counts = Counter(values)
    parent_counts = Counter(parent for parent, _ in values)
    target_counts = Counter(target for _, target in values)
    return sum(
        (count / total)
        * math.log((count * total) / (parent_counts[parent] * target_counts[target]))
        for (parent, target), count in pair_counts.items()
    )


@dataclass(frozen=True)
class DiscreteConditionalFactor:
    """Laplace-smoothed discrete CPT with deterministic conditional fallback."""

    relation_spec: RelationSpec
    target_categories: tuple[object, ...]
    parent_domains: MappingProxyType
    joint_counts: dict[tuple[object, ...], Counter[object]]
    single_parent_counts: dict[str, dict[object, Counter[object]]]
    marginal_counts: Counter[object]
    fallback_parent_field_ids: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        specs: Iterable[CompleteSpec],
        relation_spec: RelationSpec,
    ) -> "DiscreteConditionalFactor":
        if relation_spec.family != RelationFamily.DISCRETE_CPT:
            raise ValueError("discrete factor requires a discrete_cpt relation")
        if not relation_spec.parent_field_ids:
            raise ValueError("discrete factor requires at least one parent field")
        if len(relation_spec.parent_field_ids) != len(
            set(relation_spec.parent_field_ids)
        ):
            raise ValueError("discrete factor rejects duplicate parent field IDs")

        schema = _frozen_discrete_domains()
        referenced_field_ids = (
            relation_spec.target_field_id,
            *relation_spec.parent_field_ids,
        )
        try:
            definitions = {
                field_id: schema[field_id] for field_id in referenced_field_ids
            }
        except KeyError as error:
            raise ValueError(f"schema does not define field {error.args[0]}") from error
        target_categories = definitions[relation_spec.target_field_id].values
        joint_counts: dict[tuple[object, ...], Counter[object]] = {}
        single_parent_counts: dict[str, dict[object, Counter[object]]] = {
            field_id: {} for field_id in relation_spec.parent_field_ids
        }
        marginal_counts: Counter[object] = Counter()
        parent_target_rows: dict[str, list[tuple[object, object]]] = {
            field_id: [] for field_id in relation_spec.parent_field_ids
        }

        count = 0
        for spec in specs:
            values = spec.values
            try:
                target = values[relation_spec.target_field_id].value
                parent_values = tuple(
                    values[field_id].value
                    for field_id in relation_spec.parent_field_ids
                )
            except KeyError as error:
                raise ValueError(
                    f"training spec {spec.spec_id} omits relation field {error.args[0]}"
                ) from error
            _validate_schema_value(
                relation_spec.target_field_id,
                target,
                definitions[relation_spec.target_field_id],
            )
            for field_id, value in zip(
                relation_spec.parent_field_ids, parent_values, strict=True
            ):
                _validate_schema_value(field_id, value, definitions[field_id])

            joint_counts.setdefault(parent_values, Counter())[target] += 1
            marginal_counts[target] += 1
            for field_id, value in zip(
                relation_spec.parent_field_ids, parent_values, strict=True
            ):
                single_parent_counts[field_id].setdefault(value, Counter())[target] += 1
                parent_target_rows[field_id].append((value, target))
            count += 1

        if count == 0:
            raise ValueError("discrete factor requires at least one training spec")

        fallback_parent_field_ids = tuple(
            sorted(
                relation_spec.parent_field_ids,
                key=lambda field_id: (-_mutual_information(parent_target_rows[field_id]), field_id),
            )
        )
        return cls(
            relation_spec=relation_spec,
            target_categories=target_categories,
            parent_domains=MappingProxyType(
                {
                    field_id: definitions[field_id]
                    for field_id in relation_spec.parent_field_ids
                }
            ),
            joint_counts=joint_counts,
            single_parent_counts=single_parent_counts,
            marginal_counts=marginal_counts,
            fallback_parent_field_ids=fallback_parent_field_ids,
        )

    def posterior(self, state: RequirementState) -> FieldPosterior:
        known_values: dict[str, object] = {}
        for field_id in self.relation_spec.parent_field_ids:
            item = state.values.get(field_id)
            if item is None or item.status not in _AVAILABLE_STATUSES or item.value is None:
                continue
            _validate_schema_value(
                field_id, item.value, self.parent_domains[field_id]
            )
            known_values[field_id] = item.value

        counts: Counter[object] | None = None
        if len(known_values) == len(self.relation_spec.parent_field_ids):
            joint_key = tuple(
                known_values[field_id]
                for field_id in self.relation_spec.parent_field_ids
            )
            joint_counts = self.joint_counts.get(joint_key)
            if (
                joint_counts is not None
                and sum(joint_counts.values()) >= self.relation_spec.minimum_support
            ):
                counts = joint_counts

        if counts is None:
            for field_id in self.fallback_parent_field_ids:
                value = known_values.get(field_id)
                if value is None:
                    continue
                parent_counts = self.single_parent_counts[field_id].get(value)
                if (
                    parent_counts is not None
                    and sum(parent_counts.values()) >= self.relation_spec.minimum_support
                ):
                    counts = parent_counts
                    break

        if counts is None:
            counts = self.marginal_counts
        return FieldPosterior(
            field_id=self.relation_spec.target_field_id,
            branches=self._smoothed_branches(counts),
            confidence=self._confidence(counts),
            factor_ids=(self.relation_spec.relation_id,),
        )

    def _smoothed_branches(
        self, counts: Counter[object]
    ) -> tuple[OutcomeBranch, ...]:
        support = sum(counts.values())
        denominator = support + len(self.target_categories)
        probabilities = [
            (category, (counts[category] + 1) / denominator, index)
            for index, category in enumerate(self.target_categories)
        ]
        probabilities.sort(key=lambda item: (-item[1], item[2]))
        return tuple(
            OutcomeBranch(value=category, probability=probability)
            for category, probability, _ in probabilities
        )

    def _confidence(self, counts: Counter[object]) -> float:
        support = sum(counts.values())
        return max(
            (counts[category] + 1) / (support + len(self.target_categories))
            for category in self.target_categories
        )

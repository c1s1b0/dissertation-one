from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from paper1_active_req.domain.models import FieldStatus, RequirementState
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import RelationFamily, RelationSpec
from paper1_active_req.relations.posterior import FieldPosterior, OutcomeBranch


_AVAILABLE_STATUSES = {FieldStatus.OBSERVED, FieldStatus.INFERRED}
_DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
_DEFAULT_INTERVAL_QUANTILES = (0.025, 0.975)
_CONTINUOUS_FAMILIES = {
    RelationFamily.AFFINE,
    RelationFamily.RATIO,
    RelationFamily.INVERSE_TIME,
    RelationFamily.GATED_AFFINE,
}


@dataclass(frozen=True)
class _LinearFit:
    coefficients: np.ndarray
    residual_quantiles: tuple[float, ...]
    residual_interval_quantiles: tuple[float, float]
    branch_probabilities: tuple[float, ...]


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite numeric value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


@dataclass(frozen=True)
class ContinuousRelationFactor:
    """Least-squares continuous relation with frozen residual quantiles."""

    relation_spec: RelationSpec
    quantiles: tuple[float, ...]
    interval_quantiles: tuple[float, float]
    global_fit: _LinearFit
    gated_fits: dict[object, _LinearFit]

    @classmethod
    def fit(
        cls,
        specs: Iterable[CompleteSpec],
        relation_spec: RelationSpec,
        quantiles: tuple[float, ...] = _DEFAULT_QUANTILES,
        interval_quantiles: tuple[float, float] = _DEFAULT_INTERVAL_QUANTILES,
    ) -> "ContinuousRelationFactor":
        if relation_spec.family not in _CONTINUOUS_FAMILIES:
            raise ValueError("continuous factor requires a continuous relation family")
        if not relation_spec.parent_field_ids:
            raise ValueError("continuous factor requires at least one parent field")
        if len(relation_spec.parent_field_ids) != len(set(relation_spec.parent_field_ids)):
            raise ValueError("continuous factor rejects duplicate parent field IDs")
        if relation_spec.family in {RelationFamily.RATIO, RelationFamily.INVERSE_TIME} and len(
            relation_spec.parent_field_ids
        ) != 2:
            raise ValueError("ratio and inverse-time relations require exactly two parents")
        if relation_spec.family == RelationFamily.GATED_AFFINE and not relation_spec.gate_field_id:
            raise ValueError("gated affine relation requires a gate field")
        if len(quantiles) != 3 or tuple(sorted(quantiles)) != quantiles or any(
            quantile <= 0.0 or quantile >= 1.0 for quantile in quantiles
        ):
            raise ValueError("continuous quantiles must be three increasing values in (0, 1)")
        if (
            len(interval_quantiles) != 2
            or interval_quantiles[0] <= 0.0
            or interval_quantiles[1] >= 1.0
            or interval_quantiles[0] >= interval_quantiles[1]
        ):
            raise ValueError("continuous interval quantiles must be increasing values in (0, 1)")

        rows: list[tuple[list[float], float, object | None]] = []
        for spec in specs:
            try:
                parent_values = [
                    spec.values[field_id].value
                    for field_id in relation_spec.parent_field_ids
                ]
                target = spec.values[relation_spec.target_field_id].value
                gate = (
                    spec.values[relation_spec.gate_field_id].value
                    if relation_spec.gate_field_id is not None
                    else None
                )
            except KeyError as error:
                raise ValueError(
                    f"training spec {spec.spec_id} omits relation field {error.args[0]}"
                ) from error
            features = cls._transform_parent_values(parent_values, relation_spec)
            rows.append(
                (
                    features,
                    _finite_number(target, context="continuous target"),
                    gate,
                )
            )

        if len(rows) < relation_spec.minimum_support:
            raise ValueError("continuous factor has insufficient minimum support")
        global_fit = cls._fit_linear(rows, quantiles, interval_quantiles)
        gated_fits: dict[object, _LinearFit] = {}
        if relation_spec.family == RelationFamily.GATED_AFFINE:
            by_gate: dict[object, list[tuple[list[float], float, object | None]]] = {}
            for row in rows:
                by_gate.setdefault(row[2], []).append(row)
            for gate, gate_rows in by_gate.items():
                if len(gate_rows) >= relation_spec.minimum_support:
                    gated_fits[gate] = cls._fit_linear(
                        gate_rows, quantiles, interval_quantiles
                    )
        return cls(
            relation_spec=relation_spec,
            quantiles=quantiles,
            interval_quantiles=interval_quantiles,
            global_fit=global_fit,
            gated_fits=gated_fits,
        )

    @staticmethod
    def _fit_linear(
        rows: list[tuple[list[float], float, object | None]],
        quantiles: tuple[float, ...],
        interval_quantiles: tuple[float, float],
    ) -> _LinearFit:
        inputs = np.asarray([row[0] for row in rows], dtype=float)
        targets = np.asarray([row[1] for row in rows], dtype=float)
        design = np.column_stack((np.ones(len(rows), dtype=float), inputs))
        coefficients, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
        residuals = targets - design @ coefficients
        residual_quantiles = tuple(
            float(value) for value in np.quantile(residuals, quantiles)
        )
        nearest_branch = np.argmin(
            np.abs(residuals[:, np.newaxis] - np.asarray(residual_quantiles)), axis=1
        )
        counts = np.bincount(nearest_branch, minlength=len(residual_quantiles))
        branch_probabilities = tuple(
            float(value) / (len(residuals) + len(residual_quantiles))
            for value in counts + 1
        )
        return _LinearFit(
            coefficients=coefficients,
            residual_quantiles=residual_quantiles,
            residual_interval_quantiles=tuple(
                float(value) for value in np.quantile(residuals, interval_quantiles)
            ),
            branch_probabilities=branch_probabilities,
        )

    @staticmethod
    def _transform_parent_values(
        parent_values: list[object], relation_spec: RelationSpec
    ) -> list[float]:
        values = [
            _finite_number(value, context=f"parent {field_id}")
            for field_id, value in zip(relation_spec.parent_field_ids, parent_values, strict=True)
        ]
        if relation_spec.family == RelationFamily.RATIO:
            if values[1] == 0.0:
                raise ValueError("ratio relation has zero denominator")
            return [values[0] / values[1]]
        if relation_spec.family == RelationFamily.INVERSE_TIME:
            if values[1] == 0.0:
                raise ValueError("inverse-time relation has zero denominator")
            return [values[0] / values[1]]
        return values

    def is_applicable(self, state: RequirementState) -> bool:
        return all(
            item is not None and item.status in _AVAILABLE_STATUSES and item.value is not None
            for item in (state.values.get(field_id) for field_id in self.relation_spec.parent_field_ids)
        )

    def posterior(self, state: RequirementState) -> FieldPosterior:
        try:
            parent_values = [
                state.values[field_id].value for field_id in self.relation_spec.parent_field_ids
            ]
        except KeyError as error:
            raise ValueError(f"state omits relation parent {error.args[0]}") from error
        if not self.is_applicable(state):
            raise ValueError("continuous relation requires known parent values")
        features = self._transform_parent_values(parent_values, self.relation_spec)
        fit = self.global_fit
        if self.relation_spec.gate_field_id is not None:
            gate = state.values.get(self.relation_spec.gate_field_id)
            if gate is not None and gate.status in _AVAILABLE_STATUSES and gate.value is not None:
                fit = self.gated_fits.get(gate.value, self.global_fit)
        predicted = float(fit.coefficients[0] + np.dot(fit.coefficients[1:], features))
        values = tuple(predicted + residual for residual in fit.residual_quantiles)
        branches = tuple(
            OutcomeBranch(value=value, probability=probability)
            for value, probability in zip(values, fit.branch_probabilities, strict=True)
        )
        return FieldPosterior(
            field_id=self.relation_spec.target_field_id,
            branches=branches,
            confidence=max(branch.probability for branch in branches),
            interval_low=predicted + fit.residual_interval_quantiles[0],
            interval_high=predicted + fit.residual_interval_quantiles[1],
            factor_ids=(self.relation_spec.relation_id,),
        )

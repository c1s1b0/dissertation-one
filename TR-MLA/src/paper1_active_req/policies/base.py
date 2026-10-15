from __future__ import annotations

from typing import Protocol

from typing import Any
from collections import Counter

import numpy as np
from pydantic import BaseModel, Field

from paper1_active_req.domain.models import FieldValue, RequirementState
from paper1_active_req.solver.engine import SolverResult


class PolicyDecision(BaseModel):
    field_id: str | None
    utility_terms: dict[str, float] = Field(default_factory=dict)


class WeightedAnswer(BaseModel):
    value: Any
    probability: float = Field(gt=0.0, le=1.0)


class EmpiricalAnswerModel(BaseModel):
    outcomes: dict[str, list[WeightedAnswer]]

    def answers_for(self, field_id: str) -> list[WeightedAnswer]:
        return self.outcomes.get(field_id, [])

    @classmethod
    def from_complete_specs(
        cls,
        *,
        values: list[dict[str, FieldValue]],
        max_numeric_outcomes: int = 5,
    ) -> "EmpiricalAnswerModel":
        if max_numeric_outcomes < 1:
            raise ValueError("max_numeric_outcomes must be positive")
        collected: dict[str, list[Any]] = {}
        for spec_values in values:
            for field_id, field_value in spec_values.items():
                if field_value.value is not None:
                    collected.setdefault(field_id, []).append(
                        field_value.value
                    )

        outcomes: dict[str, list[WeightedAnswer]] = {}
        for field_id, observed in collected.items():
            is_numeric = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in observed
            )
            if is_numeric and len(set(observed)) > max_numeric_outcomes:
                quantiles = np.linspace(
                    0.0, 1.0, max_numeric_outcomes + 2
                )[1:-1]
                representatives = [
                    float(value)
                    for value in np.quantile(observed, quantiles)
                ]
                probability = 1.0 / len(representatives)
                outcomes[field_id] = [
                    WeightedAnswer(
                        value=value,
                        probability=probability,
                    )
                    for value in representatives
                ]
                continue

            counts = Counter(observed)
            total = sum(counts.values())
            outcomes[field_id] = [
                WeightedAnswer(
                    value=value,
                    probability=count / total,
                )
                for value, count in sorted(
                    counts.items(), key=lambda item: str(item[0])
                )
            ]
        return cls(outcomes=outcomes)


class AcquisitionPolicy(Protocol):
    name: str

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: object,
    ) -> PolicyDecision | None:
        ...

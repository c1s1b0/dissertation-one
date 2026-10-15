from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OutcomeBranch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: object
    probability: float = Field(gt=0.0, le=1.0)


class FieldPosterior(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field_id: str
    branches: tuple[OutcomeBranch, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    interval_low: float | None = None
    interval_high: float | None = None
    factor_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_probability_mass(self) -> "FieldPosterior":
        mass = sum(branch.probability for branch in self.branches)
        if abs(mass - 1.0) > 1e-9:
            raise ValueError("posterior branch probabilities must sum to 1")
        endpoints = (self.interval_low, self.interval_high)
        if any(endpoint is not None and not math.isfinite(endpoint) for endpoint in endpoints):
            raise ValueError("posterior interval endpoints must be finite")
        if (
            self.interval_low is not None
            and self.interval_high is not None
            and self.interval_low > self.interval_high
        ):
            raise ValueError("posterior interval low endpoint cannot exceed high endpoint")
        return self

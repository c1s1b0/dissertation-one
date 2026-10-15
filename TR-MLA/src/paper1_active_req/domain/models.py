from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FieldKind(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class FieldStatus(StrEnum):
    OBSERVED = "observed"
    HIDDEN = "hidden"
    INFERRED = "inferred"
    UNKNOWN_TO_USER = "unknown_to_user"
    CONFLICTING = "conflicting"


class FieldDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    field_id: str
    kind: FieldKind
    unit: str | None = None
    lower: float | None = None
    upper: float | None = None
    categories: tuple[str, ...] = ()
    criticality: float = Field(ge=0.0, le=1.0)
    answer_cost: float = Field(ge=0.0)
    queryable: bool = True

    @model_validator(mode="after")
    def validate_domain(self) -> "FieldDefinition":
        if self.kind in {FieldKind.CONTINUOUS, FieldKind.INTEGER}:
            if self.lower is None or self.upper is None:
                raise ValueError("numeric fields require lower and upper bounds")
            if self.lower >= self.upper:
                raise ValueError("lower must be smaller than upper")
        if self.kind == FieldKind.CATEGORICAL and not self.categories:
            raise ValueError("categorical fields require categories")
        return self


class FieldValue(BaseModel):
    value: Any | None
    status: FieldStatus
    source: str = "initial"


class RequirementState(BaseModel):
    values: dict[str, FieldValue]
    asked_fields: list[str] = Field(default_factory=list)

    def queryable_field_ids(self) -> list[str]:
        allowed = {
            FieldStatus.HIDDEN,
            FieldStatus.CONFLICTING,
            FieldStatus.INFERRED,
        }
        return sorted(
            field_id
            for field_id, item in self.values.items()
            if item.status in allowed and field_id not in self.asked_fields
        )

    def apply_answer(
        self, field_id: str, answer: FieldValue
    ) -> "RequirementState":
        if field_id not in self.values:
            raise KeyError(field_id)
        values = dict(self.values)
        values[field_id] = answer
        return self.model_copy(
            update={
                "values": values,
                "asked_fields": [*self.asked_fields, field_id],
            }
        )

    def public_copy(self) -> "RequirementState":
        return self.model_copy(deep=True)

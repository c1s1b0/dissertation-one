from __future__ import annotations

from enum import StrEnum
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper1_active_req.domain.models import FieldDefinition, FieldKind
from paper1_active_req.domain.schema import load_field_schema


class RelationFamily(StrEnum):
    DISCRETE_CPT = "discrete_cpt"
    AFFINE = "affine"
    RATIO = "ratio"
    INVERSE_TIME = "inverse_time"
    GATED_AFFINE = "gated_affine"


class RelationSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relation_id: str
    family: RelationFamily
    target_field_id: str
    parent_field_ids: tuple[str, ...]
    gate_field_id: str | None = None
    minimum_support: int = Field(default=3, ge=1)

    @property
    def model_fields(self) -> dict[str, Any]:
        """兼容旧式实例字段检查，避免 Pydantic 的弃用警告。"""
        return type(self).__pydantic_fields__


class V3RelationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field_schema_path: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    continuous_quantiles: tuple[float, float, float]
    continuous_interval_quantiles: tuple[float, float]
    engineering_tolerances: dict[str, float]
    relations: tuple[RelationSpec, ...]

    @model_validator(mode="after")
    def validate_continuous_quantiles(self) -> "V3RelationConfig":
        if (
            tuple(sorted(self.continuous_quantiles)) != self.continuous_quantiles
            or any(value <= 0.0 or value >= 1.0 for value in self.continuous_quantiles)
        ):
            raise ValueError("continuous quantiles must be increasing values in (0, 1)")
        low, high = self.continuous_interval_quantiles
        if low <= 0.0 or high >= 1.0 or low >= high:
            raise ValueError("continuous interval quantiles must be increasing values in (0, 1)")
        return self


def load_v3_relation_config(path: Path) -> V3RelationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v3 relation config must be a mapping")

    config = V3RelationConfig.model_validate(raw)
    schema_path = Path(config.field_schema_path)
    if not schema_path.is_absolute():
        schema_path = path.parent / schema_path

    field_schema: dict[str, FieldDefinition] = load_field_schema(schema_path)
    relation_ids = [relation.relation_id for relation in config.relations]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("relation IDs must be unique")

    for relation in config.relations:
        referenced_field_ids = {
            relation.target_field_id,
            *relation.parent_field_ids,
        }
        if relation.gate_field_id is not None:
            referenced_field_ids.add(relation.gate_field_id)
        unknown_fields = referenced_field_ids - set(field_schema)
        if unknown_fields:
            raise ValueError(f"unknown relation fields: {sorted(unknown_fields)}")
        if relation.target_field_id in relation.parent_field_ids:
            raise ValueError("relation target field cannot also be a parent")
        if (
            field_schema[relation.target_field_id].kind == FieldKind.CONTINUOUS
            and relation.target_field_id not in config.engineering_tolerances
        ):
            raise ValueError(
                "continuous relation targets require an engineering tolerance"
            )

    for field_id, tolerance in config.engineering_tolerances.items():
        definition = field_schema.get(field_id)
        if definition is None:
            raise ValueError(f"unknown engineering tolerance field: {field_id}")
        if definition.kind != FieldKind.CONTINUOUS:
            raise ValueError("engineering tolerances require continuous fields")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("engineering tolerances must be finite and positive")

    return config

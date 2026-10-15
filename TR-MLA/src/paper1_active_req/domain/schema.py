from pathlib import Path

import yaml

from .models import FieldDefinition


def load_field_schema(path: Path) -> dict[str, FieldDefinition]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("fields"), list):
        raise ValueError("schema must contain a fields list")

    definitions = [
        FieldDefinition.model_validate(item) for item in raw["fields"]
    ]
    schema = {item.field_id: item for item in definitions}
    if len(schema) != len(definitions):
        raise ValueError("field_id must be unique")
    return schema

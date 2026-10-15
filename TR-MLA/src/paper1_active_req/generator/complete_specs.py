from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.solver.engine import ConstraintEngine


class CompleteSpec(BaseModel):
    spec_id: str
    values: dict[str, FieldValue]
    split: str = ""

    def to_requirement_state(self) -> RequirementState:
        return RequirementState(values=self.values).model_copy(deep=True)


class RequirementTask(BaseModel):
    task_id: str
    spec_id: str
    split: str
    missing_rate: float
    initial_state: RequirementState
    ground_truth: dict[str, FieldValue]
    injected_conflict_type: str | None = None
    injected_conflict_field: str | None = None


class DatasetManifest(BaseModel):
    complete_spec_count: int
    task_count: int
    split_spec_counts: dict[str, int]
    train_spec_ids: list[str]
    validation_spec_ids: list[str]
    test_spec_ids: list[str]
    seed: int


def _observed(value: Any) -> FieldValue:
    return FieldValue(
        value=value,
        status=FieldStatus.OBSERVED,
        source="ground_truth",
    )


def generate_complete_spec(
    rng: np.random.Generator,
    schema: dict[str, FieldDefinition],
    engine: ConstraintEngine,
    spec_id: str,
) -> CompleteSpec:
    for _ in range(50):
        mass = float(rng.uniform(5.0, 80.0))
        length = float(rng.uniform(100.0, 1200.0))
        width = float(rng.uniform(250.0, 800.0))
        height = float(rng.uniform(20.0, 600.0))
        throughput = float(rng.uniform(60.0, 300.0))
        cycle_limit = 3600.0 / throughput
        cycle_time = float(
            rng.uniform(max(3.0, 0.65 * cycle_limit), cycle_limit)
        )
        transfer_time = float(
            rng.uniform(max(0.5, 0.25 * cycle_time), 0.8 * cycle_time)
        )
        fragile = bool(rng.random() < 0.3)
        speed_upper = 500.0 if fragile else 1500.0
        speed = float(rng.uniform(150.0, speed_upper))
        distance_upper = min(5000.0, speed * transfer_time * 0.9)
        if distance_upper <= 100.0:
            continue
        distance = float(rng.uniform(100.0, distance_upper))
        accuracy = float(rng.uniform(0.1, 5.0))
        repeatability = float(rng.uniform(0.02, accuracy))
        side_clearance = float(rng.uniform(50.0, 120.0))
        available_width = float(
            width + 2.0 * side_clearance + rng.uniform(0.0, 300.0)
        )
        available_length = float(
            distance + length + 100.0 + rng.uniform(0.0, 500.0)
        )
        available_height = float(
            height + 100.0 + rng.uniform(0.0, 500.0)
        )
        safety_factor = float(rng.uniform(1.5, 3.0))
        load_capacity = float(
            safety_factor * mass + rng.uniform(0.0, 100.0)
        )
        interface = str(rng.choice(["belt", "roller", "pallet"]))
        output_interface = str(rng.choice(["belt", "roller", "pallet"]))

        values = {
            "workpiece_mass_kg": _observed(mass),
            "workpiece_length_mm": _observed(length),
            "workpiece_width_mm": _observed(width),
            "workpiece_height_mm": _observed(height),
            "target_throughput_per_hour": _observed(throughput),
            "cycle_time_s": _observed(cycle_time),
            "travel_distance_mm": _observed(distance),
            "transfer_time_s": _observed(transfer_time),
            "conveyor_speed_mm_s": _observed(speed),
            "positioning_accuracy_mm": _observed(accuracy),
            "repeatability_mm": _observed(repeatability),
            "available_length_mm": _observed(available_length),
            "available_width_mm": _observed(available_width),
            "available_height_mm": _observed(available_height),
            "working_height_mm": _observed(
                float(rng.uniform(700.0, 1100.0))
            ),
            "side_clearance_mm": _observed(side_clearance),
            "load_capacity_kg": _observed(load_capacity),
            "safety_factor": _observed(safety_factor),
            "upstream_output_interface": _observed(interface),
            "station_input_interface": _observed(interface),
            "station_output_interface": _observed(output_interface),
            "downstream_input_interface": _observed(output_interface),
            "workpiece_fragile": _observed(fragile),
            "power_supply": _observed(
                str(rng.choice(["220vac", "380vac", "24vdc"]))
            ),
        }
        if set(values) != set(schema):
            raise RuntimeError("generated fields do not match schema")
        spec = CompleteSpec(spec_id=spec_id, values=values)
        if engine.check(spec.to_requirement_state()).satisfiable is True:
            return spec
    raise RuntimeError(f"failed to generate satisfiable spec: {spec_id}")


def _write_jsonl(path: Path, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(item.model_dump_json() for item in items)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")


def generate_dataset(
    *,
    seed: int,
    count: int,
    output_dir: Path,
    schema_path: Path,
) -> DatasetManifest:
    from .perturbations import inject_conflict, mask_spec

    if count < 5:
        raise ValueError("count must be at least 5")
    schema = load_field_schema(schema_path)
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(seed)
    specs = [
        generate_complete_spec(
            rng, schema, engine, f"spec-{index:04d}"
        )
        for index in range(count)
    ]

    train_count = round(count * 0.6)
    validation_count = round(count * 0.2)
    split_specs = {
        "train": specs[:train_count],
        "validation": specs[
            train_count : train_count + validation_count
        ],
        "test": specs[train_count + validation_count :],
    }
    for split, split_items in split_specs.items():
        for spec in split_items:
            spec.split = split

    tasks_by_split: dict[str, list[RequirementTask]] = {
        split: [] for split in split_specs
    }
    conflict_types = [
        "input_interface",
        "available_width",
        "load_capacity",
    ]
    task_index = 0
    for split, split_items in split_specs.items():
        for spec in split_items:
            for missing_rate in (0.3, 0.5, 0.7):
                task = mask_spec(
                    spec,
                    missing_rate=missing_rate,
                    rng=rng,
                    queryable_field_ids={
                        field_id
                        for field_id, definition in schema.items()
                        if definition.queryable
                    },
                )
                if task_index % 3 == 0:
                    task = inject_conflict(
                        task,
                        conflict_type=conflict_types[
                            (task_index // 3) % len(conflict_types)
                        ],
                        rng=rng,
                    )
                tasks_by_split[split].append(task)
                task_index += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "complete_specs.jsonl", specs)
    for split, tasks in tasks_by_split.items():
        _write_jsonl(output_dir / f"{split}_tasks.jsonl", tasks)

    manifest = DatasetManifest(
        complete_spec_count=len(specs),
        task_count=sum(len(items) for items in tasks_by_split.values()),
        split_spec_counts={
            split: len(items) for split, items in split_specs.items()
        },
        train_spec_ids=[item.spec_id for item in split_specs["train"]],
        validation_spec_ids=[
            item.spec_id for item in split_specs["validation"]
        ],
        test_spec_ids=[item.spec_id for item in split_specs["test"]],
        seed=seed,
    )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest

"""合成任务生成、确定性硬约束真值和冻结工件。"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from .catalog import DEFAULT_SEED, SCENE_ROLES, build_catalog
from .schema import ComponentTemplate, ConfigurationTask, FreezeManifest, TruthRecord

VALID_SPLITS = frozenset({"dev", "val", "test", "free-chat"})
FREEZE_COUNTS = {"dev": 8, "val": 8, "test": 12, "free-chat": 4}
SCALE_STATUS = "provisional_auditable_base"
FORMAL_TARGET = {"A/test": 200, "B/test": 200, "A/dev": 120, "A/val": 60}


def _by_id(catalog: Iterable[ComponentTemplate]) -> dict[str, ComponentTemplate]:
    return {item.template_id: item for item in catalog}


def _requirement_values(scene: str, rng: random.Random) -> dict[str, float]:
    if scene == "A":
        return {
            "drive.speed_rpm.min": float(rng.choice((800, 1200, 1600))),
            "conveyor.load_kg.max": float(rng.choice((180, 260, 340))),
            "positioner.payload_kg.min": float(rng.choice((160, 240, 320))),
            "transmission.ratio.max": float(rng.choice((5, 6, 7))),
        }
    return {
        "motor.torque_nm.min": float(rng.choice((60, 100, 140))),
        "gearbox.ratio.max": float(rng.choice((7, 9, 11))),
        "coupling.bore_mm.min": float(rng.choice((14, 22, 30))),
        "shaft_support.bore_mm.min": float(rng.choice((14, 22, 30))),
    }


def _build_tasks_with_truth(
    scene: str, split: str, n: int, seed: int
) -> tuple[list[ConfigurationTask], list[TruthRecord]]:
    """生成只依赖 seed 的配置/修复任务；不读取外部资料。"""
    if scene not in SCENE_ROLES:
        raise ValueError(f"unknown scene: {scene}")
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split: {split}")
    if n < 0:
        raise ValueError("n must be non-negative")
    catalog = build_catalog(seed)
    by_role: dict[str, list[ComponentTemplate]] = {role: [] for role in SCENE_ROLES[scene]}
    for item in catalog:
        if item.scene == scene:
            by_role[item.role].append(item)
    rng = random.Random((seed * 1_000_003) ^ sum(ord(char) for char in scene + split))
    tasks: list[ConfigurationTask] = []
    truths: list[TruthRecord] = []
    roles = SCENE_ROLES[scene]
    for index in range(n):
        task_rng = random.Random(rng.randrange(2**63))
        candidate_ids = {
            role: tuple(item.template_id for item in task_rng.sample(by_role[role], 3))
            for role in roles
        }
        task_type = "repair" if index % 2 else "configuration"
        requirements = _requirement_values(scene, task_rng)
        injected_conflict = None
        conflict_scope = None
        if task_type == "repair":
            conflict_scope = roles[index % len(roles)]
            injected_conflict = (
                "global_no_solution" if index % 11 == 1 else f"repair-{conflict_scope}-constraint"
            )
        requirement = (
            f"{scene}场景{split}任务 {index + 1}：选择覆盖 "
            f"{', '.join(roles)} 的兼容模板并满足数值边界。"
        )
        task = ConfigurationTask(
            task_id=f"{scene}-{split}-{seed:08d}-{index:04d}",
            scene=scene,
            split=split,
            seed=seed,
            task_type=task_type,
            requirement=requirement,
            required_roles=roles,
            candidate_ids=candidate_ids,
            numeric_requirements=requirements,
            injected_conflict=injected_conflict,
            conflict_scope=conflict_scope,
        )
        truth = evaluate_truth(task, catalog)
        task = replace(task, expected_feasible=truth.feasible)
        tasks.append(task)
        truths.append(truth)
    return tasks, truths


def build_tasks(scene: str, split: str, n: int, seed: int) -> list[ConfigurationTask]:
    return _build_tasks_with_truth(scene, split, n, seed)[0]


def _numeric_ok(item: ComponentTemplate, key: str, value: float) -> bool:
    role, field, operator = key.split(".", 2)
    if item.role != role or field not in item.parameters:
        return False
    actual = item.parameters[field]
    return actual >= value if operator == "min" else actual <= value


def _combination_ok(task: ConfigurationTask, selected: tuple[ComponentTemplate, ...]) -> bool:
    if task.injected_conflict == "global_no_solution":
        return False
    by_role = {item.role: item for item in selected}
    if set(by_role) != set(task.required_roles):
        return False
    if any(item.scene != task.scene for item in selected):
        return False
    if any(not _numeric_ok(by_role[key.split(".", 1)[0]], key, value)
           for key, value in task.numeric_requirements.items()):
        return False
    if task.scene == "A":
        drive, transmission = by_role["drive"], by_role["transmission"]
        conveyor, positioner = by_role["conveyor"], by_role["positioner"]
        if drive.parameters["capacity_kg"] < conveyor.parameters["load_kg"]:
            return False
        if positioner.parameters["payload_kg"] < conveyor.parameters["load_kg"]:
            return False
        if transmission.parameters["speed_rpm"] > drive.parameters["speed_rpm"] * 2:
            return False
    else:
        motor, gearbox = by_role["motor"], by_role["gearbox"]
        coupling, support = by_role["coupling"], by_role["shaft_support"]
        if gearbox.parameters["input_speed_rpm"] > motor.parameters["speed_rpm"]:
            return False
        if coupling.parameters["torque_nm"] < motor.parameters["torque_nm"]:
            return False
        if coupling.parameters["bore_mm"] < motor.parameters["shaft_diameter_mm"]:
            return False
        if support.parameters["bore_mm"] < motor.parameters["shaft_diameter_mm"]:
            return False
    return True


def _solutions(task: ConfigurationTask, catalog: list[ComponentTemplate], restricted: bool) -> list[dict[str, str]]:
    by_id = _by_id(catalog)
    choices = []
    for role in task.required_roles:
        ids = task.candidate_ids[role] if restricted else tuple(
            item.template_id for item in catalog if item.scene == task.scene and item.role == role
        )
        role_items = [by_id[item_id] for item_id in ids if item_id in by_id]
        role_items = [
            item for item in role_items
            if all(
                key.split(".", 1)[0] != role or _numeric_ok(item, key, value)
                for key, value in task.numeric_requirements.items()
            )
        ]
        choices.append(role_items)
    solutions: list[dict[str, str]] = []

    def visit(index: int, selected: tuple[ComponentTemplate, ...]) -> None:
        if index == len(choices):
            solutions.append({item.role: item.template_id for item in selected})
            return
        for item in choices[index]:
            candidate = selected + (item,)
            if _partial_combination_ok(task, candidate):
                visit(index + 1, candidate)

    visit(0, ())
    return solutions


def _partial_combination_ok(task: ConfigurationTask, selected: tuple[ComponentTemplate, ...]) -> bool:
    """在组合尚未完整时应用已可判定的跨部件硬约束。"""
    by_role = {item.role: item for item in selected}
    if task.scene == "A":
        drive, conveyor = by_role.get("drive"), by_role.get("conveyor")
        transmission, positioner = by_role.get("transmission"), by_role.get("positioner")
        if drive and conveyor and drive.parameters["capacity_kg"] < conveyor.parameters["load_kg"]:
            return False
        if positioner and conveyor and positioner.parameters["payload_kg"] < conveyor.parameters["load_kg"]:
            return False
        if transmission and drive and transmission.parameters["speed_rpm"] > drive.parameters["speed_rpm"] * 2:
            return False
    else:
        motor, gearbox = by_role.get("motor"), by_role.get("gearbox")
        coupling, support = by_role.get("coupling"), by_role.get("shaft_support")
        if motor and gearbox and gearbox.parameters["input_speed_rpm"] > motor.parameters["speed_rpm"]:
            return False
        if motor and coupling and coupling.parameters["torque_nm"] < motor.parameters["torque_nm"]:
            return False
        if motor and coupling and coupling.parameters["bore_mm"] < motor.parameters["shaft_diameter_mm"]:
            return False
        if motor and support and support.parameters["bore_mm"] < motor.parameters["shaft_diameter_mm"]:
            return False
    return True


def evaluate_truth(task: ConfigurationTask, catalog: list[ComponentTemplate]) -> TruthRecord:
    """用全枚举和同一组硬约束返回可复核真值。"""
    current = _solutions(task, catalog, restricted=True)
    full = _solutions(task, catalog, restricted=False)
    selected = current[0] if current else None
    violations = () if current else ("no_feasible_combination",)
    checked = 1
    for role in task.required_roles:
        checked *= len(task.candidate_ids.get(role, ()))
    return TruthRecord(
        task_id=task.task_id,
        feasible=bool(current),
        selected_candidate_ids=selected,
        violated_constraints=violations,
        checked_combinations=checked,
        full_catalog_solution_count=len(full),
        eligible_for_feasibility_rate=bool(full),
    )


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _canonical_manifest_payload(
    seed: int,
    catalog_count: int,
    task_counts: dict[str, int],
    counts: dict[str, int],
    scale_status: str,
    formal_target: dict[str, int],
    files: dict[str, str],
) -> bytes:
    return json.dumps({
        "seed": seed, "catalog_count": catalog_count,
        "task_counts": dict(sorted(task_counts.items())),
        "counts": dict(sorted(counts.items())),
        "scale_status": scale_status,
        "formal_target": dict(sorted(formal_target.items())),
        "files": dict(sorted(files.items())),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _clear_managed_outputs(output_dir: Path) -> None:
    """清理本函数管理的固定输出，不触碰目录中的其他文件。"""
    root = output_dir.resolve()
    for name in ("catalog.jsonl", "tasks", "truth", "freeze_manifest.json"):
        target = root / name
        if target.exists() and target.is_symlink():
            raise ValueError(f"managed output must not be a symlink: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _normalize_freeze_counts(counts: Mapping[str, int] | None) -> dict[str, int]:
    valid_keys = {f"{scene}/{split}" for scene in SCENE_ROLES for split in VALID_SPLITS}
    if counts is None:
        return {f"{scene}/{split}": FREEZE_COUNTS[split] for scene in SCENE_ROLES for split in FREEZE_COUNTS}
    unknown = set(counts) - valid_keys
    if unknown:
        raise ValueError(f"unknown freeze count keys: {', '.join(sorted(unknown))}")
    normalized = {key: 0 for key in sorted(valid_keys)}
    for key, value in counts.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"freeze count must be a non-negative int: {key}")
        normalized[key] = value
    return normalized


def freeze_benchmark(
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    counts: Mapping[str, int] | None = None,
    scale_status: str = SCALE_STATUS,
) -> FreezeManifest:
    """写出可重复的目录、双场景任务 JSONL 及哈希 manifest。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and output_dir.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_managed_outputs(output_dir)
    requested_counts = _normalize_freeze_counts(counts)
    if not scale_status or not isinstance(scale_status, str):
        raise ValueError("scale_status must be a non-empty string")
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    truth_dir = output_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(seed)
    _write_jsonl(output_dir / "catalog.jsonl", (item.to_dict() for item in catalog))
    task_counts: dict[str, int] = {}
    for scene in SCENE_ROLES:
        for split in FREEZE_COUNTS:
            count = requested_counts[f"{scene}/{split}"]
            tasks, truths = _build_tasks_with_truth(scene, split, count, seed)
            relative = f"tasks/{scene}_{split}.jsonl"
            _write_jsonl(output_dir / relative, (task.to_dict() for task in tasks))
            truth_relative = f"truth/{scene}_{split}.jsonl"
            _write_jsonl(
                output_dir / truth_relative,
                (truth.to_dict() for truth in truths),
            )
            task_counts[f"{scene}/{split}"] = len(tasks)
    files = {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.rglob("*.jsonl"))
    }
    manifest_hash = hashlib.sha256(
        _canonical_manifest_payload(
            seed, len(catalog), task_counts, requested_counts, scale_status, FORMAL_TARGET, files
        )
    ).hexdigest()
    manifest = FreezeManifest(
        seed, len(catalog), task_counts, requested_counts, scale_status, FORMAL_TARGET, files, manifest_hash
    )
    (output_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest

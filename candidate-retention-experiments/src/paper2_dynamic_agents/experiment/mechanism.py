"""B 场景候选池反事实重放基础模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schema import ComponentTemplate, ConfigurationTask
from .solver import enumerate_feasible_assignments, solve_full, solve_subgraph


@dataclass(frozen=True)
class PoolSnapshot:
    """同一任务的四种候选池，均保持稳定的 catalog 顺序。"""

    task_pool: tuple[str, ...]
    p_pool: tuple[str, ...]
    p_llm_pool: tuple[str, ...]
    full_pool: tuple[str, ...]


@dataclass(frozen=True)
class ReplayResult:
    """一个候选池在 typed 或非 typed solver 下的确定性结果。"""

    pool_name: str
    typed: bool
    candidate_ids: tuple[str, ...]
    role_coverage: dict[str, int]
    feasible: bool
    selected_candidate_ids: tuple[str, ...]
    violations: tuple[str, ...]
    search_nodes: int
    candidate_edge_count: int


def sanitize_pool(
    ids: Iterable[str],
    by_id: Mapping[str, ComponentTemplate],
    scene: str,
) -> tuple[str, ...]:
    """清洗候选池，去重并移除未知或异场景 ID。"""

    values: list[str] = []
    seen: set[str] = set()
    for raw_id in ids:
        item_id = str(raw_id)
        item = by_id.get(item_id)
        if item is None or item.scene != scene or item_id in seen:
            continue
        seen.add(item_id)
        values.append(item_id)
    return tuple(values)


def _task_value(task: ConfigurationTask | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(key, default)
    return getattr(task, key, default)


def _coerce_task(task: ConfigurationTask | Mapping[str, Any]) -> ConfigurationTask:
    if isinstance(task, ConfigurationTask):
        return task
    return ConfigurationTask(
        task_id=str(task["task_id"]),
        scene=str(task["scene"]),
        split=str(task["split"]),
        seed=int(task["seed"]),
        task_type=str(task["task_type"]),
        requirement=str(task["requirement"]),
        required_roles=tuple(str(role) for role in task["required_roles"]),
        candidate_ids={str(role): tuple(str(item_id) for item_id in ids) for role, ids in task["candidate_ids"].items()},
        numeric_requirements={str(key): float(value) for key, value in task.get("numeric_requirements", {}).items()},
        injected_conflict=task.get("injected_conflict"),
        conflict_scope=task.get("conflict_scope"),
        expected_feasible=bool(task.get("expected_feasible", False)),
    )


def _task_pool_ids(task: ConfigurationTask | Mapping[str, Any]) -> tuple[str, ...]:
    candidate_ids = _task_value(task, "candidate_ids", {}) or {}
    values: list[str] = []
    required_roles = tuple(_task_value(task, "required_roles", ()) or ())
    for role in required_roles:
        values.extend(str(item_id) for item_id in candidate_ids.get(role, ()))
    return tuple(values)


def _row_pool_ids(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    values: list[str] = []
    for row in rows:
        if row.get("status", "success") != "success":
            continue
        values.extend(str(item_id) for item_id in row.get("activated_ids", ()) or ())
    return tuple(values)


def reconstruct_pools(
    task: ConfigurationTask | Mapping[str, Any],
    p_rows: Iterable[Mapping[str, Any]],
    p_llm_rows: Iterable[Mapping[str, Any]],
    catalog: Iterable[ComponentTemplate],
) -> PoolSnapshot:
    """从任务和最终状态记录重建四种候选池。"""

    items = list(catalog)
    by_id = {item.template_id: item for item in items}
    scene = str(_task_value(task, "scene", ""))
    return PoolSnapshot(
        task_pool=sanitize_pool(_task_pool_ids(task), by_id, scene),
        p_pool=sanitize_pool(_row_pool_ids(p_rows), by_id, scene),
        p_llm_pool=sanitize_pool(_row_pool_ids(p_llm_rows), by_id, scene),
        full_pool=tuple(item.template_id for item in items if item.scene == scene),
    )


def _group_ids(task: ConfigurationTask | Mapping[str, Any], ids: Iterable[str], by_id: Mapping[str, ComponentTemplate]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item_id in ids:
        item = by_id.get(item_id)
        if item is not None:
            grouped.setdefault(item.role, []).append(item_id)
    required_roles = tuple(_task_value(task, "required_roles", ()) or ())
    return {role: tuple(grouped.get(role, ())) for role in required_roles}


def _candidate_edge_count(task: ConfigurationTask | Mapping[str, Any], ids: Iterable[str], by_id: Mapping[str, ComponentTemplate]) -> int:
    values = tuple(ids)
    required_roles = tuple(_task_value(task, "required_roles", ()) or ())
    total = 0
    for left_role, right_role in zip(required_roles, required_roles[1:]):
        left = [by_id[item_id] for item_id in values if item_id in by_id and by_id[item_id].role == left_role]
        right = [by_id[item_id] for item_id in values if item_id in by_id and by_id[item_id].role == right_role]
        total += sum(
            any(
                a.direction == "out"
                and b.direction == "in"
                and a.port_type == b.port_type
                and a.unit == b.unit
                for a in left_item.ports
                for b in right_item.ports
            )
            for left_item in left
            for right_item in right
        )
    return total


def replay_pool(
    task: ConfigurationTask | Mapping[str, Any],
    pool_name: str,
    candidate_ids: Iterable[str],
    catalog: Iterable[ComponentTemplate],
    typed: bool = True,
) -> ReplayResult:
    """在指定候选池上运行一次确定性 solver。"""

    task_object = _coerce_task(task)
    items = list(catalog)
    by_id = {item.template_id: item for item in items}
    scene = str(_task_value(task, "scene", ""))
    clean_ids = sanitize_pool(candidate_ids, by_id, scene)
    grouped = _group_ids(task_object, clean_ids, by_id)
    required_roles = tuple(task_object.required_roles)
    role_coverage = {role: len(grouped.get(role, ())) for role in required_roles}
    if pool_name == "full_pool" and set(clean_ids) == {item.template_id for item in items if item.scene == scene}:
        result = solve_full(task_object, items)
    else:
        result = solve_subgraph(task_object, grouped, items, typed=typed)
    violations = list(result.violations)
    if any(count == 0 for count in role_coverage.values()) and "role_coverage" not in violations:
        violations.append("role_coverage")
    selected = tuple(result.selected_candidate_ids.get(role, "") for role in required_roles if result.selected_candidate_ids.get(role))
    return ReplayResult(
        pool_name=pool_name,
        typed=typed,
        candidate_ids=clean_ids,
        role_coverage=role_coverage,
        feasible=result.feasible,
        selected_candidate_ids=selected,
        violations=tuple(sorted(set(violations))),
        search_nodes=result.search_nodes,
        candidate_edge_count=_candidate_edge_count(task_object, clean_ids, by_id),
    )


def minimal_additions(
    task: ConfigurationTask | Mapping[str, Any],
    p_pool: Iterable[str],
    catalog: Iterable[ComponentTemplate],
    feasible_assignments: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """计算使 P 候选池获得一个完整可行见证所需的最少新增模板数。"""

    task_object = _coerce_task(task)
    items = list(catalog)
    by_id = {item.template_id: item for item in items}
    clean_pool = sanitize_pool(p_pool, by_id, task_object.scene)
    full_ids = tuple(item.template_id for item in items if item.scene == task_object.scene)
    full_grouped = _group_ids(task_object, full_ids, by_id)
    assignments = (
        feasible_assignments
        if feasible_assignments is not None
        else enumerate_feasible_assignments(task_object, full_grouped, items, typed=True)
    )
    if not assignments:
        return {"count": None, "missing_roles": [], "missing_template_ids": [], "witness_assignment": {}}
    pool_set = set(clean_pool)
    candidates = []
    for assignment in assignments:
        missing = tuple(
            assignment[role]
            for role in task_object.required_roles
            if assignment[role] not in pool_set
        )
        missing_roles = tuple(
            role for role in task_object.required_roles if assignment[role] not in pool_set
        )
        candidates.append((len(missing), missing, missing_roles, assignment))
    _, missing, missing_roles, witness = min(candidates, key=lambda value: (value[0], value[1]))
    return {
        "count": len(missing),
        "missing_roles": list(missing_roles),
        "missing_template_ids": list(missing),
        "witness_assignment": dict(witness),
    }


def _truth_value(truth: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return truth.get(key, default)


def classify_mechanism(
    task: ConfigurationTask | Mapping[str, Any],
    replays: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    """按规格优先级将一个任务归入唯一主机制类别。"""

    p_typed: ReplayResult = replays["p_typed"]
    p_untyped: ReplayResult = replays["p_untyped"]
    full_typed: ReplayResult = replays["full_typed"]
    full_untyped: ReplayResult = replays["full_untyped"]
    observed_p_feasible = replays.get("observed_p_feasible")
    full_solution_count = _truth_value(truth, "full_catalog_solution_count")
    full_feasible = bool(full_typed.feasible or full_untyped.feasible or (full_solution_count or 0) > 0)
    minimal = replays.get("minimal_additions", {}) or {}
    minimal_count = minimal.get("count")
    numeric_violation = any(
        value.startswith("numeric:") or value.startswith("B:")
        for value in p_typed.violations
    )
    diagnostics = {
        "intrinsic_infeasible": not full_feasible,
        "role_coverage_gap": any(value == 0 for value in p_typed.role_coverage.values()),
        "typed_bottleneck": (not p_typed.feasible and p_untyped.feasible),
        "candidate_recall_gap": (
            full_feasible
            and not p_typed.feasible
            and not p_untyped.feasible
            and (minimal_count is None or minimal_count > 0)
        ),
        "numeric_constraint_gap": numeric_violation,
        "search_overhead": (
            p_typed.feasible
            and full_typed.feasible
            and p_typed.search_nodes > full_typed.search_nodes
        ),
    }
    priority = (
        "intrinsic_infeasible",
        "role_coverage_gap",
        "typed_bottleneck",
        "candidate_recall_gap",
        "numeric_constraint_gap",
        "search_overhead",
    )
    primary = next((name for name in priority if diagnostics[name]), "no_gap" if observed_p_feasible is not False and p_typed.feasible else "uncategorized")
    return {
        "primary_category": primary,
        "diagnostics": diagnostics,
        "minimal_additions": minimal,
        "full_catalog_solution_count": full_solution_count,
        "task_id": str(_task_value(task, "task_id", "")),
    }


def analyze_task(
    task: ConfigurationTask | Mapping[str, Any],
    p_rows: Iterable[Mapping[str, Any]],
    p_llm_rows: Iterable[Mapping[str, Any]],
    b0_row: Mapping[str, Any] | None,
    catalog: Iterable[ComponentTemplate],
    truth: Mapping[str, Any],
    full_assignments: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """完成单个任务的候选池重建、反事实重放和机制分类。"""

    items = list(catalog)
    snapshot = reconstruct_pools(task, p_rows, p_llm_rows, items)
    replays: dict[str, Any] = {
        "p_typed": replay_pool(task, "p_pool", snapshot.p_pool, items, typed=True),
        "p_untyped": replay_pool(task, "p_pool", snapshot.p_pool, items, typed=False),
        "p_llm_typed": replay_pool(task, "p_llm_pool", snapshot.p_llm_pool, items, typed=True),
        "full_typed": replay_pool(task, "full_pool", snapshot.full_pool, items, typed=True),
        "full_untyped": replay_pool(task, "full_pool", snapshot.full_pool, items, typed=False),
    }
    successful_p_rows = [row for row in p_rows if row.get("status") == "success"]
    replays["observed_p_feasible"] = bool(successful_p_rows[0].get("feasible")) if successful_p_rows else False
    replays["minimal_additions"] = minimal_additions(task, snapshot.p_pool, items, feasible_assignments=full_assignments)
    classification = classify_mechanism(task, replays, truth)
    return {
        "task_id": str(_task_value(task, "task_id", "")),
        "scene": str(_task_value(task, "scene", "")),
        "split": str(_task_value(task, "split", "")),
        "task_type": str(_task_value(task, "task_type", "")),
        "pool_snapshot": snapshot,
        "replays": replays,
        "classification": classification,
        "p_row": dict(p_rows[0]) if isinstance(p_rows, list) and p_rows else None,
        "b0_row": dict(b0_row) if b0_row else None,
    }


__all__ = [
    "PoolSnapshot",
    "ReplayResult",
    "analyze_task",
    "classify_mechanism",
    "minimal_additions",
    "reconstruct_pools",
    "replay_pool",
    "sanitize_pool",
]

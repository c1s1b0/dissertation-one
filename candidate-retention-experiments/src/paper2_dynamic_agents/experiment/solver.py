"""共享、确定性的离线硬约束裁决器。"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from .schema import ComponentTemplate, ConfigurationTask


@dataclass(frozen=True)
class SolverResult:
    feasible: bool
    selected_candidate_ids: dict[str, str]
    rejected_candidates: tuple[str, ...]
    violations: tuple[str, ...]
    search_nodes: int
    solver_calls: int
    elapsed_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def solve_full(task: ConfigurationTask, catalog: Iterable[ComponentTemplate]) -> SolverResult:
    items = list(catalog)
    choices = {
        role: tuple(item.template_id for item in items if item.scene == task.scene and item.role == role)
        for role in task.required_roles
    }
    return _solve(task, items, choices, typed=True)


def solve_subgraph(
    task: ConfigurationTask,
    candidate_ids: Mapping[str, Iterable[str]] | Iterable[str],
    catalog: Iterable[ComponentTemplate],
    typed: bool = True,
) -> SolverResult:
    items = list(catalog)
    by_id = {item.template_id: item for item in items}
    mapping_violations: tuple[str, ...] = ()
    if isinstance(candidate_ids, Mapping):
        choices = {role: tuple(ids) for role, ids in candidate_ids.items()}
        mismatch = tuple(
            f"role_mismatch:{role}:{item_id}"
            for role, ids in choices.items()
            for item_id in ids
            if item_id in by_id and by_id[item_id].role != role
        )
        mapping_violations = mismatch
    else:
        allowed = set(candidate_ids)
        choices = {
            role: tuple(item.template_id for item in items if item.role == role and item.template_id in allowed)
            for role in task.required_roles
        }
    choices = {role: tuple(item_id for item_id in choices.get(role, ()) if item_id in by_id) for role in task.required_roles}
    return _solve(task, items, choices, typed=typed, initial_violations=mapping_violations)


def enumerate_feasible_assignments(
    task: ConfigurationTask,
    candidate_ids: Mapping[str, Iterable[str]] | Iterable[str],
    catalog: Iterable[ComponentTemplate],
    typed: bool = True,
) -> list[dict[str, str]]:
    """枚举候选池中的全部可行角色组合。

    该接口与 :func:`solve_subgraph` 使用同一套硬约束裁决，但不在首个
    可行组合处提前返回，供机制分析计算完整目录中的最小补充候选数。
    未知 ID、异场景 ID 和角色映射不一致的 ID 会被排除。
    """

    items = list(catalog)
    by_id = {item.template_id: item for item in items}
    if task.injected_conflict == "global_no_solution":
        return []
    if isinstance(candidate_ids, Mapping):
        choices = {
            role: tuple(
                item_id
                for item_id in ids
                if item_id in by_id
                and by_id[item_id].scene == task.scene
                and by_id[item_id].role == role
            )
            for role, ids in candidate_ids.items()
        }
    else:
        allowed = set(candidate_ids)
        choices = {
            role: tuple(
                item.template_id
                for item in items
                if item.template_id in allowed and item.scene == task.scene and item.role == role
            )
            for role in task.required_roles
        }
    choices = {role: tuple(choices.get(role, ())) for role in task.required_roles}
    if any(not options for options in choices.values()):
        return []

    assignments: list[dict[str, str]] = []
    role_choices = [choices[role] for role in task.required_roles]
    for combination in itertools.product(*role_choices):
        chosen = {role: by_id[item_id] for role, item_id in zip(task.required_roles, combination)}
        if not _hard_constraint_failures(task, chosen, by_id, typed):
            assignments.append(dict(zip(task.required_roles, combination)))
    return assignments


def _solve(task: ConfigurationTask, catalog: list[ComponentTemplate], choices: dict[str, tuple[str, ...]], typed: bool, initial_violations: tuple[str, ...] = ()) -> SolverResult:
    started = time.perf_counter()
    by_id = {item.template_id: item for item in catalog}
    all_candidates = tuple(item_id for role in task.required_roles for item_id in choices.get(role, ()))
    violations: list[str] = list(initial_violations)
    warnings: list[str] = ["typed_check_bypassed"] if not typed else []
    if task.injected_conflict == "global_no_solution":
        violations.append("global_no_solution")
    selected: dict[str, str] = {}
    nodes = 0
    if not violations:
        role_choices = [choices.get(role, ()) for role in task.required_roles]
        if any(not options for options in role_choices):
            violations.append("role_coverage")
        else:
            for combination in itertools.product(*role_choices):
                nodes += 1
                chosen = {role: by_id[item_id] for role, item_id in zip(task.required_roles, combination)}
                failed = _hard_constraint_failures(task, chosen, by_id, typed)
                if not failed:
                    selected = dict(zip(task.required_roles, combination))
                    break
            if not selected and not violations:
                violations.extend(_hard_constraint_failures(task, {role: by_id[opts[0]] for role, opts in zip(task.required_roles, role_choices)}, by_id, typed))
                if not violations:
                    violations.append("no_feasible_combination")
    rejected = tuple(item_id for item_id in all_candidates if item_id not in selected.values())
    return SolverResult(
        feasible=bool(selected),
        selected_candidate_ids=selected,
        rejected_candidates=rejected,
        violations=tuple(dict.fromkeys(violations + warnings)),
        search_nodes=nodes,
        solver_calls=1,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _hard_constraint_failures(task: ConfigurationTask, chosen: dict[str, ComponentTemplate], by_id: dict[str, ComponentTemplate], typed: bool) -> list[str]:
    failures: list[str] = []
    if set(chosen) != set(task.required_roles):
        failures.append("role_coverage")
        return failures
    if any(item.scene != task.scene for item in chosen.values()):
        failures.append("scene")
    for key, value in task.numeric_requirements.items():
        role, field, operator = key.split(".", 2)
        item = chosen.get(role)
        if item is None or field not in item.parameters:
            failures.append(f"numeric:{key}")
        elif operator == "min" and item.parameters[field] < value:
            failures.append(f"numeric:{key}")
        elif operator == "max" and item.parameters[field] > value:
            failures.append(f"numeric:{key}")
    if task.scene == "A":
        drive, transmission = chosen.get("drive"), chosen.get("transmission")
        conveyor, positioner = chosen.get("conveyor"), chosen.get("positioner")
        if drive and conveyor and drive.parameters["capacity_kg"] < conveyor.parameters["load_kg"]:
            failures.append("A:drive_capacity")
        if positioner and conveyor and positioner.parameters["payload_kg"] < conveyor.parameters["load_kg"]:
            failures.append("A:positioner_payload")
        if transmission and drive and transmission.parameters["speed_rpm"] > drive.parameters["speed_rpm"] * 2:
            failures.append("A:transmission_speed")
    elif task.scene == "B":
        motor, gearbox = chosen.get("motor"), chosen.get("gearbox")
        coupling, support = chosen.get("coupling"), chosen.get("shaft_support")
        if motor and gearbox and gearbox.parameters["input_speed_rpm"] > motor.parameters["speed_rpm"]:
            failures.append("B:gearbox_speed")
        if motor and coupling and coupling.parameters["torque_nm"] < motor.parameters["torque_nm"]:
            failures.append("B:coupling_torque")
        if motor and coupling and coupling.parameters["bore_mm"] < motor.parameters["shaft_diameter_mm"]:
            failures.append("B:coupling_bore")
        if motor and support and support.parameters["bore_mm"] < motor.parameters["shaft_diameter_mm"]:
            failures.append("B:support_bore")
    if typed:
        roles = task.required_roles
        for left_role, right_role in zip(roles, roles[1:]):
            left, right = chosen.get(left_role), chosen.get(right_role)
            if left and right and not _ports_compatible(left, right):
                failures.append(f"typed:{left_role}->{right_role}")
    return failures


def _ports_compatible(left: ComponentTemplate, right: ComponentTemplate) -> bool:
    outgoing = [port for port in left.ports if port.direction == "out"]
    incoming = [port for port in right.ports if port.direction == "in"]
    return any(a.port_type == b.port_type and a.unit == b.unit for a in outgoing for b in incoming)

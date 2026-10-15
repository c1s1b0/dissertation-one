from pathlib import Path
from threading import Thread

import pytest
import z3

import paper1_active_req.solver.engine as engine_module
from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.solver.engine import ConstraintEngine


@pytest.fixture
def engine() -> ConstraintEngine:
    schema = load_field_schema(Path("configs/fields.yaml"))
    return ConstraintEngine(schema, timeout_ms=2_000)


def test_width_constraint_propagates_minimum_available_width(
    engine: ConstraintEngine,
):
    state = RequirementState(
        values={
            "workpiece_width_mm": FieldValue(
                value=400.0, status=FieldStatus.OBSERVED
            ),
            "side_clearance_mm": FieldValue(
                value=50.0, status=FieldStatus.OBSERVED
            ),
            "available_width_mm": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
        }
    )

    result = engine.check(state)

    assert result.satisfiable is True
    assert result.domains["available_width_mm"][0] == pytest.approx(
        500.0, abs=1e-6
    )


def test_interface_mismatch_returns_named_unsat_core(
    engine: ConstraintEngine,
):
    state = RequirementState(
        values={
            "upstream_output_interface": FieldValue(
                value="belt", status=FieldStatus.OBSERVED
            ),
            "station_input_interface": FieldValue(
                value="roller", status=FieldStatus.OBSERVED
            ),
        }
    )

    result = engine.check(state)

    assert result.satisfiable is False
    assert "input_interface_match" in result.unsat_core


def test_nonlinear_component_returns_complete_numeric_bounds(
    engine: ConstraintEngine,
):
    result = engine.check(RequirementState(values={}))

    assert result.satisfiable is True
    assert result.domains["travel_distance_mm"] == pytest.approx(
        (100.0, 5000.0), abs=1e-5
    )
    assert result.domains["transfer_time_s"] == pytest.approx(
        (0.5, 80.0), abs=1e-5
    )


def test_repeated_public_state_uses_cached_solver_result(
    engine: ConstraintEngine,
):
    state = RequirementState(
        values={
            "workpiece_width_mm": FieldValue(
                value=400.0, status=FieldStatus.OBSERVED
            )
        }
    )

    first = engine.check(state)
    second = engine.check(state.public_copy())

    assert second == first
    assert engine.cache_info() == {"hits": 1, "misses": 1, "size": 1}


def test_satisfiability_only_check_matches_full_solver_without_domains(
    engine: ConstraintEngine,
):
    """防止前瞻策略为只需可行性判断却重复计算全部字段边界。"""
    state = RequirementState(
        values={
            "workpiece_width_mm": FieldValue(
                value=400.0, status=FieldStatus.OBSERVED
            ),
            "side_clearance_mm": FieldValue(
                value=50.0, status=FieldStatus.OBSERVED
            ),
            "available_width_mm": FieldValue(
                value=600.0, status=FieldStatus.OBSERVED
            ),
        }
    )

    result = engine.check_satisfiability(state)

    assert result.satisfiable is True
    assert result.domains == {}
    assert result.unsat_core == []


def _legacy_satisfiability(engine: ConstraintEngine, state: RequirementState):
    """独立保留修复前的逐状态构造参考路径，只用于语义等价断言。"""
    solver = z3.Solver()
    solver.set(timeout=engine.timeout_ms)
    solver.add(*engine._plain_constraints(state))
    status = solver.check()
    if status == z3.unknown:
        return None, True
    return status == z3.sat, False


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            RequirementState(
                values={
                    "workpiece_width_mm": FieldValue(
                        value=400.0, status=FieldStatus.OBSERVED
                    ),
                    "side_clearance_mm": FieldValue(
                        value=50.0, status=FieldStatus.OBSERVED
                    ),
                }
            ),
            True,
        ),
        (
            RequirementState(
                values={
                    "upstream_output_interface": FieldValue(
                        value="belt", status=FieldStatus.OBSERVED
                    ),
                    "station_input_interface": FieldValue(
                        value="roller", status=FieldStatus.OBSERVED
                    ),
                }
            ),
            False,
        ),
    ],
)
def test_precompiled_satisfiability_matches_legacy_sat_and_unsat(
    engine: ConstraintEngine, state: RequirementState, expected: bool
):
    """若 base solver 漏加固定约束或观测 push/pop 泄漏，结果会偏离原逐状态 Solver。"""
    legacy_sat, legacy_timed_out = _legacy_satisfiability(engine, state)

    actual = engine.check_satisfiability(state)

    assert (actual.satisfiable, actual.timed_out) == (legacy_sat, legacy_timed_out)
    assert actual.satisfiable is expected


def test_precompiled_satisfiability_reuses_base_solver_for_distinct_states(
    monkeypatch: pytest.MonkeyPatch,
):
    """若不同反事实状态仍新建 Solver，固定 domain/rule 编译无法复用。"""
    original_solver = z3.Solver
    constructions = 0

    class CountingSolver:
        def __init__(self, *args, **kwargs):
            nonlocal constructions
            constructions += 1
            self._inner = original_solver(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    monkeypatch.setattr(engine_module.z3, "Solver", CountingSolver)
    schema = load_field_schema(Path("configs/fields.yaml"))
    checked = ConstraintEngine(schema)
    sat_state = RequirementState(
        values={
            "workpiece_width_mm": FieldValue(
                value=400.0, status=FieldStatus.OBSERVED
            )
        }
    )
    unsat_state = RequirementState(
        values={
            "upstream_output_interface": FieldValue(
                value="belt", status=FieldStatus.OBSERVED
            ),
            "station_input_interface": FieldValue(
                value="roller", status=FieldStatus.OBSERVED
            ),
        }
    )

    assert checked.check_satisfiability(sat_state).satisfiable is True
    assert checked.check_satisfiability(unsat_state).satisfiable is False
    assert checked.check_satisfiability(sat_state.public_copy()).satisfiable is True
    assert constructions == 1
    assert checked.cache_info()["hits"] == 1


def test_precompiled_satisfiability_push_pop_isolates_distinct_states(
    engine: ConstraintEngine,
):
    """若并发反事实的 push/pop 未隔离，SAT 状态可能被相邻 UNSAT 观察污染。"""
    sat_state = RequirementState(
        values={
            "workpiece_width_mm": FieldValue(
                value=400.0, status=FieldStatus.OBSERVED
            )
        }
    )
    unsat_state = RequirementState(
        values={
            "upstream_output_interface": FieldValue(
                value="belt", status=FieldStatus.OBSERVED
            ),
            "station_input_interface": FieldValue(
                value="roller", status=FieldStatus.OBSERVED
            ),
        }
    )
    results: list[bool | None] = []

    def run(state: RequirementState) -> None:
        results.append(engine.check_satisfiability(state).satisfiable)

    threads = [Thread(target=run, args=(state,)) for state in (sat_state, unsat_state)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results, key=lambda value: value is not True) == [True, False]


def test_precompiled_satisfiability_preserves_unknown_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    """若 base solver 将 unknown 误作 SAT/UNSAT，前瞻分支的可行性语义会改变。"""
    class UnknownSolver:
        def set(self, **_kwargs: object) -> None:
            return None

        def add(self, *_assertions: object) -> None:
            return None

        def push(self) -> None:
            return None

        def pop(self) -> None:
            return None

        def check(self):
            return z3.unknown

    monkeypatch.setattr(engine_module.z3, "Solver", UnknownSolver)
    schema = load_field_schema(Path("configs/fields.yaml"))
    checked = ConstraintEngine(schema)
    state = RequirementState(values={})

    legacy_sat, legacy_timed_out = _legacy_satisfiability(checked, state)
    actual = checked.check_satisfiability(state)

    assert (actual.satisfiable, actual.timed_out) == (legacy_sat, legacy_timed_out)
    assert actual.satisfiable is None
    assert actual.timed_out is True


RULE_CASES = [
    ("cycle_throughput_relation", {"cycle_time_s": 10, "target_throughput_per_hour": 300}, True),
    ("cycle_throughput_relation", {"cycle_time_s": 20, "target_throughput_per_hour": 300}, False),
    ("transfer_within_cycle", {"transfer_time_s": 5, "cycle_time_s": 10}, True),
    ("transfer_within_cycle", {"transfer_time_s": 12, "cycle_time_s": 10}, False),
    ("speed_distance_time", {"conveyor_speed_mm_s": 100, "transfer_time_s": 10, "travel_distance_mm": 900}, True),
    ("speed_distance_time", {"conveyor_speed_mm_s": 50, "transfer_time_s": 10, "travel_distance_mm": 600}, False),
    ("available_width_clearance", {"available_width_mm": 600, "workpiece_width_mm": 400, "side_clearance_mm": 50}, True),
    ("available_width_clearance", {"available_width_mm": 450, "workpiece_width_mm": 400, "side_clearance_mm": 50}, False),
    ("available_length_travel", {"available_length_mm": 2000, "travel_distance_mm": 1000, "workpiece_length_mm": 500}, True),
    ("available_length_travel", {"available_length_mm": 1500, "travel_distance_mm": 1000, "workpiece_length_mm": 500}, False),
    ("available_height_clearance", {"available_height_mm": 1000, "workpiece_height_mm": 500}, True),
    ("available_height_clearance", {"available_height_mm": 550, "workpiece_height_mm": 500}, False),
    ("load_capacity_safety", {"load_capacity_kg": 100, "safety_factor": 2, "workpiece_mass_kg": 40}, True),
    ("load_capacity_safety", {"load_capacity_kg": 70, "safety_factor": 2, "workpiece_mass_kg": 40}, False),
    ("repeatability_not_worse_than_accuracy", {"repeatability_mm": 0.3, "positioning_accuracy_mm": 0.5}, True),
    ("repeatability_not_worse_than_accuracy", {"repeatability_mm": 0.8, "positioning_accuracy_mm": 0.5}, False),
    ("input_interface_match", {"upstream_output_interface": "belt", "station_input_interface": "belt"}, True),
    ("input_interface_match", {"upstream_output_interface": "belt", "station_input_interface": "roller"}, False),
    ("output_interface_match", {"station_output_interface": "pallet", "downstream_input_interface": "pallet"}, True),
    ("output_interface_match", {"station_output_interface": "pallet", "downstream_input_interface": "belt"}, False),
    ("workpiece_fits_station_width", {"available_width_mm": 500, "workpiece_width_mm": 400}, True),
    ("workpiece_fits_station_width", {"available_width_mm": 300, "workpiece_width_mm": 400}, False),
    ("workpiece_fits_station_height", {"available_height_mm": 600, "workpiece_height_mm": 500}, True),
    ("workpiece_fits_station_height", {"available_height_mm": 400, "workpiece_height_mm": 500}, False),
    ("positive_throughput", {"target_throughput_per_hour": 100}, True),
    ("positive_throughput", {"target_throughput_per_hour": -1}, False),
    ("positive_travel_distance", {"travel_distance_mm": 100}, True),
    ("positive_travel_distance", {"travel_distance_mm": 0}, False),
    ("positive_transfer_time", {"transfer_time_s": 1}, True),
    ("positive_transfer_time", {"transfer_time_s": 0}, False),
    ("working_height_ergonomic_range", {"working_height_mm": 900}, True),
    ("working_height_ergonomic_range", {"working_height_mm": 650}, False),
    ("safety_factor_minimum", {"safety_factor": 2}, True),
    ("safety_factor_minimum", {"safety_factor": 1.2}, False),
    ("speed_upper_bound_for_fragile_workpiece", {"workpiece_fragile": True, "conveyor_speed_mm_s": 400}, True),
    ("speed_upper_bound_for_fragile_workpiece", {"workpiece_fragile": True, "conveyor_speed_mm_s": 600}, False),
]


@pytest.mark.parametrize("rule_id,assignments,expected_sat", RULE_CASES)
def test_each_rule_has_satisfying_and_violating_example(
    engine: ConstraintEngine,
    rule_id: str,
    assignments: dict[str, float | str | bool],
    expected_sat: bool,
):
    rule = next(item for item in engine.rules if item.rule_id == rule_id)
    solver = z3.Solver()
    solver.add(rule.build(engine.variables))
    for field_id, value in assignments.items():
        solver.add(engine.encode_equality(field_id, value))

    assert (solver.check() == z3.sat) is expected_sat

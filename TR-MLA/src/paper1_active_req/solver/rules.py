from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import z3


VariableMap = dict[str, z3.ExprRef]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    scope: tuple[str, ...]
    build: Callable[[VariableMap], z3.BoolRef]


def build_rules() -> list[Rule]:
    return [
        Rule(
            "cycle_throughput_relation",
            ("cycle_time_s", "target_throughput_per_hour"),
            lambda v: (
                v["cycle_time_s"] * v["target_throughput_per_hour"]
                <= 3600
            ),
        ),
        Rule(
            "transfer_within_cycle",
            ("transfer_time_s", "cycle_time_s"),
            lambda v: v["transfer_time_s"] <= v["cycle_time_s"],
        ),
        Rule(
            "speed_distance_time",
            (
                "conveyor_speed_mm_s",
                "transfer_time_s",
                "travel_distance_mm",
            ),
            lambda v: (
                v["conveyor_speed_mm_s"] * v["transfer_time_s"]
                >= v["travel_distance_mm"]
            ),
        ),
        Rule(
            "available_width_clearance",
            (
                "available_width_mm",
                "workpiece_width_mm",
                "side_clearance_mm",
            ),
            lambda v: (
                v["available_width_mm"]
                >= v["workpiece_width_mm"] + 2 * v["side_clearance_mm"]
            ),
        ),
        Rule(
            "available_length_travel",
            (
                "available_length_mm",
                "travel_distance_mm",
                "workpiece_length_mm",
            ),
            lambda v: (
                v["available_length_mm"]
                >= v["travel_distance_mm"] + v["workpiece_length_mm"] + 100
            ),
        ),
        Rule(
            "available_height_clearance",
            ("available_height_mm", "workpiece_height_mm"),
            lambda v: (
                v["available_height_mm"] >= v["workpiece_height_mm"] + 100
            ),
        ),
        Rule(
            "load_capacity_safety",
            ("load_capacity_kg", "safety_factor", "workpiece_mass_kg"),
            lambda v: (
                v["load_capacity_kg"]
                >= v["safety_factor"] * v["workpiece_mass_kg"]
            ),
        ),
        Rule(
            "repeatability_not_worse_than_accuracy",
            ("repeatability_mm", "positioning_accuracy_mm"),
            lambda v: (
                v["repeatability_mm"] <= v["positioning_accuracy_mm"]
            ),
        ),
        Rule(
            "input_interface_match",
            ("upstream_output_interface", "station_input_interface"),
            lambda v: (
                v["upstream_output_interface"] == v["station_input_interface"]
            ),
        ),
        Rule(
            "output_interface_match",
            ("station_output_interface", "downstream_input_interface"),
            lambda v: (
                v["station_output_interface"] == v["downstream_input_interface"]
            ),
        ),
        Rule(
            "workpiece_fits_station_width",
            ("available_width_mm", "workpiece_width_mm"),
            lambda v: v["available_width_mm"] >= v["workpiece_width_mm"],
        ),
        Rule(
            "workpiece_fits_station_height",
            ("available_height_mm", "workpiece_height_mm"),
            lambda v: v["available_height_mm"] >= v["workpiece_height_mm"],
        ),
        Rule(
            "positive_throughput",
            ("target_throughput_per_hour",),
            lambda v: v["target_throughput_per_hour"] > 0,
        ),
        Rule(
            "positive_travel_distance",
            ("travel_distance_mm",),
            lambda v: v["travel_distance_mm"] > 0,
        ),
        Rule(
            "positive_transfer_time",
            ("transfer_time_s",),
            lambda v: v["transfer_time_s"] > 0,
        ),
        Rule(
            "working_height_ergonomic_range",
            ("working_height_mm",),
            lambda v: z3.And(
                v["working_height_mm"] >= 700,
                v["working_height_mm"] <= 1100,
            ),
        ),
        Rule(
            "safety_factor_minimum",
            ("safety_factor",),
            lambda v: v["safety_factor"] >= 1.5,
        ),
        Rule(
            "speed_upper_bound_for_fragile_workpiece",
            ("workpiece_fragile", "conveyor_speed_mm_s"),
            lambda v: z3.Implies(
                v["workpiece_fragile"], v["conveyor_speed_mm_s"] <= 500
            ),
        ),
    ]

"""确定性双场景合成零部件目录。"""

from __future__ import annotations

import random

from .schema import ComponentTemplate, EvidenceRecord, TypedPort

DEFAULT_SEED = 20260816
SCENE_ROLES: dict[str, tuple[str, ...]] = {
    "A": ("drive", "transmission", "conveyor", "positioner"),
    "B": ("motor", "gearbox", "coupling", "shaft_support"),
}
TEMPLATES_PER_ROLE = 20


def _number(rng: random.Random, low: float, high: float, step: float) -> float:
    count = int(round((high - low) / step))
    return round(low + rng.randrange(count + 1) * step, 4)


def _ports(scene: str, role: str) -> tuple[TypedPort, ...]:
    if scene == "A":
        return (
            TypedPort(f"{role}-mechanical-in", "mechanical", "in", "N.m"),
            TypedPort(f"{role}-mechanical-out", "mechanical", "out", "N.m"),
        )
    return (
        TypedPort(f"{role}-shaft-in", "shaft", "in", "mm"),
        TypedPort(f"{role}-shaft-out", "shaft", "out", "mm"),
    )


def _parameters(scene: str, role: str, rng: random.Random) -> tuple[dict[str, float], dict[str, str]]:
    if scene == "A":
        options = {
            "drive": {
                "speed_rpm": _number(rng, 800, 2800, 100),
                "torque_nm": _number(rng, 80, 420, 20),
                "capacity_kg": _number(rng, 80, 520, 20),
            },
            "transmission": {
                "ratio": _number(rng, 1, 8, 0.5),
                "torque_nm": _number(rng, 80, 500, 20),
                "speed_rpm": _number(rng, 500, 2600, 100),
            },
            "conveyor": {
                "speed_mps": _number(rng, 0.2, 2.0, 0.1),
                "load_kg": _number(rng, 40, 420, 20),
                "width_mm": _number(rng, 200, 1000, 50),
            },
            "positioner": {
                "payload_kg": _number(rng, 40, 520, 20),
                "stroke_mm": _number(rng, 100, 800, 50),
                "repeatability_mm": _number(rng, 0.05, 0.5, 0.05),
            },
        }[role]
        units = {
            "speed_rpm": "rpm", "torque_nm": "N.m", "capacity_kg": "kg",
            "ratio": "ratio", "speed_mps": "m/s", "load_kg": "kg",
            "width_mm": "mm", "payload_kg": "kg", "stroke_mm": "mm",
            "repeatability_mm": "mm",
        }
    else:
        options = {
            "motor": {
                "speed_rpm": _number(rng, 900, 3000, 100),
                "torque_nm": _number(rng, 40, 300, 10),
                "shaft_diameter_mm": _number(rng, 10, 50, 2),
            },
            "gearbox": {
                "ratio": _number(rng, 2, 12, 0.5),
                "torque_nm": _number(rng, 80, 700, 20),
                "input_speed_rpm": _number(rng, 700, 3000, 100),
            },
            "coupling": {
                "bore_mm": _number(rng, 10, 50, 2),
                "torque_nm": _number(rng, 80, 700, 20),
                "length_mm": _number(rng, 40, 180, 10),
            },
            "shaft_support": {
                "bore_mm": _number(rng, 10, 50, 2),
                "speed_rpm": _number(rng, 700, 3000, 100),
                "flange_diameter_mm": _number(rng, 40, 160, 10),
            },
        }[role]
        units = {
            "speed_rpm": "rpm", "torque_nm": "N.m", "shaft_diameter_mm": "mm",
            "ratio": "ratio", "input_speed_rpm": "rpm", "bore_mm": "mm",
            "length_mm": "mm", "flange_diameter_mm": "mm",
        }
    return options, {name: units[name] for name in options}


def build_catalog(seed: int = DEFAULT_SEED) -> list[ComponentTemplate]:
    """返回固定规模、只含合成参数的 A/B 目录。"""
    rng = random.Random(seed)
    catalog: list[ComponentTemplate] = []
    for scene, roles in SCENE_ROLES.items():
        for role in roles:
            for index in range(1, TEMPLATES_PER_ROLE + 1):
                parameters, units = _parameters(scene, role, rng)
                catalog.append(
                    ComponentTemplate(
                        template_id=f"{scene}-{role.upper()}-{index:03d}",
                        scene=scene,
                        role=role,
                        parameters=parameters,
                        parameter_units=units,
                        ports=_ports(scene, role),
                        constraints=(f"scene={scene}", f"role={role}", "typed-ports-required"),
                        evidence=EvidenceRecord(
                            source_seed=seed,
                            derived_rule="bounded deterministic synthetic parameter generator",
                            synthetic=True,
                            unit="SI-or-declared-domain-unit",
                            version="catalog-v1",
                        ),
                    )
                )
    return catalog

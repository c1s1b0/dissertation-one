from __future__ import annotations

from pathlib import Path


def _components(path: Path) -> set[str]:
    return {item.lower() for item in path.resolve(strict=False).parts}


def validate_v3_input_paths(*paths: Path) -> None:
    if any(_components(path).intersection({"v1", "v2", "formal"}) for path in paths):
        raise ValueError("v3 paths only: v1/v2/formal input paths are not allowed")


def validate_v3_output_path(path: Path, *, formal: bool = False) -> None:
    components = _components(path)
    if components.intersection({"v1", "v2"}):
        raise ValueError("v3 paths only: v1/v2 output paths are not allowed")
    if ("formal" in components) != formal:
        raise ValueError("v3 paths only: formal output role does not match path")


def validate_v3_paths(*paths: Path, allow_formal: bool = False) -> None:
    if not paths:
        return
    validate_v3_input_paths(*paths[:-1])
    validate_v3_output_path(paths[-1], formal=allow_formal)

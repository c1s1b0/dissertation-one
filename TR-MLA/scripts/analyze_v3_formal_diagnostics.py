"""只读导出冻结 v3 formal 结果的诊断证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from typing import Any

import pandas as pd

from paper1_active_req.domain.models import FieldDefinition, FieldKind, FieldStatus, FieldValue
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.v3_protocol import frozen_critical_field_ids
from paper1_active_req.experiment.v3_runner import V3ExperimentRecord
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel


EXPECTED_FORMAL_RECORD_COUNT = 1800
EXPECTED_TR_MLA_RECORD_COUNT = 180
DEFAULT_EPISODES_SHA256 = "3D219CCA3159C1C42790C58C385CEB2E7A8170A085D762B1C0760C7A20DBCBA5"


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"formal input is unreadable: {path}") from error


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable or invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return raw


def _read_jsonl(path: Path, model: type[Any], *, label: str) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    try:
        values = [model.model_validate_json(line) for line in lines if line.strip()]
    except ValueError as error:
        raise ValueError(f"{label} contains invalid JSONL: {path}") from error
    if not values:
        raise ValueError(f"{label} must not be empty: {path}")
    return values


def _require_succeeded_outcome(outcome_path: Path) -> dict[str, Any]:
    if not outcome_path.is_file():
        raise ValueError(f"formal outcome is missing: {outcome_path}")
    outcome = _read_json(outcome_path, label="formal outcome")
    if outcome.get("status") != "succeeded":
        raise ValueError("formal outcome must have status=succeeded")
    for field_name in ("record_count", "audit_record_count", "completed_record_count"):
        if outcome.get(field_name) != EXPECTED_FORMAL_RECORD_COUNT:
            raise ValueError(
                f"formal outcome {field_name} must equal {EXPECTED_FORMAL_RECORD_COUNT}"
            )
    if outcome.get("pending_audit_record_count") != 0:
        raise ValueError("formal outcome must have pending_audit_record_count=0")
    return outcome


def _require_formal_roster(records: list[V3ExperimentRecord]) -> None:
    if len(records) != EXPECTED_FORMAL_RECORD_COUNT:
        raise ValueError(
            f"formal records must contain {EXPECTED_FORMAL_RECORD_COUNT} rows, got {len(records)}"
        )
    pairs = {(record.task_id, record.policy_name) for record in records}
    if len(pairs) != len(records):
        raise ValueError("formal records contain duplicate task-policy pairs")
    tr_mla_records = [record for record in records if record.policy_name == "tr_mla"]
    if len(tr_mla_records) != EXPECTED_TR_MLA_RECORD_COUNT:
        raise ValueError(
            "formal records must contain "
            f"{EXPECTED_TR_MLA_RECORD_COUNT} tr_mla rows, got {len(tr_mla_records)}"
        )


def _matches_truth(predicted: object, truth: FieldValue, definition: FieldDefinition) -> bool:
    if truth.value is None:
        return predicted is None
    if definition.kind in {FieldKind.CONTINUOUS, FieldKind.INTEGER}:
        if not isinstance(predicted, (int, float)) or not isinstance(truth.value, (int, float)):
            return False
        return isclose(float(predicted), float(truth.value), rel_tol=0.0, abs_tol=1e-9)
    return predicted == truth.value


def _precision(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def _audit_tr_mla_inferences(
    records: list[V3ExperimentRecord],
    tasks: list[V3RequirementTask],
    training_specs: list[CompleteSpec],
    schema: dict[str, FieldDefinition],
    relation_config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != EXPECTED_TR_MLA_RECORD_COUNT:
        raise ValueError("released test tasks must contain 180 unique task IDs")

    model = TypedRelationalModel.fit(training_specs, schema, relation_config, use_relations=True)
    field_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for record in records:
        if record.policy_name != "tr_mla":
            continue
        task = task_by_id.get(record.task_id)
        if task is None:
            raise ValueError(f"formal record references unknown task: {record.task_id}")
        if record.episode.task_id != record.task_id or record.episode.policy_name != "tr_mla":
            raise ValueError(f"invalid episode binding for task: {record.task_id}")

        inferred = model.reliably_resolved(record.episode.final_state.public_copy())
        inferred_total = 0
        inferred_correct = 0
        for field_id in frozen_critical_field_ids(task.initial_state, schema):
            definition = schema[field_id]
            final_value = record.episode.final_state.values[field_id]
            truth = task.ground_truth[field_id]
            direct_correct = (
                final_value.status == FieldStatus.OBSERVED
                and _matches_truth(final_value.value, truth, definition)
            )
            if direct_correct or field_id not in inferred:
                continue
            correct = _matches_truth(inferred[field_id], truth, definition)
            inferred_total += 1
            inferred_correct += int(correct)
            field_rows.append(
                {
                    "task_id": record.task_id,
                    "spec_id": record.spec_id,
                    "missing_rate": record.missing_rate,
                    "mechanism": record.mechanism,
                    "pivot": record.pivot,
                    "field_id": field_id,
                    "field_kind": definition.kind.value,
                    "criticality": definition.criticality,
                    "source": "model_inference",
                    "correct": int(correct),
                    "exact_match_rule": "abs_tol=1e-9 for continuous/integer; equality otherwise",
                }
            )
        precision = _precision(inferred_correct, inferred_total)
        metric_precision = record.metrics.get("inference_precision")
        if metric_precision is None or not isclose(precision, float(metric_precision), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "reconstructed TR-MLA inference precision does not match frozen metric "
                f"for task {record.task_id}"
            )
        episode_rows.append(
            {
                "task_id": record.task_id,
                "spec_id": record.spec_id,
                "missing_rate": record.missing_rate,
                "mechanism": record.mechanism,
                "pivot": record.pivot,
                "episode_inferred_total": inferred_total,
                "episode_inferred_correct": inferred_correct,
                "episode_precision": precision,
            }
        )
    episode_frame = pd.DataFrame(episode_rows)
    field_frame = pd.DataFrame(field_rows)
    if len(episode_frame) != EXPECTED_TR_MLA_RECORD_COUNT:
        raise ValueError("TR-MLA episode audit did not process 180 records")
    if field_frame.empty:
        raise ValueError("TR-MLA formal audit contains no model inferences")
    return episode_frame, field_frame


def _summary_row(
    *,
    table: str,
    episode_frame: pd.DataFrame,
    field_frame: pd.DataFrame,
    **scope: object,
) -> dict[str, object]:
    inferred_total = int(episode_frame["episode_inferred_total"].sum())
    inferred_correct = int(episode_frame["episode_inferred_correct"].sum())
    return {
        "table": table,
        "missing_rate": scope.get("missing_rate"),
        "mechanism": scope.get("mechanism"),
        "pivot": scope.get("pivot"),
        "field_id": scope.get("field_id"),
        "field_kind": scope.get("field_kind"),
        "tr_mla_record_count": int(len(episode_frame)),
        "episode_with_inference_count": int((episode_frame["episode_inferred_total"] > 0).sum()),
        "inferred_total": inferred_total,
        "inferred_correct": inferred_correct,
        "micro_precision": _precision(inferred_correct, inferred_total),
        "episode_precision_mean": float(episode_frame["episode_precision"].mean()),
        "correct_rate_among_inference_rows": (
            float(field_frame["correct"].mean()) if not field_frame.empty else 0.0
        ),
        "analysis_label": "exploratory",
    }


def _field_summary_row(
    *,
    field_frame: pd.DataFrame,
    tr_mla_record_count: int,
    field_id: str,
) -> dict[str, object]:
    """按原子 inferred-field 行聚合，绝不回聚合整条 episode。"""
    inferred_total = int(len(field_frame))
    inferred_correct = int(field_frame["correct"].sum())
    micro_precision = _precision(inferred_correct, inferred_total)
    return {
        "table": "inference_by_field",
        "missing_rate": None,
        "mechanism": None,
        "pivot": None,
        "field_id": field_id,
        "field_kind": str(field_frame["field_kind"].iloc[0]),
        "tr_mla_record_count": tr_mla_record_count,
        "episode_with_inference_count": int(field_frame["task_id"].nunique()),
        "inferred_total": inferred_total,
        "inferred_correct": inferred_correct,
        "micro_precision": micro_precision,
        "episode_precision_mean": micro_precision,
        "correct_rate_among_inference_rows": micro_precision,
        "analysis_label": "exploratory",
    }


def _build_tables(episode_frame: pd.DataFrame, field_frame: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_row(table="inference_overall", episode_frame=episode_frame, field_frame=field_frame)]
    for column, table_name in (
        ("missing_rate", "inference_by_missing_rate"),
        ("mechanism", "inference_by_mechanism"),
        ("pivot", "inference_by_pivot"),
    ):
        for value, group in episode_frame.groupby(column, sort=True, dropna=False):
            field_group = field_frame[field_frame[column] == value]
            rows.append(
                _summary_row(
                    table=table_name,
                    episode_frame=group,
                    field_frame=field_group,
                    **{column: value.item() if hasattr(value, "item") else value},
                )
            )
    for field_id, group in field_frame.groupby("field_id", sort=True):
        rows.append(
            _field_summary_row(
                field_frame=group,
                tr_mla_record_count=len(episode_frame),
                field_id=field_id,
            )
        )
    return pd.DataFrame(rows)


def _require_fresh_output_dir(output_dir: Path, episodes_path: Path) -> None:
    resolved_output = output_dir.resolve(strict=False)
    formal_dir = episodes_path.resolve().parent
    if resolved_output.parent != formal_dir or resolved_output.name != "diagnostics":
        raise ValueError("diagnostic output_dir must be the formal/diagnostics sibling directory")
    if output_dir.exists():
        raise FileExistsError(f"diagnostic output_dir already exists: {output_dir}")


def _render_report(
    *,
    summary: dict[str, object],
    primary: dict[str, Any],
    outcome: dict[str, Any],
) -> str:
    comparison = primary["comparison"]
    return "\n".join(
        [
            "# v3 formal 推断诊断报告",
            "",
            "## 证据边界",
            "",
            "本报告为对冻结 v3 formal 输出的只读后验诊断（exploratory）。未运行、重写或覆盖任何 formal episode。",
            "",
            "## 完整性",
            "",
            f"- Formal records: {summary['formal_record_count']}",
            f"- TR-MLA records: {summary['tr_mla_record_count']}",
            f"- Audit outcome: {outcome['status']} (completed={outcome['completed_record_count']})",
            "",
            "## 确认性主结果（引用冻结 summary）",
            "",
            f"- TR-MLA vs Criticality-Greedy, CRR@8 mean difference: {comparison['mean_difference']:.8f}",
            f"- 95% bootstrap CI: [{comparison['ci_low']:.8f}, {comparison['ci_high']:.8f}]",
            f"- Wilcoxon p value: {comparison['p_value']:.6g}; rank-biserial: {comparison['rank_biserial']:.6f}",
            "",
            "## 探索性推断诊断",
            "",
            f"- Inferred fields: {summary['inferred_total']}",
            f"- Correct inferred fields: {summary['inferred_correct']}",
            f"- Micro precision: {summary['micro_precision']:.8f}",
            f"- Episode-mean precision: {summary['episode_precision_mean']:.8f}",
            "",
            "连续变量遵循冻结指标的 absolute tolerance=1e-9；上述推断诊断不得替代确认性主比较，也不得用于宣称可靠推断已经验证。",
            "",
        ]
    )


def _write_outputs(
    *,
    output_dir: Path,
    tables: pd.DataFrame,
    provenance: dict[str, object],
    report: str,
) -> None:
    output_dir.mkdir(parents=False, exist_ok=False)
    tables.to_csv(output_dir / "formal_diagnostic_tables.csv", index=False)
    (output_dir / "formal_diagnostic_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "formal_diagnostic_report.md").write_text(report, encoding="utf-8")


def run_diagnostics(
    *,
    episodes_path: Path,
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    relation_config_path: Path,
    outcome_path: Path,
    primary_summary_path: Path,
    output_dir: Path,
    expected_episodes_sha256: str,
) -> dict[str, object]:
    """复核冻结 v3 formal 输入，并导出不可覆盖的派生诊断文件。"""
    actual_hash = _sha256(episodes_path)
    if actual_hash.lower() != expected_episodes_sha256.lower():
        raise ValueError(
            "episodes SHA-256 mismatch: "
            f"expected {expected_episodes_sha256}, got {actual_hash}"
        )
    outcome = _require_succeeded_outcome(outcome_path)
    _require_fresh_output_dir(output_dir, episodes_path)
    records = _read_jsonl(episodes_path, V3ExperimentRecord, label="formal records")
    _require_formal_roster(records)
    tasks = _read_jsonl(tasks_path, V3RequirementTask, label="released test tasks")
    training_specs = _read_jsonl(training_specs_path, CompleteSpec, label="training specs")
    schema = load_field_schema(schema_path)
    relation_config = load_v3_relation_config(relation_config_path)
    primary = _read_json(primary_summary_path, label="confirmatory primary summary")
    if primary.get("analysis_label") != "confirmatory" or not isinstance(primary.get("comparison"), dict):
        raise ValueError("confirmatory primary summary has an invalid protocol label")

    episode_frame, field_frame = _audit_tr_mla_inferences(
        records, tasks, training_specs, schema, relation_config
    )
    tables = _build_tables(episode_frame, field_frame)
    inferred_total = int(episode_frame["episode_inferred_total"].sum())
    inferred_correct = int(episode_frame["episode_inferred_correct"].sum())
    summary: dict[str, object] = {
        "formal_record_count": len(records),
        "tr_mla_record_count": len(episode_frame),
        "inferred_total": inferred_total,
        "inferred_correct": inferred_correct,
        "micro_precision": _precision(inferred_correct, inferred_total),
        "episode_precision_mean": float(episode_frame["episode_precision"].mean()),
    }
    provenance = {
        "analysis_label": "exploratory",
        "formal_execution_performed": False,
        "formal_record_count": len(records),
        "tr_mla_record_count": len(episode_frame),
        "episodes_sha256": actual_hash.upper(),
        "input_sha256": {
            "episodes": actual_hash.upper(),
            "tasks": _sha256(tasks_path).upper(),
            "training_specs": _sha256(training_specs_path).upper(),
            "schema": _sha256(schema_path).upper(),
            "relation_config": _sha256(relation_config_path).upper(),
            "outcome": _sha256(outcome_path).upper(),
            "primary_summary": _sha256(primary_summary_path).upper(),
        },
        "script_sha256": _sha256(Path(__file__)).upper(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_metric_rule": "abs_tol=1e-9 for continuous/integer; equality otherwise",
    }
    _write_outputs(
        output_dir=output_dir,
        tables=tables,
        provenance=provenance,
        report=_render_report(summary=summary, primary=primary, outcome=outcome),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-path", type=Path, required=True)
    parser.add_argument("--tasks-path", type=Path, required=True)
    parser.add_argument("--training-specs-path", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--relation-config-path", type=Path, required=True)
    parser.add_argument("--outcome-path", type=Path, required=True)
    parser.add_argument("--primary-summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes-sha256", default=DEFAULT_EPISODES_SHA256)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    print(json.dumps(run_diagnostics(**vars(arguments)), ensure_ascii=False, indent=2, sort_keys=True))

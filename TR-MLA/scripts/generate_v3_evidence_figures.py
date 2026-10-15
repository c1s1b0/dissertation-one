"""Build traceable manuscript figures from frozen v3 formal artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from paper1_active_req.experiment.v3_runner import V3ExperimentRecord
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORMAL_POLICY_ROSTER = frozenset(
    {
        "random",
        "fixed_questionnaire",
        "criticality_greedy",
        "uncertainty_afa",
        "constraint_range_only",
        "ccla_one_step",
        "v2_style_two_step_unconditioned",
        "marginal_lookahead_no_relations",
        "typed_relational_myopic",
        "tr_mla",
    }
)
BEHAVIOUR_POLICY_NAMES = (
    "criticality_greedy",
    "marginal_lookahead_no_relations",
    "tr_mla",
    "typed_relational_myopic",
)
MECHANISM_ORDER = (
    "conditional_tolerance",
    "continuous_clearance",
    "interface_compatibility",
)

def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required evidence input is unavailable: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Return content hashes for the exact inputs recorded in a figure provenance file."""
    if not paths or any(not label for label in paths):
        raise ValueError("evidence inputs require non-empty labels")
    return {label: _sha256(path) for label, path in paths.items()}


def build_primary_paired_effect_data(primary_csv: Path, primary_json: Path) -> pd.DataFrame:
    """Reconstruct the frozen 60-spec TR-MLA minus Criticality-Greedy endpoint effect."""
    primary = pd.read_csv(primary_csv)
    required = {
        "spec_id",
        "method",
        "crr_at_8",
        "primary_missing_rates",
        "missing_rate_scope",
    }
    missing = required.difference(primary.columns)
    if missing:
        raise ValueError(f"primary evidence is missing columns: {sorted(missing)}")
    if not primary["primary_missing_rates"].eq("m50,m70").all() or not primary[
        "missing_rate_scope"
    ].eq("m50_m70_mean").all():
        raise ValueError("primary evidence must contain only frozen m50/m70 spec means")
    compared = primary.loc[
        primary["method"].isin(("tr_mla", "criticality_greedy")),
        ["spec_id", "method", "crr_at_8"],
    ]
    if compared.duplicated(["spec_id", "method"]).any():
        raise ValueError("primary evidence contains duplicate spec/method pairs")
    pairs = compared.pivot(index="spec_id", columns="method", values="crr_at_8")
    if set(pairs.columns) != {"tr_mla", "criticality_greedy"} or pairs.isna().any().any():
        raise ValueError("primary evidence requires complete TR-MLA and Criticality-Greedy pairs")
    if len(pairs) != 60:
        raise ValueError("primary evidence requires exactly 60 specification pairs")
    summary = json.loads(primary_json.read_text(encoding="utf-8"))
    comparison = summary.get("comparison") if isinstance(summary, dict) else None
    if (
        summary.get("analysis_label") != "confirmatory"
        or not isinstance(comparison, dict)
        or comparison.get("n_specs") != 60
    ):
        raise ValueError("primary summary is not a valid frozen confirmatory result")
    table = pd.DataFrame(
        {
            "spec_id": pairs.index,
            "delta_crr_at_8": pairs["tr_mla"] - pairs["criticality_greedy"],
        }
    ).sort_values("delta_crr_at_8", kind="stable").reset_index(drop=True)
    if abs(float(table["delta_crr_at_8"].mean()) - float(comparison["mean_difference"])) > 1e-12:
        raise ValueError("primary effect does not match the frozen confirmatory summary")
    return table


def build_operating_profile_data(overall_csv: Path) -> pd.DataFrame:
    """Return the complete ten-policy CRR@8 and mean answer-cost profile."""
    overall = pd.read_csv(overall_csv)
    required = {"policy_name", "crr_at_8", "answer_cost"}
    missing = required.difference(overall.columns)
    if missing:
        raise ValueError(f"operating profile is missing columns: {sorted(missing)}")
    table = overall.loc[:, ["policy_name", "crr_at_8", "answer_cost"]].copy()
    if table["policy_name"].duplicated().any() or set(table["policy_name"]) != FORMAL_POLICY_ROSTER:
        raise ValueError("operating profile requires the complete ten-policy formal roster")
    if table[["crr_at_8", "answer_cost"]].isna().any().any():
        raise ValueError("operating profile contains missing numeric metrics")
    return table.sort_values("policy_name", kind="stable").reset_index(drop=True)


def _read_jsonl(path: Path, model: type[V3ExperimentRecord] | type[V3RequirementTask]) -> list[object]:
    if not path.is_file():
        raise ValueError(f"required evidence input is unavailable: {path}")
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_policy_behaviour_table(episodes_path: Path, tasks_path: Path) -> pd.DataFrame:
    """Reconstruct first-two-step pivot selection and CRR@8 by policy and mechanism."""
    tasks = _read_jsonl(tasks_path, V3RequirementTask)
    typed_tasks = [task for task in tasks if isinstance(task, V3RequirementTask)]
    tasks_by_id = {task.task_id: task for task in typed_tasks}
    if len(tasks_by_id) != 180 or len(tasks_by_id) != len(typed_tasks):
        raise ValueError("released task evidence requires exactly 180 unique test tasks")
    records = _read_jsonl(episodes_path, V3ExperimentRecord)
    typed_records = [record for record in records if isinstance(record, V3ExperimentRecord)]
    selected_records = [
        record for record in typed_records if record.policy_name in BEHAVIOUR_POLICY_NAMES
    ]
    keys = {(record.task_id, record.policy_name) for record in selected_records}
    if len(selected_records) != 180 * len(BEHAVIOUR_POLICY_NAMES) or len(keys) != len(selected_records):
        raise ValueError("behaviour evidence requires one record per selected policy and test task")
    rows: list[dict[str, object]] = []
    for record in selected_records:
        task = tasks_by_id.get(record.task_id)
        if task is None or record.split != "test":
            raise ValueError("behaviour record has no matching released test task")
        if (
            record.spec_id != task.spec_id
            or record.mechanism != task.audit.mechanism
            or record.pivot != (task.audit.pivot_field_id is not None)
        ):
            raise ValueError("behaviour record does not match released task metadata")
        pivot_field_id = task.audit.pivot_field_id
        selected_first_two = bool(
            pivot_field_id is not None
            and pivot_field_id in {step.field_id for step in record.episode.steps[:2]}
        )
        rows.append(
            {
                "policy_name": record.policy_name,
                "mechanism": record.mechanism,
                "pivot": record.pivot,
                "selected_first_two": selected_first_two,
                "crr_at_8": float(record.metrics["crr_at_8"]),
            }
        )
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for policy_name in BEHAVIOUR_POLICY_NAMES:
        for mechanism in MECHANISM_ORDER:
            cell = frame.loc[
                (frame["policy_name"] == policy_name) & (frame["mechanism"] == mechanism)
            ]
            pivot_cell = cell.loc[cell["pivot"]]
            if len(cell) != 60 or len(pivot_cell) != 45:
                raise ValueError("behaviour evidence does not match the frozen mechanism/pivot task counts")
            summary_rows.append(
                {
                    "policy_name": policy_name,
                    "mechanism": mechanism,
                    "pivot_task_count": len(pivot_cell),
                    "all_task_count": len(cell),
                    "pivot_selected_first_two_rate": float(pivot_cell["selected_first_two"].mean()),
                    "crr_at_8_mean": float(cell["crr_at_8"].mean()),
                }
            )
    return pd.DataFrame(summary_rows)


def _require_succeeded_formal_outcome(path: Path) -> None:
    outcome = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(outcome, dict)
        or outcome.get("status") != "succeeded"
        or outcome.get("record_count") != 1800
        or outcome.get("completed_record_count") != 1800
    ):
        raise ValueError("evidence figures require the completed frozen formal outcome")


def _save_figure(figure: plt.Figure, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(pdf_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_provenance(
    *,
    pdf_path: Path,
    analysis_label: str,
    data_rule: str,
    input_paths: dict[str, Path],
) -> None:
    payload = {
        "analysis_label": analysis_label,
        "data_rule": data_rule,
        "input_sha256": input_hashes(input_paths),
        "output_file": pdf_path.name,
    }
    pdf_path.with_suffix(".provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_primary_paired_effect(table: pd.DataFrame, comparison: dict[str, object]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    ranks = np.arange(1, len(table) + 1)
    mean = float(comparison["mean_difference"])
    ci_low = float(comparison["ci_low"])
    ci_high = float(comparison["ci_high"])
    axis.axvspan(ci_low, ci_high, color="#0072B2", alpha=0.13, label="Formal 95% bootstrap CI")
    axis.axvline(0.0, color="#3B3B3B", linestyle="--", linewidth=1.0, label="No difference")
    axis.axvline(mean, color="#0072B2", linewidth=2.2, label="Formal mean difference")
    axis.scatter(
        table["delta_crr_at_8"],
        ranks,
        s=20,
        color="#56B4E9",
        edgecolor="#0072B2",
        linewidth=0.45,
        zorder=3,
    )
    axis.set_xlim(min(-0.02, float(table["delta_crr_at_8"].min()) - 0.01), max(0.10, float(table["delta_crr_at_8"].max()) + 0.01))
    axis.set_ylim(0, len(table) + 2)
    axis.set_xlabel(r"Paired $\Delta$ CRR@8 (TR-MLA $-$ Criticality-Greedy)")
    axis.set_ylabel("Specification rank (sorted by paired effect)")
    axis.text(
        0.99,
        0.02,
        f"Confirmatory; n=60\nmean={mean:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#9AA0A6"},
    )
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left", fontsize=8.3)
    figure.tight_layout()
    return figure


def _render_operating_profile(table: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    highlighted = {
        "criticality_greedy": ("Criticality-Greedy", "#3B3B3B", (7, -18)),
        "marginal_lookahead_no_relations": ("No-relations\nlookahead", "#7F8C8D", (7, 10)),
        "tr_mla": ("TR-MLA", "#0072B2", (5, 8)),
        "typed_relational_myopic": ("Relational-myopic", "#D55E00", (5, 8)),
    }
    for row in table.itertuples(index=False):
        if row.policy_name in highlighted:
            label, color, offset = highlighted[row.policy_name]
            axis.scatter(row.answer_cost, row.crr_at_8, s=72, color=color, zorder=3)
            axis.annotate(
                label,
                (row.answer_cost, row.crr_at_8),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.9,
                color=color,
                weight="bold" if row.policy_name == "tr_mla" else "normal",
            )
        else:
            axis.scatter(row.answer_cost, row.crr_at_8, s=38, color="#B8B8B8", zorder=2)
    axis.set_xlabel("Mean answer cost")
    axis.set_ylabel("CRR@8")
    axis.set_ylim(0.70, 0.92)
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.01,
        0.02,
        "Descriptive operating profile from one frozen benchmark",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4A4A4A",
    )
    figure.tight_layout()
    return figure


def _render_behaviour_heatmap(table: pd.DataFrame) -> plt.Figure:
    policy_labels = {
        "criticality_greedy": "Criticality-Greedy",
        "marginal_lookahead_no_relations": "No-relations\nlookahead",
        "tr_mla": "TR-MLA",
        "typed_relational_myopic": "Relational-myopic",
    }
    mechanism_labels = [
        "Conditional\ntolerance",
        "Continuous\nclearance",
        "Interface\ncompatibility",
    ]
    pivot_matrix = np.array(
        [
            [
                table.loc[
                    (table["policy_name"] == policy) & (table["mechanism"] == mechanism),
                    "pivot_selected_first_two_rate",
                ].iloc[0]
                for mechanism in MECHANISM_ORDER
            ]
            for policy in BEHAVIOUR_POLICY_NAMES
        ]
    )
    crr_matrix = np.array(
        [
            [
                table.loc[
                    (table["policy_name"] == policy) & (table["mechanism"] == mechanism),
                    "crr_at_8_mean",
                ].iloc[0]
                for mechanism in MECHANISM_ORDER
            ]
            for policy in BEHAVIOUR_POLICY_NAMES
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 4.25), sharey=True)
    panels = (
        (axes[0], pivot_matrix, "Pivot selected in first two queries", "Blues"),
        (axes[1], crr_matrix, "Mean CRR@8", "YlOrRd"),
    )
    for axis, matrix, title, cmap in panels:
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
        axis.set_title(title, fontsize=9.6)
        axis.set_xticks(range(3), mechanism_labels, fontsize=8.1)
        axis.set_yticks(range(4), [policy_labels[name] for name in BEHAVIOUR_POLICY_NAMES], fontsize=8.5)
        for row_index, column_index in np.ndindex(matrix.shape):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if value > 0.55 else "#202020",
            )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.03)
        colorbar.ax.tick_params(labelsize=7.5)
    figure.text(
        0.5,
        0.01,
        "Exploratory descriptive analysis; left: 45 pivot tasks/mechanism, right: 60 tasks/mechanism",
        ha="center",
        fontsize=8.0,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.1)
    return figure


def render_evidence_figures(
    *,
    formal_root: Path,
    generated_root: Path,
    manuscript_root: Path,
) -> dict[str, Path]:
    """Render the three approved evidence figures without modifying formal inputs."""
    primary_csv = formal_root / "summary" / "primary_m50_m70.csv"
    primary_json = formal_root / "summary" / "confirmatory_primary.json"
    overall_csv = formal_root / "summary" / "overall.csv"
    episodes_path = formal_root / "episodes.jsonl"
    outcome_path = formal_root / "episodes.jsonl.formal-attempt.json.audit" / "outcome.json"
    tasks_path = generated_root / "test_tasks.jsonl"
    _require_succeeded_formal_outcome(outcome_path)
    primary = build_primary_paired_effect_data(primary_csv, primary_json)
    overall = build_operating_profile_data(overall_csv)
    behaviour = build_policy_behaviour_table(episodes_path, tasks_path)
    comparison = json.loads(primary_json.read_text(encoding="utf-8"))["comparison"]
    common_inputs = {"formal_outcome": outcome_path, "formal_episodes": episodes_path}
    figure_paths = {
        "fig_primary_paired_effect": manuscript_root / "figures" / "fig_primary_paired_effect.pdf",
        "fig_operating_profile": manuscript_root / "figures" / "fig_operating_profile.pdf",
        "figS1_policy_behaviour_heatmap": manuscript_root
        / "supplementary"
        / "figS1_policy_behaviour_heatmap.pdf",
    }
    _save_figure(_render_primary_paired_effect(primary, comparison), figure_paths["fig_primary_paired_effect"])
    _write_provenance(
        pdf_path=figure_paths["fig_primary_paired_effect"],
        analysis_label="confirmatory",
        data_rule="60 specification-level m50/m70 means; TR-MLA minus Criticality-Greedy",
        input_paths={**common_inputs, "primary_csv": primary_csv, "primary_summary": primary_json},
    )
    _save_figure(_render_operating_profile(overall), figure_paths["fig_operating_profile"])
    _write_provenance(
        pdf_path=figure_paths["fig_operating_profile"],
        analysis_label="descriptive",
        data_rule="all ten formal policies; mean CRR@8 versus mean answer cost",
        input_paths={**common_inputs, "overall_csv": overall_csv},
    )
    _save_figure(_render_behaviour_heatmap(behaviour), figure_paths["figS1_policy_behaviour_heatmap"])
    _write_provenance(
        pdf_path=figure_paths["figS1_policy_behaviour_heatmap"],
        analysis_label="exploratory",
        data_rule="episode steps 0-1 for pivot tasks; CRR@8 over all released test tasks",
        input_paths={**common_inputs, "release_tasks": tasks_path},
    )
    return figure_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps({name: str(path) for name, path in render_evidence_figures(**vars(_parse_args())).items()}, ensure_ascii=False, indent=2))

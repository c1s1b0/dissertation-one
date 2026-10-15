from __future__ import annotations

import pandas as pd
import pytest


def _records(*, difference: float = 0.2) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(60):
        spec_id = f"spec-{index:02d}"
        baseline = 0.20 + (index % 3) * 0.01
        method = baseline + difference
        for missing_rate in (0.3, 0.5, 0.7):
            records.append(
                {
                    "spec_id": spec_id,
                    "method": "criticality_greedy",
                    "missing_rate": missing_rate,
                    "crr_at_8": baseline if missing_rate != 0.3 else 0.99,
                }
            )
            records.append(
                {
                    "spec_id": spec_id,
                    "method": "tr_mla",
                    "missing_rate": missing_rate,
                    "crr_at_8": method if missing_rate != 0.3 else 0.01,
                }
            )
    return records


def test_aggregate_primary_by_spec_excludes_m30_and_averages_m50_m70():
    """若把 m30 混进主终点或未先规格内平均，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import aggregate_primary_by_spec

    aggregated = aggregate_primary_by_spec(_records())
    first_method = aggregated.loc[
        (aggregated["spec_id"] == "spec-00") & (aggregated["method"] == "tr_mla")
    ].iloc[0]

    assert len(aggregated) == 120
    assert first_method["crr_at_8"] == pytest.approx(0.40)
    assert set(aggregated["primary_missing_rates"]) == {"m50,m70"}
    assert set(aggregated["missing_rate_scope"]) == {"m50_m70_mean"}


def test_primary_comparison_uses_60_spec_pairs_not_120_task_pairs():
    """若对 m50/m70 任务伪重复配对，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import (
        aggregate_primary_by_spec,
        primary_paired_comparison,
    )

    comparison = primary_paired_comparison(_records())

    assert comparison.n_specs == 60
    assert comparison.mean_difference == pytest.approx(0.2)


def test_primary_comparison_is_reproducible_with_fixed_bootstrap_seed():
    """若 bootstrap 未固定 seed 或重复次数，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import (
        aggregate_primary_by_spec,
        primary_paired_comparison,
    )

    records = _records(difference=0.15)
    first = primary_paired_comparison(records)
    second = primary_paired_comparison(records)

    assert first == second
    assert first.bootstrap_replicates == 10_000
    assert first.seed == 20260806


def test_primary_comparison_handles_zero_differences_without_wilcoxon_error():
    """若全零差值使检验崩溃或效应量非零，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import (
        aggregate_primary_by_spec,
        primary_paired_comparison,
    )

    comparison = primary_paired_comparison(_records(difference=0.0))

    assert comparison.p_value == 1.0
    assert comparison.rank_biserial == 0.0
    assert comparison.ci_low == 0.0
    assert comparison.ci_high == 0.0


def test_primary_comparison_rejects_missing_spec_pairing():
    """若缺失一个方法-规格单元仍被静默分析，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import (
        aggregate_primary_by_spec,
        primary_paired_comparison,
    )

    incomplete = [
        record
        for record in _records()
        if not (
            record["spec_id"] == "spec-59"
            and record["method"] == "tr_mla"
            and record["missing_rate"] == 0.7
        )
    ]

    with pytest.raises(ValueError, match="requires both"):
        primary_paired_comparison(incomplete)


@pytest.mark.parametrize(
    ("missing_rate", "error_pattern"),
    ((0.3, "m50 and m70"), (0.5, "requires both")),
)
def test_primary_comparison_rejects_direct_unaggregated_records(
    missing_rate: float, error_pattern: str
):
    """若绕过 spec 聚合直接分析 m30 或单一缺失率，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import primary_paired_comparison

    raw = pd.DataFrame(
        record for record in _records() if record["missing_rate"] == missing_rate
    )

    with pytest.raises(ValueError, match=error_pattern):
        primary_paired_comparison(raw)


def test_primary_comparison_rejects_duplicate_raw_spec_policy_rate_rows():
    """若原始 records 重复一个 spec/policy/rate 单元仍可分析，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import primary_paired_comparison

    duplicate_cell = next(
        record
        for record in _records()
        if record["spec_id"] == "spec-00"
        and record["method"] == "criticality_greedy"
        and record["missing_rate"] == 0.5
    )
    duplicate = [*_records(), duplicate_cell]

    with pytest.raises(ValueError, match="must occur once"):
        primary_paired_comparison(duplicate)


def test_primary_comparison_rejects_scope_forged_frame_without_raw_rate_evidence():
    """若删去原始缺失率证据后伪造 scope 仍可主分析，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import (
        aggregate_primary_by_spec,
        primary_paired_comparison,
    )

    forged = aggregate_primary_by_spec(_records()).drop(
        columns=["primary_missing_rates"]
    )
    forged["missing_rate_scope"] = "m50_m70_mean"

    with pytest.raises(ValueError, match="missing_rate"):
        primary_paired_comparison(forged)


def test_secondary_comparisons_allow_three_exploratory_mechanisms_of_twenty_pairs():
    """若探索性机制统计错误套用主比较 60-pair 门，本测试应失败。"""
    from paper1_active_req.evaluation.v3_statistics import secondary_paired_comparisons

    records = pd.DataFrame(_records()).iloc[:0].copy()
    mechanism_records: list[dict[str, object]] = []
    for mechanism_index, mechanism in enumerate(
        ("interface_compatibility", "continuous_clearance", "conditional_tolerance")
    ):
        for index in range(20):
            spec_id = f"{mechanism}-{index:02d}"
            for missing_rate in (0.5, 0.7):
                mechanism_records.extend(
                    (
                        {
                            "spec_id": spec_id,
                            "method": "criticality_greedy",
                            "missing_rate": missing_rate,
                            "crr_at_8": 0.2,
                            "mechanism": mechanism,
                        },
                        {
                            "spec_id": spec_id,
                            "method": "tr_mla",
                            "missing_rate": missing_rate,
                            "crr_at_8": 0.2 + 0.01 * (mechanism_index + 1),
                            "mechanism": mechanism,
                        },
                    )
                )
    records = pd.DataFrame(mechanism_records)
    result = secondary_paired_comparisons(records)

    assert set(result["analysis_label"]) == {"exploratory"}
    assert set(result["n_specs"]) == {20}
    assert set(result["mechanism"]) == {
        "interface_compatibility",
        "continuous_clearance",
        "conditional_tolerance",
    }
    assert (result["holm_p_value"] >= result["p_value"]).all()

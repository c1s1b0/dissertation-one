from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


_PRIMARY_RATES = (0.5, 0.7)
_PRIMARY_RATE_LABEL = "m50,m70"
_PRIMARY_SCOPE_COLUMN = "missing_rate_scope"
_PRIMARY_SCOPE_VALUE = "m50_m70_mean"
_PRIMARY_SPEC_COUNT = 60
_BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True)
class PrimaryComparison:
    method: str
    baseline: str
    metric: str
    n_specs: int
    mean_difference: float
    ci_low: float
    ci_high: float
    wilcoxon_statistic: float
    p_value: float
    rank_biserial: float
    seed: int
    bootstrap_replicates: int = _BOOTSTRAP_REPLICATES


@dataclass(frozen=True)
class ExploratoryComparison:
    method: str
    baseline: str
    metric: str
    n_specs: int
    mean_difference: float
    wilcoxon_statistic: float
    p_value: float
    rank_biserial: float


def _as_frame(records: pd.DataFrame | Iterable[Mapping[str, object]]) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if "method" not in frame and "policy_name" in frame:
        frame = frame.rename(columns={"policy_name": "method"})
    required = {"spec_id", "method", "missing_rate", "crr_at_8"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"records missing required columns: {sorted(missing)}")
    return frame.copy()


def _normalise_missing_rate(value: object) -> float:
    if isinstance(value, str):
        labels = {"m30": 0.3, "m50": 0.5, "m70": 0.7}
        if value in labels:
            return labels[value]
    try:
        rate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid missing_rate: {value!r}") from error
    if rate not in {0.3, 0.5, 0.7}:
        raise ValueError(f"unsupported missing_rate: {value!r}")
    return rate


def aggregate_primary_by_spec(
    records: pd.DataFrame | Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """剔除 m30，并在每个 spec/method 内平均 m50 与 m70。"""
    frame = _as_frame(records)
    frame["_missing_rate"] = frame["missing_rate"].map(_normalise_missing_rate)
    primary = frame.loc[frame["_missing_rate"].isin(_PRIMARY_RATES)].copy()
    if primary.empty:
        raise ValueError("primary analysis requires m50 and m70 records")

    counts = primary.groupby(["spec_id", "method", "_missing_rate"], dropna=False).size()
    if (counts != 1).any():
        raise ValueError("each primary spec/method/missing-rate cell must occur once")
    observed_rates = primary.groupby(["spec_id", "method"])["_missing_rate"].agg(set)
    if not observed_rates.map(lambda rates: rates == set(_PRIMARY_RATES)).all():
        raise ValueError("each primary spec/method pair requires both m50 and m70")

    excluded = {"spec_id", "method", "missing_rate", "_missing_rate", "policy_name"}
    metric_columns = [
        column
        for column in primary.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(primary[column])
    ]
    if "crr_at_8" not in metric_columns:
        raise ValueError("crr_at_8 must be numeric")
    grouped = (
        primary.groupby(["spec_id", "method"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["spec_id", "method"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["primary_missing_rates"] = _PRIMARY_RATE_LABEL
    grouped[_PRIMARY_SCOPE_COLUMN] = _PRIMARY_SCOPE_VALUE
    return grouped


def _require_aggregated_m50_m70_frame(frame: pd.DataFrame) -> None:
    """确认输入保留了主终点聚合器产生的不可混淆范围标记。"""
    if _PRIMARY_SCOPE_COLUMN not in frame:
        raise ValueError("primary frame requires missing_rate_scope='m50_m70_mean'")
    if not frame[_PRIMARY_SCOPE_COLUMN].eq(_PRIMARY_SCOPE_VALUE).all():
        raise ValueError("primary frame requires missing_rate_scope='m50_m70_mean'")
    if "missing_rate" in frame:
        raise ValueError("primary frame must be aggregated before comparison")
    if frame.duplicated(["spec_id", "method"]).any():
        raise ValueError("primary frame has duplicate spec/method rows")


def _paired_differences(
    frame: pd.DataFrame,
    method: str,
    baseline: str,
    metric: str,
    expected_spec_count: int | None,
) -> np.ndarray:
    required = {"spec_id", "method", metric}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"primary frame missing required columns: {sorted(missing)}")
    subset = frame.loc[frame["method"].isin((method, baseline)), ["spec_id", "method", metric]]
    pivoted = subset.pivot(index="spec_id", columns="method", values=metric)
    if method not in pivoted or baseline not in pivoted:
        raise ValueError("paired methods are missing from primary frame")
    paired = pivoted[[method, baseline]].dropna()
    if len(paired) != len(pivoted):
        raise ValueError("paired comparison requires complete spec pairs")
    if expected_spec_count is not None and len(paired) != expected_spec_count:
        raise ValueError("primary comparison requires exactly 60 complete paired specs")
    if len(paired) == 0:
        raise ValueError("paired comparison requires at least one complete spec pair")
    return (paired[method] - paired[baseline]).to_numpy(dtype=float)


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    signed_rank_sum = float(np.sum(np.sign(nonzero) * ranks))
    return signed_rank_sum / float(np.sum(ranks))


def _wilcoxon_summary(differences: np.ndarray) -> tuple[float, float]:
    if np.all(differences == 0.0):
        return 0.0, 1.0
    outcome = wilcoxon(differences, alternative="two-sided")
    return float(outcome.statistic), float(outcome.pvalue)


def primary_paired_comparison(
    records: pd.DataFrame | Iterable[Mapping[str, object]],
    method: str = "tr_mla",
    baseline: str = "criticality_greedy",
    seed: int = 20260806,
) -> PrimaryComparison:
    """从原始 records 聚合后执行冻结的 60-spec 主终点统计。

    主比较不接受调用方构造的聚合帧，避免用可伪造的范围字符串绕过
    m50/m70 的原始缺失率证据和 spec 内平均。
    """
    frame = aggregate_primary_by_spec(records)
    differences = _paired_differences(
        frame, method, baseline, "crr_at_8", _PRIMARY_SPEC_COUNT
    )
    statistic, p_value = _wilcoxon_summary(differences)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, differences.size, size=(_BOOTSTRAP_REPLICATES, differences.size))
    bootstrap_means = differences[samples].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))
    return PrimaryComparison(
        method=method,
        baseline=baseline,
        metric="crr_at_8",
        n_specs=differences.size,
        mean_difference=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        wilcoxon_statistic=statistic,
        p_value=p_value,
        rank_biserial=_rank_biserial(differences),
        seed=seed,
    )


def exploratory_paired_comparison(
    frame: pd.DataFrame,
    method: str = "tr_mla",
    baseline: str = "criticality_greedy",
) -> ExploratoryComparison:
    """对已聚合的任意完整机制 spec pairs 进行探索性配对比较。"""
    _require_aggregated_m50_m70_frame(frame)
    differences = _paired_differences(frame, method, baseline, "crr_at_8", None)
    statistic, p_value = _wilcoxon_summary(differences)
    return ExploratoryComparison(
        method=method,
        baseline=baseline,
        metric="crr_at_8",
        n_specs=differences.size,
        mean_difference=float(differences.mean()),
        wilcoxon_statistic=statistic,
        p_value=p_value,
        rank_biserial=_rank_biserial(differences),
    )


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (count - rank) * p_values[int(index)]))
        adjusted[int(index)] = running_max
    return adjusted


def holm_adjust_p_values(p_values: list[float]) -> list[float]:
    """对一族探索性检验执行 Holm 校正，并保留输入顺序。"""
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("Holm p-values must be in [0, 1]")
    return _holm_adjust(p_values)


def secondary_paired_comparisons(
    records: pd.DataFrame | Iterable[Mapping[str, object]],
    method: str = "tr_mla",
    baseline: str = "criticality_greedy",
) -> pd.DataFrame:
    """输出探索性机制比较；所有 p 值均作 Holm 校正。"""
    source = _as_frame(records)
    if "mechanism" not in source:
        raise ValueError("secondary comparisons require a mechanism column")
    rows: list[dict[str, object]] = []
    for mechanism, subset in source.groupby("mechanism", sort=True):
        comparison = exploratory_paired_comparison(
            aggregate_primary_by_spec(subset), method, baseline
        )
        rows.append(
            {
                "mechanism": mechanism,
                "analysis_label": "exploratory",
                "n_specs": comparison.n_specs,
                "mean_difference": comparison.mean_difference,
                "p_value": comparison.p_value,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["mechanism", "analysis_label", "n_specs", "mean_difference", "p_value", "holm_p_value"])
    result["holm_p_value"] = holm_adjust_p_values(result["p_value"].tolist())
    return result

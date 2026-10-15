from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MANUSCRIPT = PROJECT_ROOT / "manuscript_v2" / "manuscript.tex"
SUPPLEMENTARY = PROJECT_ROOT / "manuscript_v2" / "supplementary.tex"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_primary_effect_uses_exactly_60_formal_spec_pairs() -> None:
    """主效应图必须忠实复现冻结的 60-spec 配对比较，而非重新抽样。"""
    from generate_v3_evidence_figures import build_primary_paired_effect_data

    table = build_primary_paired_effect_data(
        PROJECT_ROOT / "results" / "v3" / "formal" / "summary" / "primary_m50_m70.csv",
        PROJECT_ROOT / "results" / "v3" / "formal" / "summary" / "confirmatory_primary.json",
    )

    assert len(table) == 60
    assert table["spec_id"].is_unique
    assert table["delta_crr_at_8"].mean() == pytest.approx(0.06323247499532464)
    assert set(table) >= {"spec_id", "delta_crr_at_8"}


def test_operating_profile_uses_the_complete_ten_policy_roster() -> None:
    """性能—成本图不得只挑选对 TR-MLA 有利的策略。"""
    from generate_v3_evidence_figures import build_operating_profile_data

    table = build_operating_profile_data(
        PROJECT_ROOT / "results" / "v3" / "formal" / "summary" / "overall.csv"
    )

    assert len(table) == 10
    assert table["policy_name"].is_unique
    assert set(table["policy_name"]) >= {
        "criticality_greedy",
        "marginal_lookahead_no_relations",
        "tr_mla",
        "typed_relational_myopic",
    }
    assert set(table) == {"policy_name", "crr_at_8", "answer_cost"}


def test_heatmap_counts_first_two_pivot_selection_from_episode_steps() -> None:
    """行为热图必须从每条 episode 的前两步重建 pivot 选择率。"""
    from generate_v3_evidence_figures import build_policy_behaviour_table

    table = build_policy_behaviour_table(
        PROJECT_ROOT / "results" / "v3" / "formal" / "episodes.jsonl",
        PROJECT_ROOT / "data" / "generated" / "v3" / "test_tasks.jsonl",
    )

    assert set(table["policy_name"]) == {
        "criticality_greedy",
        "marginal_lookahead_no_relations",
        "tr_mla",
        "typed_relational_myopic",
    }
    assert len(table) == 12
    assert table["pivot_task_count"].eq(45).all()
    assert table["all_task_count"].eq(60).all()
    assert table["pivot_selected_first_two_rate"].between(0.0, 1.0).all()


def test_renderer_writes_three_pdf_png_pairs_and_provenance(tmp_path: Path) -> None:
    """删去任意渲染或来源记录，论文图件便不再可追溯。"""
    from generate_v3_evidence_figures import render_evidence_figures

    outputs = render_evidence_figures(
        formal_root=PROJECT_ROOT / "results" / "v3" / "formal",
        generated_root=PROJECT_ROOT / "data" / "generated" / "v3",
        manuscript_root=tmp_path / "manuscript_v2",
    )

    assert set(outputs) == {
        "fig_primary_paired_effect",
        "fig_operating_profile",
        "figS1_policy_behaviour_heatmap",
    }
    assert {path.suffix for path in outputs.values()} == {".pdf"}
    for pdf_path in outputs.values():
        assert pdf_path.is_file()
        assert pdf_path.with_suffix(".png").is_file()
        assert pdf_path.with_suffix(".provenance.json").is_file()


def test_manuscript_keeps_claim_boundary_and_adds_evidence_figures() -> None:
    """主文必须让新增图服务于已有结论，而不能扩大结论范围。"""
    source = MANUSCRIPT.read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "fig:primary-paired-effect" in source
    assert "fig:operating-profile" in source
    assert "pre-specified composite success gate is not met" in source
    assert "independent multi-seed robustness study was not completed" in normalized


def test_manuscript_labels_document_derived_case_without_claiming_external_validation() -> None:
    """案例只能作为下一轮研究的审计背景，不能被提升为外部性能证据。"""
    source = " ".join(MANUSCRIPT.read_text(encoding="utf-8").split())

    assert "document-derived rail-harness scenario" in source
    assert "not entered into the formal comparison" in source
    assert "not an external performance result" in source


def test_manuscript_positions_llms_as_unexecuted_future_baselines() -> None:
    """LLM 可作为未来接口或基线，但不能被误写成当前 formal 结果。"""
    source = " ".join(MANUSCRIPT.read_text(encoding="utf-8").split())

    assert "Language-model-assisted requirement elicitation" in source
    assert "no LLM is invoked in the formal execution" in source
    assert "model identifier, provider endpoint, prompt version, decoding parameters" in source
    assert "not an LLM comparison" in source


def test_related_work_covers_acquisition_planning_engineering_and_reproducibility() -> None:
    """相关工作应以可核验文献连接算法定位、工程约束与证据边界。"""
    source = MANUSCRIPT.read_text(encoding="utf-8")
    bibliography = (PROJECT_ROOT / "manuscript_v2" / "references.bib").read_text(
        encoding="utf-8"
    )

    for heading in (
        "Cost-aware and instance-wise information acquisition",
        "Generative, policy-learning, and non-myopic acquisition",
        "Mechanical requirement representation, configuration, and feasibility",
        "Traceability and reproducible empirical evidence",
    ):
        assert f"\\subsection{{{heading}}}" in source

    for key in (
        "DulacArnold2011",
        "Shim2018",
        "Pacheco2019",
        "Zheng2018",
        "McKay2001",
        "Kristjansdottir2018",
        "Bouthillier2019",
    ):
        assert key in source
        assert f"{{{key}," in bibliography

    related_work = source[
        source.index("\\section{Related Work}") : source.index("\\section{Method}")
    ]
    assert related_work.count("\\cite") >= 14


def test_method_layout_does_not_force_figures_and_tables_to_leave_body_text_stranded() -> None:
    """方法和实验正文应能绕开大图表续排，而不是被强制浮动体截断。"""
    source = MANUSCRIPT.read_text(encoding="utf-8")

    assert r"\begin{figure}[H]" not in source
    assert r"\begin{table}[H]" not in source

    method_figures_end = source.index(r"\label{fig:trmla-protocol}")
    method_barrier = source.index(r"\FloatBarrier", method_figures_end)
    experiment_start = source.index(r"\section{Experiments}")
    assert method_figures_end < method_barrier < experiment_start
    assert source.index(r"\label{fig:trmla-complete-architecture}") < source.index(
        "The expectations range over the normalized feasible branches"
    )
    assert source.index("Two controls isolate the main components.") < source.index(
        r"\label{fig:trmla-protocol}"
    )


def test_bibliography_uses_compact_local_typesetting_to_avoid_orphaned_reference_pages() -> None:
    """新增文献不应因默认参考文献行距单独形成大片留白页。"""
    source = MANUSCRIPT.read_text(encoding="utf-8")

    bibliography_start = source.index(r"\begingroup\small")
    bibliography_end = source.index(r"\endgroup", bibliography_start)
    bibliography_block = source[bibliography_start:bibliography_end]
    assert r"\setlength{\itemsep}{0.18em}" in bibliography_block
    assert r"\bibliography{references}" in bibliography_block


def test_manuscript_uses_plain_pagestyle_without_running_headers() -> None:
    """匿名稿保留页码，但不应出现运行页眉。"""
    source = MANUSCRIPT.read_text(encoding="utf-8")

    assert r"\pagestyle{plain}" in source
    assert r"\usepackage{fancyhdr}" not in source
    assert r"\fancyhead" not in source
    assert r"\thispagestyle{fancy}" not in source


def test_main_figure_order_preserves_method_architecture_and_evidence_sequence() -> None:
    """方法架构图与新增证据图必须保持既定的源文件顺序。"""
    source = MANUSCRIPT.read_text(encoding="utf-8")

    assert "fig:trmla-complete-architecture" in source
    assert "fig2_trmla_complete_architecture_v1.png" in source
    assert source.index("\\label{fig:architecture}") < source.index("\\label{fig:trmla-complete-architecture}")
    assert source.index("\\label{fig:trmla-complete-architecture}") < source.index("\\label{fig:budget}")
    assert "reports no empirical performance" in source
    assert source.index("\\label{fig:inference}") < source.index("\\label{fig:primary-paired-effect}")
    assert source.index("\\label{fig:primary-paired-effect}") < source.index("\\label{fig:operating-profile}")
    assert source.index("\\FloatBarrier") < source.index("\\subsection{Evidence boundary and reproducibility}")


def test_trmla_protocol_figure_is_embedded_before_experimental_figures() -> None:
    """图3应在方法架构图后解释决策协议，且不扩张实验结论。"""
    source = MANUSCRIPT.read_text(encoding="utf-8")

    asset = "fig3_trmla_protocol_v5_single_image.png"
    assert asset in source
    assert "fig:trmla-protocol" in source
    assert source.index(r"\label{fig:trmla-complete-architecture}") < source.index(
        r"\label{fig:trmla-protocol}"
    )
    assert source.index(r"\label{fig:trmla-protocol}") < source.index(r"\label{fig:budget}")

    figure_start = source.index(asset)
    figure_end = source.index(r"\end{figure}", figure_start)
    figure_block = source[figure_start:figure_end]
    assert "does not report experimental performance" in figure_block


def test_method_exposes_reproducible_state_posterior_and_search_contract() -> None:
    """方法必须公开状态、可行分支、确定性选择及两步枚举的计算边界。"""
    source = " ".join(MANUSCRIPT.read_text(encoding="utf-8").split())

    assert "\\label{eq:candidate-set}" in source
    assert "\\label{eq:feasible-posterior}" in source
    assert "confidence threshold is 0.95" in source
    assert "lower normalized answer cost and then the lexicographic field identifier" in source
    assert "$O(nq+n^2q^2)$" in source


def test_supplementary_declares_trace_level_exploratory_behaviour_rule() -> None:
    """补充图必须公开逐轨迹计数规则及其探索性范围。"""
    assert SUPPLEMENTARY.is_file()
    source = " ".join(SUPPLEMENTARY.read_text(encoding="utf-8").split())

    assert "fig:s1-behaviour" in source
    assert r"\renewcommand{\thefigure}{S\arabic{figure}}" in source
    assert "first two queried fields" in source
    assert "45 pivot tasks per mechanism" in source
    assert "exploratory descriptive analysis" in source


def test_supplementary_records_benchmark_and_decision_contract() -> None:
    """补充材料必须公开冻结边界与两步决策的可审计约定。"""
    source = " ".join(SUPPLEMENTARY.read_text(encoding="utf-8").split())

    assert "S2. Benchmark-generation and protection contract" in source
    assert "S3. Relation and decision contract" in source
    assert "20260806" in source
    assert "1,800 formal episode records" in source
    assert "validation only" in source
    assert "$O(nq+n^2q^2)$" in source

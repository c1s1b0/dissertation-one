from pathlib import Path

from paper1_active_req.external_cases.audit import audit_external_case


CASE_DIR = Path("data/external/EXT-RAIL-001")


def test_document_derived_case_is_not_eligible_for_formal_evaluation():
    report = audit_external_case(CASE_DIR)

    assert report.case_id == "EXT-RAIL-001"
    assert report.source_class == "document_derived_engineering_case"
    assert report.formal_evaluation_eligible is False
    assert "complete_ground_truth" in report.blocking_requirements
    assert "independent_engineer_review" in report.blocking_requirements


def test_audit_rejects_unsupported_evaluation_use(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case_metadata.yaml").write_text(
        """
case_id: TEST-001
source_class: document_derived_engineering_case
formal_evaluation_eligible: false
blocking_requirements: [complete_ground_truth]
""".strip(),
        encoding="utf-8",
    )
    (case_dir / "field_register.yaml").write_text(
        """
fields:
  - field_id: unsupported_target
    evidence_level: design_assumption
    evaluation_use: formal_evaluation_input
""".strip(),
        encoding="utf-8",
    )

    report = audit_external_case(case_dir)

    assert report.valid is False
    assert "unsupported_target" in report.violating_field_ids

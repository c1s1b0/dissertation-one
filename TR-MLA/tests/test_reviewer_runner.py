from pathlib import Path
import json

import pytest


def test_root_cannot_write_formal(tmp_path):
    from paper1_active_req.experiment.reviewer_runner import validate_root
    with pytest.raises(ValueError, match="isolated"):
        validate_root(tmp_path / "results" / "v3" / "formal")


def test_protocol_rejects_changed_input(tmp_path):
    from paper1_active_req.experiment.reviewer_runner import bind_protocol
    root = tmp_path / "research_artifacts" / "reviewer_supplement_20261009"
    source = tmp_path / "input.json"
    source.write_text("original", encoding="utf-8")
    first = bind_protocol(root, {"seeds": [1]}, {"input": source})
    assert bind_protocol(root, {"seeds": [1]}, {"input": source}) == first
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        bind_protocol(root, {"seeds": [1]}, {"input": source})


def test_checkpoint_rejects_duplicate_and_wrong_protocol(tmp_path):
    from paper1_active_req.experiment.reviewer_runner import read_checkpoint
    path = tmp_path / "records.jsonl"
    row = {"key": "one", "protocol_hash": "frozen", "status": "success"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_checkpoint(path, "frozen")
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        read_checkpoint(path, "different")


def test_checkpoint_keeps_failed_keys_and_does_not_call_them_success(tmp_path):
    from paper1_active_req.experiment.reviewer_runner import read_checkpoint
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"key": "failed", "protocol_hash": "frozen", "status": "failed"}) + "\n", encoding="utf-8")
    rows = read_checkpoint(path, "frozen")
    assert rows["failed"]["status"] == "failed"


def test_plan_has_exact_cost_keys_and_reuses_zero():
    from paper1_active_req.experiment.reviewer_runner import job_plan
    from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
    from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
    from paper1_active_req.generator.v3_relational_specs import V3AuditMetadata
    task = V3RequirementTask(task_id="one", spec_id="supp-test-interface_compatibility-001", split="test", missing_rate=.7,
        initial_state=RequirementState(values={"x": FieldValue(value=None, status=FieldStatus.HIDDEN)}),
        ground_truth={"x": FieldValue(value=1, status=FieldStatus.OBSERVED)},
        audit=V3AuditMetadata(module_id="test", pivot_field_id="x", mechanism="interface_compatibility"))
    jobs = job_plan([task], 20261011)
    assert len(jobs) == 10
    assert len({j["key"] for j in jobs}) == len(jobs)
    assert sum(j["lambda_cost"] == 0 for j in jobs) == 4
    assert {j["lambda_cost"] for j in jobs} == {0, .05, .1, .2}

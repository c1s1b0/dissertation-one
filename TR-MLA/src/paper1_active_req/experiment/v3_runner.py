from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import secrets
import msvcrt
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics
from paper1_active_req.evaluation.v3_statistics import (
    aggregate_primary_by_spec,
    exploratory_paired_comparison,
    holm_adjust_p_values,
    primary_paired_comparison,
    secondary_paired_comparisons,
)
from paper1_active_req.experiment.v3_episode import V3EpisodeResult, run_v3_episode
from paper1_active_req.experiment.v3_paths import (
    validate_v3_input_paths,
    validate_v3_output_path,
    validate_v3_paths,
)
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3DatasetManifest, V3RequirementTask
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.base import EmpiricalAnswerModel
from paper1_active_req.policies.baselines import (
    ConstraintRangeOnlyPolicy,
    CriticalityGreedyPolicy,
    FixedQuestionnairePolicy,
    RandomPolicy,
    UncertaintyAFAPolicy,
)
from paper1_active_req.policies.lookahead import (
    CCLAOneStepPolicy,
    CCLATwoStepPolicy,
    ConditionalAnswerModel,
)
from paper1_active_req.policies.marginal_lookahead import (
    MarginalLookaheadNoRelationsPolicy,
    TRMLAPolicy,
    TypedRelationalMyopicPolicy,
)
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.solver.engine import ConstraintEngine


V3_POLICY_NAMES: tuple[str, ...] = (
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
)
_LAMBDA_CANDIDATES = (0.0, 0.05, 0.1, 0.2)
_PRIMARY_RATES = (0.5, 0.7)
_PROTOCOL_MASTER_SEED = 20260806
_PROTOCOL_BUDGET = 8
_DEVELOPMENT_TRAIN_SPEC_COUNT = 180
_DEVELOPMENT_VALIDATION_SPEC_COUNT = 60
_DEVELOPMENT_SPEC_COUNT = (
    _DEVELOPMENT_TRAIN_SPEC_COUNT + _DEVELOPMENT_VALIDATION_SPEC_COUNT
)
_DEVELOPMENT_TASK_COUNT = _DEVELOPMENT_SPEC_COUNT * 3
_VALIDATION_TASK_COUNT = _DEVELOPMENT_VALIDATION_SPEC_COUNT * 3
_FORMAL_TEST_SPEC_COUNT = 60
_FORMAL_TASK_COUNT = _FORMAL_TEST_SPEC_COUNT * 3
_FORMAL_RECORD_COUNT = _FORMAL_TASK_COUNT * len(V3_POLICY_NAMES)
_FORMAL_SOURCE_RELATIVE_PATHS = (
    "src/paper1_active_req/experiment/v3_runner.py",
    "src/paper1_active_req/experiment/v3_episode.py",
    "src/paper1_active_req/experiment/v3_paths.py",
    "src/paper1_active_req/experiment/episode.py",
    "src/paper1_active_req/evaluation/v3_metrics.py",
    "src/paper1_active_req/evaluation/v3_protocol.py",
    "src/paper1_active_req/evaluation/v3_statistics.py",
    "src/paper1_active_req/policies/base.py",
    "src/paper1_active_req/policies/baselines.py",
    "src/paper1_active_req/policies/lookahead.py",
    "src/paper1_active_req/policies/marginal_lookahead.py",
    "src/paper1_active_req/relations/config.py",
    "src/paper1_active_req/relations/model.py",
    "src/paper1_active_req/relations/discrete.py",
    "src/paper1_active_req/relations/resources/v3_discrete_schema.yaml",
    "src/paper1_active_req/relations/continuous.py",
    "src/paper1_active_req/relations/posterior.py",
    "src/paper1_active_req/solver/engine.py",
    "src/paper1_active_req/solver/rules.py",
    "src/paper1_active_req/domain/models.py",
    "src/paper1_active_req/domain/schema.py",
    "src/paper1_active_req/generator/complete_specs.py",
    "src/paper1_active_req/generator/v3_relational_specs.py",
    "src/paper1_active_req/oracle/simulated.py",
)


class V3ExperimentRecord(BaseModel):
    task_id: str
    spec_id: str
    split: str
    missing_rate: float
    mechanism: str
    pivot: bool
    policy_name: str
    config_hash: str
    model_hash: str
    episode: V3EpisodeResult
    metrics: dict[str, float]


class V3FreezeManifest(BaseModel):
    status: str = "frozen"
    test_released: bool = False
    master_seed: int
    budget: int
    selected_lambda: float
    training_specs_hash: str
    validation_tasks_hash: str
    validation_candidate_records_hash: str
    development_manifest_hash: str
    schema_hash: str
    relation_config_hash: str
    experiment_config_hash: str
    source_hash: str
    model_summary_hash: str
    selection_file_hash: str
    artifact_paths: dict[str, str]
    source_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"required v3 artifact is unreadable: {path}") from error


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _as_absolute(path: Path) -> Path:
    return path.resolve(strict=False)


def formal_v3_source_paths() -> tuple[Path, ...]:
    """正式执行不可缩减的本项目源码清单。"""
    project_root = Path(__file__).resolve().parents[3]
    return tuple(project_root / relative for relative in _FORMAL_SOURCE_RELATIVE_PATHS)


def _deterministic_model_summary_hash(
    training_specs_path: Path,
    schema_hash: str,
    relation_config_hash: str,
) -> str:
    specs = _load_training_specs(training_specs_path)
    return _json_hash(
        {
            "training_specs": [spec.model_dump(mode="json") for spec in specs],
            "schema_hash": schema_hash,
            "relation_config_hash": relation_config_hash,
        }
    )


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterable[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"v3 output is locked: {lock_path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
            lock_file.write(str(os.getpid()))
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _exclusive_output(output_path: Path) -> Iterable[None]:
    """以 O_EXCL 保护一个输出；崩溃不会默默覆盖已有运行。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    with _exclusive_lock(lock_path):
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        yield


def _write_jsonl_once(path: Path, records: Iterable[BaseModel]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary v3 output already exists: {temporary}")
    try:
        temporary.write_text(
            "\n".join(record.model_dump_json() for record in records) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_once(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary v3 output already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class FormalAttempt:
    started_path: Path
    audit_dir: Path
    owner_token: str
    lease_path: Path
    lease_fd: int
    state_lock: Any = field(default_factory=threading.RLock, repr=False)


_FORMAL_ATTEMPT_OWNERS: dict[Path, FormalAttempt] = {}
_FORMAL_LEASE_OFFSETS = (0, 1)


def _lock_formal_lease_byte(lease_fd: int, offset: int) -> None:
    os.lseek(lease_fd, offset, os.SEEK_SET)
    msvcrt.locking(lease_fd, msvcrt.LK_NBLCK, 1)


def _unlock_formal_lease_byte(lease_fd: int, offset: int) -> None:
    os.lseek(lease_fd, offset, os.SEEK_SET)
    msvcrt.locking(lease_fd, msvcrt.LK_UNLCK, 1)


def _release_formal_lease(lease_fd: int) -> None:
    """逐字节释放 lease；任一 unlock 失败也不能跳过 close。"""
    try:
        for offset in _FORMAL_LEASE_OFFSETS:
            try:
                _unlock_formal_lease_byte(lease_fd, offset)
            except OSError:
                pass
    finally:
        try:
            os.close(lease_fd)
        except OSError:
            pass


def _assert_formal_attempt_owner(attempt: FormalAttempt, owner_token: str) -> None:
    if _FORMAL_ATTEMPT_OWNERS.get(attempt.started_path.resolve()) is not attempt:
        raise PermissionError("formal attempt handle is not the active owner")
    if not secrets.compare_digest(owner_token, attempt.owner_token):
        raise PermissionError("formal attempt owner does not match")
    try:
        os.fstat(attempt.lease_fd)
        # 两个字节分别持锁；验证一个字节时另一个始终作为跨进程护栏。
        # 只有真实持锁进程才能成功解锁并重新锁定同一字节。
        for offset in _FORMAL_LEASE_OFFSETS:
            _unlock_formal_lease_byte(attempt.lease_fd, offset)
            _lock_formal_lease_byte(attempt.lease_fd, offset)
    except OSError as error:
        raise PermissionError("formal attempt lease is not held") from error


def _begin_formal_attempt(
    *, output_path: Path, freeze_manifest_path: Path, release_manifest_path: Path
) -> FormalAttempt:
    """在加载 test task 前以 O_EXCL 留下不可复用的正式尝试审计起点。"""
    audit_path = output_path.with_name(f"{output_path.name}.formal-attempt.json")
    owner_token = secrets.token_urlsafe(32)
    audit_dir = audit_path.with_name(f"{audit_path.name}.audit")
    lease_path = audit_path.with_name(f"{audit_path.name}.lease")
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_fd = os.open(lease_path, os.O_CREAT | os.O_RDWR)
    try:
        os.write(lease_fd, b"00")
        for offset in _FORMAL_LEASE_OFFSETS:
            _lock_formal_lease_byte(lease_fd, offset)
    except OSError as error:
        _release_formal_lease(lease_fd)
        raise RuntimeError(f"formal attempt lease is held: {lease_path}") from error
    try:
        with _exclusive_output(audit_path):
            try:
                audit_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError as error:
                raise FileExistsError(f"refusing to overwrite {audit_dir}") from error
            _write_json_once(
                audit_path,
                {
                    "status": "started",
                    "freeze_manifest_hash": _sha256(freeze_manifest_path),
                    "release_manifest_hash": _sha256(release_manifest_path),
                    "output_path": str(_as_absolute(output_path)),
                    "attempt_id": hashlib.sha256(owner_token.encode("utf-8")).hexdigest(),
                },
            )
        attempt = FormalAttempt(audit_path, audit_dir, owner_token, lease_path, lease_fd)
        _FORMAL_ATTEMPT_OWNERS[audit_path.resolve()] = attempt
        return attempt
    except BaseException:
        _release_formal_lease(lease_fd)
        raise


def _load_formal_attempt(started_path: Path) -> FormalAttempt:
    attempt = _FORMAL_ATTEMPT_OWNERS.get(started_path.resolve())
    if attempt is None:
        raise PermissionError("formal attempt owner is not held by this process")
    return attempt


def _write_formal_attempt_outcome(
    attempt: FormalAttempt,
    *,
    owner_token: str,
    status: str,
    record_count: int | None = None,
    error: BaseException | None = None,
    output_path: Path | None = None,
    _locked: bool = False,
) -> None:
    if not _locked:
        with attempt.state_lock:
            return _write_formal_attempt_outcome(
                attempt, owner_token=owner_token, status=status, record_count=record_count,
                error=error, output_path=output_path, _locked=True,
            )
    _assert_formal_attempt_owner(attempt, owner_token)
    if status not in {"succeeded", "failed", "aborted"}:
        raise ValueError("invalid formal terminal status")
    checkpoints = sorted((attempt.audit_dir / "checkpoints").glob("*.json"))
    records = sorted((attempt.audit_dir / "records").glob("*.json"))
    if [path.stem for path in checkpoints] != [f"{index:05d}" for index in range(len(checkpoints))] or [path.stem for path in records] != [f"{index:05d}" for index in range(len(records))]:
        raise ValueError("formal audit records/checkpoints are not contiguous")
    if status == "succeeded" and len(checkpoints) != len(records):
        raise ValueError("formal success audit records/checkpoints differ")
    paired_records: list[V3ExperimentRecord] = []
    attempt_hash = _sha256(attempt.started_path)
    for index, (checkpoint, record_path) in enumerate(zip(checkpoints, records)):
        checkpoint_data = _read_json_object(checkpoint, label="formal checkpoint")
        audit_data = _read_json_object(record_path, label="formal audit record")
        if checkpoint_data.get("record_hash") != _sha256(record_path):
            raise ValueError("formal checkpoint record hash does not match")
        record = V3ExperimentRecord.model_validate(audit_data.get("record"))
        if (
            checkpoint_data.get("record_index") != index
            or checkpoint_data.get("task_id") != record.task_id
            or checkpoint_data.get("policy_name") != record.policy_name
            or checkpoint_data.get("attempt_hash") != attempt_hash
            or audit_data.get("attempt_hash") != attempt_hash
        ):
            raise ValueError("formal checkpoint metadata does not match audit record")
        paired_records.append(record)
    if len(checkpoints) > len(records):
        raise ValueError("formal audit has checkpoint without record")
    if status == "succeeded" and (len(records) != _FORMAL_RECORD_COUNT or record_count != _FORMAL_RECORD_COUNT):
        raise ValueError("formal success requires exactly 1800 audited records")
    if status == "succeeded":
        if output_path is None:
            raise ValueError("formal success requires final output path")
        final_records = _read_jsonl(output_path, V3ExperimentRecord)
        if len(final_records) != _FORMAL_RECORD_COUNT:
            raise ValueError("formal success output requires exactly 1800 records")
        for index, (audit_path, final_record) in enumerate(zip(records, final_records, strict=True)):
            audit = _read_json_object(audit_path, label="formal audit record")
            if audit_path.stem != f"{index:05d}" or audit.get("record") != final_record.model_dump(mode="json"):
                raise ValueError("formal success output does not match audited records")
        task_counts = Counter(record.task_id for record in paired_records)
        if (
            _FORMAL_RECORD_COUNT != _FORMAL_TASK_COUNT * len(V3_POLICY_NAMES)
            or len({(record.task_id, record.policy_name) for record in paired_records}) != _FORMAL_RECORD_COUNT
            or {record.policy_name for record in paired_records} != set(V3_POLICY_NAMES)
            or len(task_counts) != _FORMAL_TASK_COUNT
            or any(count != len(V3_POLICY_NAMES) for count in task_counts.values())
        ):
            raise ValueError("formal success audit does not cover ten policies per task")
    started = _read_json_object(attempt.started_path, label="formal attempt")
    payload: dict[str, object] = {
        "status": status,
        "freeze_manifest_hash": started["freeze_manifest_hash"],
        "release_manifest_hash": started["release_manifest_hash"],
        "attempt_hash": _sha256(attempt.started_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed_record_count": len(checkpoints),
        "audit_record_count": len(records),
        "pending_audit_record_count": len(records) - len(checkpoints),
    }
    if record_count is not None:
        payload["record_count"] = record_count
    if error is not None:
        summary = f"{type(error).__name__}: {error}"
        payload["error_summary"] = summary[:500]
    outcome_path = attempt.audit_dir / "outcome.json"
    with _exclusive_output(outcome_path):
        _write_json_once(outcome_path, payload)
    try:
        _release_formal_lease(attempt.lease_fd)
    finally:
        _FORMAL_ATTEMPT_OWNERS.pop(attempt.started_path.resolve(), None)


def _write_formal_checkpoint(
    attempt: FormalAttempt, *, owner_token: str, record_index: int, task_id: str, policy_name: str, record_hash: str, _locked: bool = False
) -> None:
    if not _locked:
        with attempt.state_lock:
            return _write_formal_checkpoint(attempt, owner_token=owner_token, record_index=record_index, task_id=task_id, policy_name=policy_name, record_hash=record_hash, _locked=True)
    _assert_formal_attempt_owner(attempt, owner_token)
    if (attempt.audit_dir / "outcome.json").exists():
        raise RuntimeError("formal attempt is already terminal")
    checkpoint = attempt.audit_dir / "checkpoints" / f"{record_index:05d}.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_output(checkpoint):
        _write_json_once(checkpoint, {"record_index": record_index, "task_id": task_id, "policy_name": policy_name, "record_hash": record_hash, "attempt_hash": _sha256(attempt.started_path), "timestamp": datetime.now(timezone.utc).isoformat()})


def _write_formal_audit_record(attempt: FormalAttempt, *, owner_token: str, record_index: int, record: V3ExperimentRecord, _locked: bool = False) -> str:
    if not _locked:
        with attempt.state_lock:
            return _write_formal_audit_record(attempt, owner_token=owner_token, record_index=record_index, record=record, _locked=True)
    _assert_formal_attempt_owner(attempt, owner_token)
    if (attempt.audit_dir / "outcome.json").exists():
        raise RuntimeError("formal attempt is already terminal")
    path = attempt.audit_dir / "records" / f"{record_index:05d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_output(path):
        _write_json_once(path, {"record": record.model_dump(mode="json"), "attempt_hash": _sha256(attempt.started_path)})
    return _sha256(path)


def _write_formal_record_checkpoint(attempt: FormalAttempt, *, record: V3ExperimentRecord, record_index: int) -> None:
    """在同一 attempt 状态锁中完成 record 落盘与其 checkpoint。"""
    with attempt.state_lock:
        record_hash = _write_formal_audit_record(
            attempt, owner_token=attempt.owner_token, record_index=record_index, record=record, _locked=True
        )
        _write_formal_checkpoint(
            attempt, owner_token=attempt.owner_token, record_index=record_index,
            task_id=record.task_id, policy_name=record.policy_name, record_hash=record_hash, _locked=True,
        )


def _read_jsonl(path: Path, model: type[BaseModel]) -> list[Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"required v3 input is unreadable: {path}") from error
    try:
        return [model.model_validate_json(line) for line in lines if line.strip()]
    except ValueError as error:
        raise ValueError(f"invalid v3 JSONL input: {path}") from error


def _load_training_specs(path: Path) -> list[CompleteSpec]:
    specs = _read_jsonl(path, CompleteSpec)
    if not specs:
        raise ValueError("v3 runner requires training specs")
    if any(spec.split != "train" for spec in specs):
        raise ValueError("v3 runner training specs must all have split=train")
    return specs


def _load_tasks(path: Path) -> list[V3RequirementTask]:
    tasks = _read_jsonl(path, V3RequirementTask)
    if not tasks:
        raise ValueError("v3 runner requires at least one task")
    return tasks


def _policy_factory(
    *,
    task: V3RequirementTask,
    schema: dict[str, Any],
    engine: ConstraintEngine,
    training_specs: list[CompleteSpec],
    relation_config: Any,
    lambda_cost: float,
    seed: int,
    selected_policy_names: set[str],
) -> dict[str, tuple[Any, Any]]:
    policies: dict[str, tuple[Any, Any]] = {}
    relational_policy_names = selected_policy_names - {
        "marginal_lookahead_no_relations",
    }
    relational_model = None
    if relational_policy_names:
        relational_model = TypedRelationalModel.fit(
            training_specs, schema, relation_config, use_relations=True
        )

    def required_relational_model() -> Any:
        if relational_model is None:
            raise RuntimeError("selected policy requires a relational model")
        return relational_model

    empirical = None
    if selected_policy_names.intersection({"uncertainty_afa", "constraint_range_only"}):
        empirical = EmpiricalAnswerModel.from_complete_specs(
            values=[spec.values for spec in training_specs]
        )

    if "random" in selected_policy_names:
        policies["random"] = (RandomPolicy(seed), required_relational_model())
    if "fixed_questionnaire" in selected_policy_names:
        policies["fixed_questionnaire"] = (
            FixedQuestionnairePolicy(),
            required_relational_model(),
        )
    if "criticality_greedy" in selected_policy_names:
        policies["criticality_greedy"] = (
            CriticalityGreedyPolicy(),
            required_relational_model(),
        )
    if "uncertainty_afa" in selected_policy_names:
        policies["uncertainty_afa"] = (
            UncertaintyAFAPolicy(empirical),
            required_relational_model(),
        )
    if "constraint_range_only" in selected_policy_names:
        policies["constraint_range_only"] = (
            ConstraintRangeOnlyPolicy(engine, empirical),
            required_relational_model(),
        )

    if selected_policy_names.intersection({"ccla_one_step", "v2_style_two_step_unconditioned"}):
        module_by_field = {field_id: "v3" for field_id in schema}
        if "ccla_one_step" in selected_policy_names:
            conditioned = ConditionalAnswerModel.fit(
                training_specs=[spec.values for spec in training_specs],
                module_by_field=module_by_field,
                use_module_conditioning=True,
            )
            policies["ccla_one_step"] = (
                CCLAOneStepPolicy(
                    engine=engine,
                    answer_model=conditioned,
                    lambda_cost=lambda_cost,
                    lambda_solver=0.0,
                ),
                required_relational_model(),
            )
        if "v2_style_two_step_unconditioned" in selected_policy_names:
            unconditioned = ConditionalAnswerModel.fit(
                training_specs=[spec.values for spec in training_specs],
                module_by_field=module_by_field,
                use_module_conditioning=False,
            )
            two_step = CCLATwoStepPolicy(
                engine=engine,
                answer_model=unconditioned,
                lambda_cost=lambda_cost,
                lambda_solver=0.0,
            )
            two_step.name = "v2_style_two_step_unconditioned"
            policies["v2_style_two_step_unconditioned"] = (
                two_step,
                required_relational_model(),
            )

    if "marginal_lookahead_no_relations" in selected_policy_names:
        no_relations = MarginalLookaheadNoRelationsPolicy.from_training_specs(
            initial_state=task.initial_state,
            training_specs=training_specs,
            schema=schema,
            relation_config=relation_config,
            engine=engine,
            lambda_cost=lambda_cost,
        )
        policies["marginal_lookahead_no_relations"] = (
            no_relations,
            no_relations.evaluator.model,
        )
    if "typed_relational_myopic" in selected_policy_names:
        policies["typed_relational_myopic"] = (
            TypedRelationalMyopicPolicy(
                initial_state=task.initial_state,
                model=required_relational_model(),
                engine=engine,
                lambda_cost=lambda_cost,
            ),
            required_relational_model(),
        )
    if "tr_mla" in selected_policy_names:
        policies["tr_mla"] = (
            TRMLAPolicy(
                initial_state=task.initial_state,
                model=required_relational_model(),
                engine=engine,
                lambda_cost=lambda_cost,
            ),
            required_relational_model(),
        )
    if set(policies) != selected_policy_names:
        raise RuntimeError("v3 policy factory does not match the frozen policy roster")
    return policies


def _run_v3_policies(
    *,
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    relation_config_path: Path,
    output_path: Path,
    seed: int,
    budget: int,
    lambda_cost: float,
    policy_names: set[str] | None,
    allow_formal: bool,
    checkpoint: Callable[[V3ExperimentRecord, int], None] | None = None,
) -> int:
    validate_v3_input_paths(tasks_path, training_specs_path, schema_path, relation_config_path)
    validate_v3_output_path(output_path, formal=allow_formal)
    if budget < 0 or lambda_cost < 0.0:
        raise ValueError("v3 budget and lambda_cost must be non-negative")
    selected = set(V3_POLICY_NAMES) if policy_names is None else set(policy_names)
    if not selected or not selected.issubset(V3_POLICY_NAMES):
        raise ValueError("unknown v3 policy name")
    schema = load_field_schema(schema_path)
    relation_config = load_v3_relation_config(relation_config_path)
    engine = ConstraintEngine(schema)
    tasks = _load_tasks(tasks_path)
    training_specs = _load_training_specs(training_specs_path)
    config_hash = _json_hash(
        {
            "schema_hash": _sha256(schema_path),
            "relation_config_hash": _sha256(relation_config_path),
            "seed": seed,
            "budget": budget,
            "lambda_cost": lambda_cost,
            "policy_names": sorted(selected),
        }
    )
    model_hash = _json_hash(
        [spec.model_dump(mode="json") for spec in training_specs]
    )
    records: list[V3ExperimentRecord] = []
    for task in tasks:
        policies = _policy_factory(
            task=task,
            schema=schema,
            engine=engine,
            training_specs=training_specs,
            relation_config=relation_config,
            lambda_cost=lambda_cost,
            seed=int(_json_hash({"seed": seed, "task": task.task_id})[:16], 16),
            selected_policy_names=selected,
        )
        for policy_name in V3_POLICY_NAMES:
            if policy_name not in selected:
                continue
            policy, inference_model = policies[policy_name]
            episode = run_v3_episode(
                task=task,
                policy=policy,
                oracle=SimulatedOracle(task.ground_truth),
                engine=engine,
                budget=budget,
            )
            record = V3ExperimentRecord(
                    task_id=task.task_id,
                    spec_id=task.spec_id,
                    split=task.split,
                    missing_rate=task.missing_rate,
                    mechanism=task.audit.mechanism,
                    pivot=task.audit.pivot_field_id is not None,
                    policy_name=policy_name,
                    config_hash=config_hash,
                    model_hash=model_hash,
                    episode=episode,
                    metrics=v3_episode_metrics(
                        episode, task, schema, inference_model, engine
                    ),
                )
            records.append(record)
            if checkpoint is not None:
                checkpoint(record, len(records) - 1)
    with _exclusive_output(output_path):
        _write_jsonl_once(output_path, records)
    return len(records)


def run_v3_policies(
    *,
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    relation_config_path: Path,
    output_path: Path,
    seed: int,
    budget: int,
    lambda_cost: float,
    policy_names: set[str] | None = None,
) -> int:
    return _run_v3_policies(
        tasks_path=tasks_path,
        training_specs_path=training_specs_path,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
        output_path=output_path,
        seed=seed,
        budget=budget,
        lambda_cost=lambda_cost,
        policy_names=policy_names,
        allow_formal=False,
    )


def _candidate_summary(candidate: Mapping[str, object]) -> dict[str, float]:
    records = candidate.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("lambda candidate requires records")
    per_spec: dict[str, dict[float, Mapping[str, object]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("lambda candidate record must be a mapping")
        metrics_record = record.get("metrics", record)
        if not isinstance(metrics_record, Mapping):
            raise ValueError("lambda candidate record metrics must be a mapping")
        try:
            spec_id = str(record["spec_id"])
            rate = float(record["missing_rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("lambda candidate record lacks spec/rate") from error
        if rate not in _PRIMARY_RATES:
            continue
        if rate in per_spec.setdefault(spec_id, {}):
            raise ValueError("lambda candidate has duplicate primary spec/rate")
        per_spec[spec_id][rate] = metrics_record
    if not per_spec or any(set(values) != set(_PRIMARY_RATES) for values in per_spec.values()):
        raise ValueError("lambda candidate requires one m50 and m70 record per spec")
    spec_means = []
    for values in per_spec.values():
        try:
            spec_means.append(
                {
                    metric: sum(float(values[rate][metric]) for rate in _PRIMARY_RATES) / 2.0
                    for metric in ("crr_at_8", "crr_auc", "answer_cost")
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("lambda candidate records lack selection metrics") from error
    return {
        metric: sum(row[metric] for row in spec_means) / len(spec_means)
        for metric in ("crr_at_8", "crr_auc", "answer_cost")
    }


def select_v3_lambda_candidate(candidates: list[Mapping[str, object]]) -> dict[str, object]:
    """按冻结顺序：m50/m70 的 spec 均值、AUC、成本、较小 lambda。"""
    evaluated: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            lambda_cost = float(candidate["lambda_cost"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("lambda candidate lacks lambda_cost") from error
        if lambda_cost not in _LAMBDA_CANDIDATES:
            raise ValueError("lambda candidate is outside the frozen grid")
        evaluated.append(
            {
                **dict(candidate),
                "lambda_cost": lambda_cost,
                "selection_metrics": _candidate_summary(candidate),
            }
        )
    if {item["lambda_cost"] for item in evaluated} != set(_LAMBDA_CANDIDATES):
        raise ValueError("v3 tuning requires exactly the four frozen lambda candidates")
    if len(evaluated) != len(_LAMBDA_CANDIDATES):
        raise ValueError("v3 tuning candidates must be unique")
    return min(
        evaluated,
        key=lambda item: (
            -float(item["selection_metrics"]["crr_at_8"]),
            -float(item["selection_metrics"]["crr_auc"]),
            float(item["selection_metrics"]["answer_cost"]),
            float(item["lambda_cost"]),
        ),
    )


def run_v3_validation_tuning(
    *,
    validation_tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    relation_config_path: Path,
    output_dir: Path,
    seed: int,
    budget: int,
    experiment_config_path: Path | None = None,
) -> dict[str, object]:
    if seed != _PROTOCOL_MASTER_SEED or budget != _PROTOCOL_BUDGET:
        raise ValueError("v3 validation tuning requires frozen seed 20260806 and budget 8")
    experiment_config_path = experiment_config_path or schema_path.parent / "v3_experiment.yaml"
    validate_v3_input_paths(
        validation_tasks_path, training_specs_path, schema_path, relation_config_path, experiment_config_path
    )
    validate_v3_output_path(output_dir)
    tasks = _load_tasks(validation_tasks_path)
    if any(task.split != "validation" for task in tasks):
        raise ValueError("v3 tuning accepts validation tasks only")
    if (
        len(tasks) != _VALIDATION_TASK_COUNT
        or len({task.spec_id for task in tasks})
        != _DEVELOPMENT_VALIDATION_SPEC_COUNT
    ):
        raise ValueError("v3 tuning requires the frozen 60-spec/180-task validation roster")
    if any(
        {item.missing_rate for item in tasks if item.spec_id == spec_id} != {0.3, 0.5, 0.7}
        for spec_id in {task.spec_id for task in tasks}
    ):
        raise ValueError("v3 tuning validation roster requires m30/m50/m70 per spec")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse v3 tuning output {output_dir}")
    lock_path = output_dir.with_name(f"{output_dir.name}.lock")
    with _exclusive_lock(lock_path):
        output_dir.mkdir(parents=True, exist_ok=False)
        candidates: list[dict[str, object]] = []
        for lambda_cost in _LAMBDA_CANDIDATES:
            record_path = output_dir / f"candidate-lambda-{lambda_cost:.2f}.jsonl"
            run_v3_policies(
                tasks_path=validation_tasks_path,
                training_specs_path=training_specs_path,
                schema_path=schema_path,
                relation_config_path=relation_config_path,
                output_path=record_path,
                seed=seed,
                budget=budget,
                lambda_cost=lambda_cost,
                policy_names={"tr_mla"},
            )
            records = [
                V3ExperimentRecord.model_validate_json(line).model_dump(mode="json")
                for line in record_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            candidates.append(
                {
                    "lambda_cost": lambda_cost,
                    "policy_names": ["tr_mla"],
                    "records_path": record_path.name,
                    "records": records,
                }
            )
        selected = select_v3_lambda_candidate(candidates)
        model_summary_hash = _json_hash(
            sorted(
                record["model_hash"]
                for candidate in candidates
                for record in candidate["records"]
            )
        )
        selection = {
            "selection_split": "validation",
            "seed": seed,
            "budget": budget,
            "selected_lambda": {"lambda_cost": selected["lambda_cost"]},
            "selection_metrics": selected["selection_metrics"],
            "model_summary_hash": model_summary_hash,
            "validation_tasks_hash": _sha256(validation_tasks_path),
            "training_specs_hash": _sha256(training_specs_path),
            "schema_hash": _sha256(schema_path),
            "relation_config_hash": _sha256(relation_config_path),
            "experiment_config_hash": _sha256(experiment_config_path),
            "candidate_records_hash": _json_hash(candidates),
            "candidates": candidates,
        }
        selection_path = output_dir / "selected_lambda.json"
        _write_json_once(selection_path, selection)
        return selection


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}")
    return value


def _validate_frozen_validation_selection(
    *,
    selection: Mapping[str, object],
    validation_tasks_path: Path,
    training_specs_path: Path,
    schema_hash: str,
    relation_config_hash: str,
    experiment_config_hash: str,
    master_seed: int,
    budget: int,
) -> None:
    """冻结前验证 selection 的完整 validation 来源、四个候选和逐条输出。"""
    if selection.get("selection_split") != "validation":
        raise ValueError("freeze requires a validation selection")
    if selection.get("seed") != master_seed or selection.get("budget") != budget:
        raise ValueError("freeze selection seed/budget does not match experiment config")
    if master_seed != _PROTOCOL_MASTER_SEED or budget != _PROTOCOL_BUDGET:
        raise ValueError("freeze requires frozen seed 20260806 and budget 8")
    expected_hashes = {
        "validation_tasks_hash": _sha256(validation_tasks_path),
        "training_specs_hash": _sha256(training_specs_path),
        "schema_hash": schema_hash,
        "relation_config_hash": relation_config_hash,
        "experiment_config_hash": experiment_config_hash,
    }
    if any(selection.get(name) != value for name, value in expected_hashes.items()):
        raise ValueError("freeze selection provenance hash does not match")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(_LAMBDA_CANDIDATES):
        raise ValueError("freeze selection requires all four non-empty lambda candidates")
    if selection.get("candidate_records_hash") != _json_hash(candidates):
        raise ValueError("freeze selection candidate records hash does not match")
    tasks = _load_tasks(validation_tasks_path)
    if (
        len(tasks) != _VALIDATION_TASK_COUNT
        or len({task.spec_id for task in tasks})
        != _DEVELOPMENT_VALIDATION_SPEC_COUNT
    ):
        raise ValueError("freeze selection validation roster does not match")
    task_by_id = {task.task_id: task for task in tasks}
    expected_model_hash = _json_hash(
        [spec.model_dump(mode="json") for spec in _load_training_specs(training_specs_path)]
    )
    observed_model_hashes: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("freeze selection candidate is invalid")
        try:
            lambda_cost = float(candidate["lambda_cost"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("freeze selection candidate lacks lambda_cost") from error
        records = candidate.get("records")
        if candidate.get("policy_names") != ["tr_mla"] or not isinstance(records, list):
            raise ValueError("freeze selection candidate roster is invalid")
        expected_config_hash = _json_hash({
            "schema_hash": schema_hash, "relation_config_hash": relation_config_hash,
            "seed": master_seed, "budget": budget, "lambda_cost": lambda_cost,
            "policy_names": ["tr_mla"],
        })
        if len(records) != _VALIDATION_TASK_COUNT:
            raise ValueError("freeze selection candidate record roster is incomplete")
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("freeze selection record is invalid")
            task = task_by_id.get(str(record.get("task_id")))
            if task is None or task.task_id in seen:
                raise ValueError("freeze selection record does not bind validation task")
            if (
                record.get("split") != "validation"
                or record.get("policy_name") != "tr_mla"
                or record.get("spec_id") != task.spec_id
                or float(record.get("missing_rate", -1.0)) != task.missing_rate
                or record.get("config_hash") != expected_config_hash
                or record.get("model_hash") != expected_model_hash
            ):
                raise ValueError("freeze selection record provenance does not match")
            seen.add(task.task_id)
            observed_model_hashes.append(expected_model_hash)
        if seen != set(task_by_id):
            raise ValueError("freeze selection candidate record roster is incomplete")
    if {float(candidate["lambda_cost"]) for candidate in candidates} != set(_LAMBDA_CANDIDATES):
        raise ValueError("freeze selection candidates do not match frozen grid")
    if selection.get("model_summary_hash") != _json_hash(sorted(observed_model_hashes)):
        raise ValueError("freeze selection model summary hash does not match")
    recomputed = select_v3_lambda_candidate(candidates)
    selected = selection.get("selected_lambda")
    selected_lambda = selected.get("lambda_cost") if isinstance(selected, Mapping) else selected
    if selected_lambda != recomputed["lambda_cost"] or selection.get("selection_metrics") != recomputed["selection_metrics"]:
        raise ValueError("freeze selection lambda/summary is not the frozen candidate result")


def _validate_development_manifest(
    development: Mapping[str, object], *, training_specs_path: Path, validation_tasks_path: Path,
    schema_hash: str, relation_hash: str, master_seed: int,
) -> None:
    try:
        manifest = V3DatasetManifest.model_validate(development)
    except ValueError as error:
        raise ValueError("development manifest is incomplete") from error
    training = _load_training_specs(training_specs_path)
    tasks = _load_tasks(validation_tasks_path)
    if (
        manifest.seed != master_seed or manifest.schema_hash != schema_hash
        or manifest.relation_config_hash != relation_hash
        or manifest.complete_spec_count != _DEVELOPMENT_SPEC_COUNT
        or manifest.task_count != _DEVELOPMENT_TASK_COUNT
        or manifest.split_spec_counts != {
            "train": _DEVELOPMENT_TRAIN_SPEC_COUNT,
            "validation": _DEVELOPMENT_VALIDATION_SPEC_COUNT,
        }
        or len(training) != _DEVELOPMENT_TRAIN_SPEC_COUNT
        or len({spec.spec_id for spec in training}) != _DEVELOPMENT_TRAIN_SPEC_COUNT
        or manifest.train_spec_ids != [spec.spec_id for spec in training]
        or len(manifest.validation_spec_ids) != _DEVELOPMENT_VALIDATION_SPEC_COUNT
        or set(manifest.validation_spec_ids) != {task.spec_id for task in tasks}
        or len(tasks) != _VALIDATION_TASK_COUNT
    ):
        raise ValueError("development manifest/train/validation roster does not match frozen protocol")


def freeze_v3_artifacts(
    *,
    development_manifest_path: Path,
    selection_file: Path,
    schema_path: Path,
    relation_config_path: Path,
    experiment_config_path: Path,
    source_paths: tuple[Path, ...],
    output_path: Path,
) -> V3FreezeManifest:
    validate_v3_paths(
        development_manifest_path, selection_file, schema_path, relation_config_path,
        experiment_config_path, *source_paths, output_path
    )
    expected_sources = tuple(sorted(_as_absolute(path) for path in formal_v3_source_paths()))
    supplied_sources = tuple(sorted(_as_absolute(path) for path in source_paths))
    if supplied_sources != expected_sources:
        raise ValueError("freeze requires the complete fixed formal source roster")
    development = _read_json_object(development_manifest_path, label="development manifest")
    selection = _read_json_object(selection_file, label="validation selection")
    if selection.get("selection_split") != "validation":
        raise ValueError("freeze requires a validation selection")
    selected = selection.get("selected_lambda")
    try:
        selected_lambda = float(selected["lambda_cost"] if isinstance(selected, Mapping) else selected)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("freeze selection lacks lambda_cost") from error
    if selected_lambda not in _LAMBDA_CANDIDATES:
        raise ValueError("freeze selection lambda is outside the frozen grid")
    try:
        experiment = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("invalid experiment config") from error
    if not isinstance(experiment, dict):
        raise ValueError("invalid experiment config")
    try:
        master_seed = int(experiment["master_seed"])
        budget = int(experiment["budget"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("experiment config requires master_seed and budget") from error
    schema_hash = _sha256(schema_path)
    relation_hash = _sha256(relation_config_path)
    training_specs_path = development_manifest_path.parent / "train_complete_specs.jsonl"
    training_specs_hash = _sha256(training_specs_path)
    validation_tasks_path = development_manifest_path.parent / "validation_tasks.jsonl"
    validation_tasks_hash = _sha256(validation_tasks_path)
    model_summary_hash = _deterministic_model_summary_hash(
        training_specs_path, schema_hash, relation_hash
    )
    _validate_development_manifest(
        development, training_specs_path=training_specs_path, validation_tasks_path=validation_tasks_path,
        schema_hash=schema_hash, relation_hash=relation_hash, master_seed=master_seed,
    )
    _validate_frozen_validation_selection(
        selection=selection,
        validation_tasks_path=validation_tasks_path,
        training_specs_path=training_specs_path,
        schema_hash=schema_hash,
        relation_config_hash=relation_hash,
        experiment_config_hash=_sha256(experiment_config_path),
        master_seed=master_seed,
        budget=budget,
    )
    source_hashes = {
        str(_as_absolute(path)): _sha256(path)
        for path in sorted(source_paths, key=lambda item: str(_as_absolute(item)))
    }
    artifact_paths = {
        "development_manifest": str(_as_absolute(development_manifest_path)),
        "selection_file": str(_as_absolute(selection_file)),
        "schema": str(_as_absolute(schema_path)),
        "relation_config": str(_as_absolute(relation_config_path)),
        "experiment_config": str(_as_absolute(experiment_config_path)),
        "training_specs": str(_as_absolute(training_specs_path)),
        "validation_tasks": str(_as_absolute(validation_tasks_path)),
    }
    manifest = V3FreezeManifest(
        master_seed=master_seed,
        budget=budget,
        selected_lambda=selected_lambda,
        training_specs_hash=training_specs_hash,
        validation_tasks_hash=validation_tasks_hash,
        validation_candidate_records_hash=str(selection["candidate_records_hash"]),
        development_manifest_hash=_sha256(development_manifest_path),
        schema_hash=schema_hash,
        relation_config_hash=relation_hash,
        experiment_config_hash=_sha256(experiment_config_path),
        source_hash=_json_hash(source_hashes),
        model_summary_hash=model_summary_hash,
        selection_file_hash=_sha256(selection_file),
        artifact_paths=artifact_paths,
        source_hashes=source_hashes,
    )
    with _exclusive_output(output_path):
        temporary = output_path.with_name(f"{output_path.name}.tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(output_path)
    return manifest


def _verify_freeze_manifest(path: Path) -> V3FreezeManifest:
    try:
        manifest = V3FreezeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("invalid freeze manifest") from error
    if manifest.status != "frozen" or manifest.test_released is not False:
        raise ValueError("freeze manifest is not a pristine frozen manifest")
    required_artifact_keys = {
        "development_manifest", "selection_file", "schema", "relation_config",
        "experiment_config", "training_specs", "validation_tasks",
    }
    if set(manifest.artifact_paths) != required_artifact_keys:
        raise ValueError("freeze artifact path roster is incomplete")
    validate_v3_input_paths(*(Path(manifest.artifact_paths[name]) for name in required_artifact_keys))
    required_paths = {
        "development_manifest": manifest.development_manifest_hash,
        "selection_file": manifest.selection_file_hash,
        "schema": manifest.schema_hash,
        "relation_config": manifest.relation_config_hash,
        "experiment_config": manifest.experiment_config_hash,
        "training_specs": manifest.training_specs_hash,
        "validation_tasks": manifest.validation_tasks_hash,
    }
    for label, expected_hash in required_paths.items():
        raw_path = manifest.artifact_paths.get(label)
        if raw_path is None or _sha256(Path(raw_path)) != expected_hash:
            raise ValueError(f"freeze {label} hash does not match")
    current_model_summary = _deterministic_model_summary_hash(
        Path(manifest.artifact_paths["training_specs"]),
        manifest.schema_hash,
        manifest.relation_config_hash,
    )
    if current_model_summary != manifest.model_summary_hash:
        raise ValueError("freeze training model summary does not match")
    expected_source_keys = {
        str(_as_absolute(source_path)) for source_path in formal_v3_source_paths()
    }
    if set(manifest.source_hashes) != expected_source_keys:
        raise ValueError("freeze source roster does not match fixed formal roster")
    actual_sources = {source_path: _sha256(Path(source_path)) for source_path in manifest.source_hashes}
    if actual_sources != manifest.source_hashes or _json_hash(actual_sources) != manifest.source_hash:
        raise ValueError("freeze source hash does not match")
    selection = _read_json_object(Path(manifest.artifact_paths["selection_file"]), label="validation selection")
    if selection.get("selection_split") != "validation":
        raise ValueError("freeze selection is no longer validation-only")
    selected = selection.get("selected_lambda")
    current_lambda = selected.get("lambda_cost") if isinstance(selected, Mapping) else selected
    if current_lambda != manifest.selected_lambda:
        raise ValueError("freeze selected lambda does not match")
    experiment = yaml.safe_load(
        Path(manifest.artifact_paths["experiment_config"]).read_text(encoding="utf-8")
    )
    if not isinstance(experiment, Mapping) or (
        experiment.get("master_seed") != manifest.master_seed
        or experiment.get("budget") != manifest.budget
        or manifest.selected_lambda not in experiment.get("lambda_candidates", [])
    ):
        raise ValueError("freeze experiment configuration no longer matches seed/budget/lambda")
    _validate_frozen_validation_selection(
        selection=selection,
        validation_tasks_path=Path(manifest.artifact_paths["validation_tasks"]),
        training_specs_path=Path(manifest.artifact_paths["training_specs"]),
        schema_hash=manifest.schema_hash,
        relation_config_hash=manifest.relation_config_hash,
        experiment_config_hash=manifest.experiment_config_hash,
        master_seed=manifest.master_seed,
        budget=manifest.budget,
    )
    if selection.get("candidate_records_hash") != manifest.validation_candidate_records_hash:
        raise ValueError("freeze validation candidate records do not match")
    return manifest


def _run_v3_selected_experiment_impl(
    *,
    test_tasks_path: Path,
    training_specs_path: Path,
    freeze_manifest_path: Path,
    output_path: Path,
    attempt_path: FormalAttempt,
) -> int:
    manifest = _verify_freeze_manifest(freeze_manifest_path)
    if _sha256(training_specs_path) != manifest.training_specs_hash:
        raise ValueError("freeze training hash does not match")
    if _as_absolute(training_specs_path) != Path(manifest.artifact_paths["training_specs"]):
        raise ValueError("freeze training path does not match")
    release_manifest_path = test_tasks_path.parent / "test_release_manifest.json"
    tasks = _load_tasks(test_tasks_path)
    if any(task.split != "test" for task in tasks):
        raise ValueError("formal runner accepts released test tasks only")
    release = _read_json_object(release_manifest_path, label="test release manifest")
    if release.get("freeze_manifest_hash") != _sha256(freeze_manifest_path):
        raise ValueError("formal runner test release does not match freeze manifest")
    test_specs_path = test_tasks_path.parent / "test_complete_specs.jsonl"
    required_release_hashes = {
        "test_tasks_hash": _sha256(test_tasks_path),
        "test_complete_specs_hash": _sha256(test_specs_path),
        "schema_hash": manifest.schema_hash,
        "relation_config_hash": manifest.relation_config_hash,
    }
    if any(release.get(field) != value for field, value in required_release_hashes.items()):
        raise ValueError("formal runner test release hash does not match")
    released_specs = _read_jsonl(test_specs_path, CompleteSpec)
    if release.get("released_task_ids") != [task.task_id for task in tasks]:
        raise ValueError("formal runner test release task IDs do not match")
    if release.get("released_spec_ids") != [spec.spec_id for spec in released_specs]:
        raise ValueError("formal runner test release spec IDs do not match")
    if {task.spec_id for task in tasks} != {spec.spec_id for spec in released_specs}:
        raise ValueError("formal runner test release task/spec IDs do not match")
    if len(released_specs) != _FORMAL_TEST_SPEC_COUNT or len(tasks) != _FORMAL_TASK_COUNT:
        raise ValueError("formal runner test release roster does not match the frozen protocol")
    if len({spec.spec_id for spec in released_specs}) != _FORMAL_TEST_SPEC_COUNT or any(
        spec.split != "test" for spec in released_specs
    ):
        raise ValueError("formal runner test release roster has invalid specs")
    rates_by_spec: dict[str, set[float]] = {}
    tasks_by_spec: Counter[str] = Counter()
    for task in tasks:
        rates_by_spec.setdefault(task.spec_id, set()).add(task.missing_rate)
        tasks_by_spec[task.spec_id] += 1
    if any(
        rates_by_spec.get(spec.spec_id) != {0.3, 0.5, 0.7}
        or tasks_by_spec[spec.spec_id] != 3
        for spec in released_specs
    ):
        raise ValueError("formal runner test release roster has invalid task rates")
    schema_path = Path(manifest.artifact_paths["schema"])
    relation_config_path = Path(manifest.artifact_paths["relation_config"])
    written = _run_v3_policies(
        tasks_path=test_tasks_path,
        training_specs_path=training_specs_path,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
        output_path=output_path,
        seed=manifest.master_seed,
        budget=manifest.budget,
        lambda_cost=manifest.selected_lambda,
        policy_names=None,
        allow_formal=True,
        checkpoint=lambda record, index: _write_formal_record_checkpoint(
            attempt_path, record=record, record_index=index
        ),
    )
    _write_formal_attempt_outcome(attempt_path, owner_token=attempt_path.owner_token, status="succeeded", record_count=written, output_path=output_path)
    return written


def run_v3_selected_experiment(
    *,
    test_tasks_path: Path,
    training_specs_path: Path,
    freeze_manifest_path: Path,
    output_path: Path,
) -> int:
    """执行唯一 formal 尝试；起点后的失败或中止均写入不可覆盖 outcome。"""
    release_manifest_path = test_tasks_path.parent / "test_release_manifest.json"
    test_specs_path = test_tasks_path.parent / "test_complete_specs.jsonl"
    validate_v3_input_paths(test_tasks_path, training_specs_path, freeze_manifest_path, release_manifest_path, test_specs_path)
    validate_v3_output_path(output_path, formal=True)
    attempt: FormalAttempt | None = None
    try:
        _verify_freeze_manifest(freeze_manifest_path)
        attempt = _begin_formal_attempt(
            output_path=output_path,
            freeze_manifest_path=freeze_manifest_path,
            release_manifest_path=release_manifest_path,
        )
        return _run_v3_selected_experiment_impl(
            test_tasks_path=test_tasks_path,
            training_specs_path=training_specs_path,
            freeze_manifest_path=freeze_manifest_path,
            output_path=output_path,
            attempt_path=attempt,
        )
    except BaseException as error:
        if attempt is not None:
            status = "aborted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            try:
                _write_formal_attempt_outcome(attempt, owner_token=attempt.owner_token, status=status, error=error)
            except BaseException as audit_error:
                error.add_note(f"formal audit outcome could not be written: {audit_error}")
        raise


def summarize_v3_results(
    *,
    episodes_path: Path,
    output_dir: Path,
    seed: int = 20260806,
) -> None:
    if seed != 20260806:
        raise ValueError("formal summary requires frozen bootstrap seed 20260806")
    records = _read_jsonl(episodes_path, V3ExperimentRecord)
    if len(records) != _FORMAL_RECORD_COUNT:
        raise ValueError(f"formal summary requires exactly {_FORMAL_RECORD_COUNT} records")
    if {record.policy_name for record in records} != set(V3_POLICY_NAMES):
        raise ValueError("formal summary requires all ten v3 policies")
    if any(record.split != "test" for record in records):
        raise ValueError("formal summary requires released test records only")
    keys = [
        (record.spec_id, record.missing_rate, record.policy_name)
        for record in records
    ]
    if len(set(keys)) != _FORMAL_RECORD_COUNT:
        raise ValueError("formal summary has duplicate task/policy records")
    grouped = Counter(keys)
    if any(count != 1 for count in grouped.values()):
        raise ValueError("formal summary has duplicate task/policy records")
    if len({record.spec_id for record in records}) != _FORMAL_TEST_SPEC_COUNT:
        raise ValueError(
            f"formal summary requires {_FORMAL_TEST_SPEC_COUNT} test specs"
        )
    rows = [
        {
            "spec_id": record.spec_id,
            "policy_name": record.policy_name,
            "missing_rate": record.missing_rate,
            "mechanism": record.mechanism,
            "pivot": record.pivot,
            **record.metrics,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    rate_sets = frame.groupby(["spec_id", "policy_name"])["missing_rate"].agg(set)
    if not rate_sets.map(lambda rates: rates == {0.3, 0.5, 0.7}).all():
        raise ValueError("formal summary requires m30/m50/m70 for every spec and policy")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v3 summary {output_dir}")
    with _exclusive_lock(output_dir.with_name(f"{output_dir.name}.lock")):
        output_dir.mkdir(parents=True, exist_ok=False)
        metric_columns = sorted(
            column
            for column in frame
            if column.startswith("crr_")
            or column
            in {
                "answer_cost",
                "noncritical_query_count",
                "inference_precision",
                "inference_coverage",
                "pivot_selected",
                "runtime_seconds",
                "solver_calls",
                "hard_constraint_satisfied",
            }
        )
        frame.groupby("policy_name", as_index=False)[metric_columns].mean().to_csv(output_dir / "overall.csv", index=False)
        primary_by_spec = aggregate_primary_by_spec(frame)
        primary_by_spec.to_csv(output_dir / "primary_m50_m70.csv", index=False)
        frame.groupby(["policy_name", "missing_rate"], as_index=False)[metric_columns].mean().to_csv(output_dir / "missing_rate.csv", index=False)
        frame.groupby(["policy_name", "mechanism"], as_index=False)[metric_columns].mean().to_csv(output_dir / "mechanism.csv", index=False)
        frame.groupby(["policy_name", "pivot"], as_index=False)[metric_columns].mean().to_csv(output_dir / "pivot.csv", index=False)
        frame.loc[frame["policy_name"].isin(("tr_mla", "typed_relational_myopic", "marginal_lookahead_no_relations"))].groupby("policy_name", as_index=False)[metric_columns].mean().to_csv(output_dir / "ablation.csv", index=False)
        primary_comparison = primary_paired_comparison(
            frame, method="tr_mla", baseline="criticality_greedy", seed=seed
        )
        _write_json_once(
            output_dir / "confirmatory_primary.json",
            {
                "analysis_label": "confirmatory",
                "sample_unit": "spec (m50/m70 within-spec mean)",
                "comparison": asdict(primary_comparison),
            },
        )
        secondary_paired_comparisons(frame).to_csv(
            output_dir / "exploratory_mechanism.csv", index=False
        )
        ablation_rows: list[dict[str, object]] = []
        for baseline in ("typed_relational_myopic", "marginal_lookahead_no_relations"):
            comparison = exploratory_paired_comparison(
                primary_by_spec, method="tr_mla", baseline=baseline
            )
            ablation_rows.append(
                {"analysis_label": "exploratory", **asdict(comparison)}
            )
        ablations = pd.DataFrame(ablation_rows)
        ablations["holm_p_value"] = holm_adjust_p_values(
            ablations["p_value"].tolist()
        )
        ablations.to_csv(output_dir / "exploratory_ablation.csv", index=False)
        tr_mla = frame.loc[frame["policy_name"] == "tr_mla"]
        gate = {
            "ci_lower_positive": primary_comparison.ci_low > 0.0,
            "p_value_below_0_05": primary_comparison.p_value < 0.05,
            "hard_constraints_all_satisfied": bool(
                tr_mla["hard_constraint_satisfied"].eq(1.0).all()
            ),
            "inference_precision_at_least_0_95": bool(
                tr_mla["inference_precision"].mean() >= 0.95
            ),
        }
        _write_json_once(
            output_dir / "success_gate.json", {**gate, "success": all(gate.values())}
        )

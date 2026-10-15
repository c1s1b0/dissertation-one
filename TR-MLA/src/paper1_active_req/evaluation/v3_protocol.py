from __future__ import annotations

from collections.abc import Mapping

from paper1_active_req.domain.models import FieldDefinition, FieldStatus, RequirementState


FROZEN_CRITICALITY_THRESHOLD = 0.8
_UNRESOLVED_INITIAL_STATUSES = frozenset(
    {FieldStatus.HIDDEN, FieldStatus.CONFLICTING, FieldStatus.INFERRED}
)


def frozen_critical_field_ids(
    initial_state: RequirementState,
    schema: Mapping[str, FieldDefinition],
) -> tuple[str, ...]:
    """返回预注册 CRR 的冻结关键字段集 K。

    K 只由任务初始公开状态和 schema 决定：hidden/conflicting/inferred、
    可询问且关键性不低于 0.8。字段排序固定，避免策略和指标出现不同分母。
    """
    return tuple(
        field_id
        for field_id in initial_state.values
        if (
            (definition := schema.get(field_id)) is not None
            and initial_state.values[field_id].status in _UNRESOLVED_INITIAL_STATUSES
            and definition.queryable
            and definition.criticality >= FROZEN_CRITICALITY_THRESHOLD
        )
    )

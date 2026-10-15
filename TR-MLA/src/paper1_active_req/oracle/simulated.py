from __future__ import annotations

from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
)


class SimulatedOracle:
    def __init__(
        self,
        ground_truth: dict[str, FieldValue],
        unknown_fields: set[str] | None = None,
    ) -> None:
        self._ground_truth = {
            field_id: value.model_copy(deep=True)
            for field_id, value in ground_truth.items()
        }
        self._unknown_fields = set(unknown_fields or set())

    def answer(self, field_id: str) -> FieldValue:
        if field_id not in self._ground_truth:
            raise KeyError(field_id)
        if field_id in self._unknown_fields:
            return FieldValue(
                value=None,
                status=FieldStatus.UNKNOWN_TO_USER,
                source="oracle",
            )
        return self._ground_truth[field_id].model_copy(
            update={
                "status": FieldStatus.OBSERVED,
                "source": "oracle",
            },
            deep=True,
        )

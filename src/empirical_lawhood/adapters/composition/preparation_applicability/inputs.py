"""Fresh one-use streams for fixed authenticated preparation input bytes."""

from dataclasses import dataclass, field, replace

from empirical_lawhood.adapters.composition.record_provider import RecordCampaignProvider
from empirical_lawhood.runtime.providers import ExternalInputPayload


@dataclass(slots=True)
class PreparationApplicabilityInputBytes:
    payload: bytes = field(repr=False)
    consumed: bool = False

    def chunks(self, maximum_chunk_bytes):
        if maximum_chunk_bytes <= 0 or self.consumed:
            raise ValueError("preparation input stream is invalid or already consumed")
        self.consumed=True
        for offset in range(0,len(self.payload),maximum_chunk_bytes):
            yield self.payload[offset:offset+maximum_chunk_bytes]

    def close(self):
        return


def preparation_input_template(**kwargs):
    """Retain only the existing capped, preauthenticated immutable bytes."""
    payload=ExternalInputPayload.from_bytes(**kwargs)
    return replace(payload,source=PreparationApplicabilityInputBytes(kwargs["payload"]))


class PreparationApplicabilityRecordProvider(RecordCampaignProvider):
    def external_inputs(self,plan,source_records=()):
        values=super().external_inputs(plan,source_records)
        refreshed=[]
        for value in values:
            if value in self.inputs:
                if not isinstance(value.source,PreparationApplicabilityInputBytes):
                    raise TypeError("preparation provider requires its code-owned immutable input template")
                value=replace(value,source=PreparationApplicabilityInputBytes(value.source.payload))
            refreshed.append(value)
        return tuple(refreshed)

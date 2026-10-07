"""Task-scoped reuse of already acknowledged canonical control publications."""

from typing import Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord


class ControlPublisher(Protocol):
    def publish_record(self, object_id: str, record: CanonicalRecord) -> ArtifactIdentity: ...


class ControlPublicationSession:
    """Avoid repeated writes within one task; never substitute for durable custody.

    Only successful publication acknowledgements are remembered. Recovery starts
    a new session and replays the underlying store's identity checks. The bounded
    map retains identities, not the potentially large scientific records.
    """

    def __init__(self, publisher: ControlPublisher, *, maximum_records: int) -> None:
        if maximum_records < 1:
            raise ValueError("publication session needs a positive record bound")
        self._publisher = publisher
        self._maximum_records = maximum_records
        self._published: dict[str, tuple[ObjectIdentity, ArtifactIdentity]] = {}

    def publish_record(self, object_id: str, record: CanonicalRecord) -> ArtifactIdentity:
        prior = self._published.get(object_id)
        subject = ObjectIdentity.from_record(object_id, record)
        if prior is not None:
            if prior[0] != subject:
                raise ValueError("control publication substituted an acknowledged identity")
            return prior[1]
        artifact = self._publisher.publish_record(object_id, record)
        # The acknowledgement may identify a lossless archive rather than the
        # logical record. Its authentication remains the publisher's job.
        # Bounded reuse is an optimization, never a new limit on valid publication.
        if len(self._published) < self._maximum_records:
            self._published[object_id] = (subject, artifact)
        return artifact

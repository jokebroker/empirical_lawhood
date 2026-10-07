"""Static method factories resolve only their declared canonical record/port."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT
from .extension_bundle import METHOD_CAPABILITIES, METHOD_COMPONENTS
from .provider import CANDIDATE_PAYLOAD_PORT, PreparationMethodProvider
from .topology import METHOD_CONFIG_TYPES, METHOD_ROLES
from .continuation import RETAINED_NATIVE_PORT, PreparationProjectionContinuation, PreparationRetainedNativePort
from .contracts import PreparationProjectionConfig
from .method_continuation import METHOD_CONTINUATION_ROLES, retained_method_port_key, method_continuation_type, METHOD_IMPORT_TYPES, _PreparationMethodImport, PreparationMethodContinuation, PreparationRetainedMethodPort


METHOD_BINDINGS = tuple(
    replace(
        executable(
            capability,
            components,
            (
                METHOD_CONFIG_TYPES[role],
                *((PreparationProjectionContinuation,) if role == "projection" else ()),
                *(
                    (method_continuation_type(role),)
                    if role in METHOD_CONTINUATION_ROLES
                    else ()
                ),
            ),
        ),
        required_issued_payload_schemas=(METHOD_CONFIG_TYPES[role].SCHEMA,),
        required_platform_port_keys=(CANDIDATE_PAYLOAD_PORT, retained_method_port_key(role))
        if role == "description"
        else (RETAINED_NATIVE_PORT,)
        if role == "projection"
        else (retained_method_port_key(role),)
        if role in METHOD_CONTINUATION_ROLES
        else (),
    )
    for role, capability, components in zip(
        METHOD_ROLES, METHOD_CAPABILITIES, METHOD_COMPONENTS, strict=True
    )
)


@dataclass(frozen=True, slots=True)
class PreparationMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> PreparationMethodProvider:
        index = METHOD_BINDINGS.index(self.binding)
        role = METHOD_ROLES[index]
        configs = tuple(r for r in records if type(r) is METHOD_CONFIG_TYPES[role])
        imports = tuple(r for r in records if isinstance(r, PreparationProjectionContinuation))
        method_imports = tuple(
            r
            for r in records
            if isinstance(r, (PreparationMethodContinuation, _PreparationMethodImport))
        )
        if (
            len(configs) != 1
            or len(imports) > 1
            or len(method_imports) > 1
            or (method_imports and role not in METHOD_CONTINUATION_ROLES)
            or len(records) != 1 + len(imports) + len(method_imports)
            or (imports and role != "projection")
            or tuple(p.port_key for p in platform_ports) != self.binding.required_platform_port_keys
        ):
            raise ValueError(
                "preparation method factory changes its exact config/platform port roster"
            )
        method_sources = None
        if role in METHOD_CONTINUATION_ROLES:
            method_sources = cast(
                PreparationRetainedMethodPort,
                next(
                    p.port
                    for p in platform_ports
                    if p.port_key == retained_method_port_key(role)
                ),
            )
            continuation = method_sources.continuation
            if (continuation is None) != (not method_imports):
                raise ValueError("method retained-input declaration and port disagree")
            if continuation is not None:
                declaration = method_imports[0]
                config_identity = ObjectIdentity.from_record(
                    str(getattr(configs[0], "config_id")), configs[0]
                )
                if role == "description":
                    if declaration != continuation or continuation.assessment != config_identity:
                        raise ValueError("method continuation changed its frozen assessment")
                elif (
                    not isinstance(declaration, _PreparationMethodImport)
                    or declaration.ROLE != role
                    or declaration.continuation
                    != ObjectIdentity.from_record(continuation.config_id, continuation)
                    or declaration.scientific_config != config_identity
                ):
                    raise ValueError("method import differs from its exact issued declaration")
        plane = None
        retained = None
        if role == "projection":
            value = platform_ports[0].port
            if not hasattr(value, "continuation") or not callable(
                getattr(value, "create_source", None)
            ):
                raise TypeError("projection requires its bounded retained-native port")
            retained = cast(PreparationRetainedNativePort, value)
            native_continuation = retained.continuation
            config = cast(PreparationProjectionConfig, configs[0])
            if (native_continuation is None) != (not imports):
                raise ValueError("projection retained-input declaration and port disagree")
            if native_continuation is not None and (
                imports[0].continuation
                != ObjectIdentity.from_record(native_continuation.config_id, native_continuation)
                or imports[0].projection_config
                != ObjectIdentity.from_record(config.config_id, config)
                or config.source_config != native_continuation.source_config
            ):
                raise ValueError("projection native imports differ from their issued identities")
        if role == "description":
            value = platform_ports[0].port
            if not all(
                callable(getattr(value, key, None))
                for key in ("publish_candidate_payload", "read_candidate_payload")
            ):
                raise TypeError(
                    "preparation description requires its bounded candidate publication/read port"
                )
            plane = cast(CandidatePayloadPlane, value)
        return PreparationMethodProvider(
            registry,
            METHOD_CAPABILITIES[index],
            configs[0],
            role,
            plane,
            retained_sources=retained,
            retained_method_sources=method_sources,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{DEVELOPMENT}.methods",
    "1.0.0",
    tuple(sorted(METHOD_BINDINGS, key=lambda b: b.binding_id)),
)
EXECUTABLE_BINDING_FACTORIES = tuple(
    PreparationMethodFactory(b) for b in EXECUTABLE_BINDING_CONTRIBUTION.bindings
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = tuple(
    sorted(
        set(
            (
                *METHOD_CONFIG_TYPES.values(),
                PreparationProjectionContinuation,
                PreparationMethodContinuation,
                *METHOD_IMPORT_TYPES.values(),
            )
        ),
        key=lambda t: t.SCHEMA,
    )
)

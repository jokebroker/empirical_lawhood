# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
"""Purpose-built executor enforcement double for service-level tests.

It runs in-process and is never production evidence.  The explicit typed
capability exists so tests can exercise the service's enforcing branch without
coupling that branch to the unrelated Git-identity switch.
"""

from __future__ import annotations

from dataclasses import replace

from empirical_lawhood.infrastructure.execution import DirectTaskExecutor
from empirical_lawhood.runtime.execution import ExecutorEnforcementCapability


ENFORCING_TEST_EXECUTOR_CAPABILITY = ExecutorEnforcementCapability(
    capability_id="enforcing-test-task-executor",
    cpu_time_limit=True,
    address_space_limit=True,
    wall_time_limit=True,
    source_scan_limit=True,
    scratch_limit=True,
    output_limit=True,
    network_isolation=False,
)


class EnforcingTestTaskExecutor:
    """Delegate execution while supplying explicit synthetic enforcement evidence."""

    enforcement_capability = ENFORCING_TEST_EXECUTOR_CAPABILITY

    def __init__(
        self,
        delegate: object | None = None,
        *,
        network_isolation: bool = False,
    ) -> None:
        self._delegate = delegate or DirectTaskExecutor()
        self.enforcement_capability = replace(
            ENFORCING_TEST_EXECUTOR_CAPABILITY,
            network_isolation=network_isolation,
        )

    def execute(
        self,
        runner: object,
        context: object,
        *,
        timeout_seconds: int,
    ) -> object:
        execute = getattr(self._delegate, "execute")
        return execute(runner, context, timeout_seconds=timeout_seconds)


__all__ = [
    "ENFORCING_TEST_EXECUTOR_CAPABILITY",
    "EnforcingTestTaskExecutor",
]

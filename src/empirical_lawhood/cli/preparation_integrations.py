# SPDX-License-Identifier: MPL-2.0
"""Preparation-family presentation over current typed inputs and guarded custody."""

from pathlib import Path
from typing import Annotated

import typer

from empirical_lawhood.cli.attempts import ATTEMPT_DIRECTORY_OPTION, attempt_stage, retained_command


def _helpers():
    from empirical_lawhood.cli import integrations
    return integrations


def _new_output(path: Path, plane, *, directory=False):
    """Validate before inputs or native work; API owners create the destination."""
    if directory:
        from empirical_lawhood.api.authoring_handoff import preflight_output_directory
        path=preflight_output_directory(directory=path)
    else:
        from empirical_lawhood.api.configuration import preflight_new_configuration_output
        path=preflight_new_configuration_output(path)
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Preparation output refuses symbolic links and parent traversal")
    path = path.absolute()
    if not path.is_relative_to(Path(plane.root.contract.canonical_path)):
        raise ValueError("Preparation output must be under the selected guarded external root")
    plane.root.verify(for_write=True)
    if path.exists():
        raise FileExistsError("Preparation output must use a fresh path")
    if not directory and not path.parent.is_dir():
        raise FileNotFoundError("Preparation output file requires its explicitly selected existing parent directory")
    plane.root.resolve(path.relative_to(Path(plane.root.contract.canonical_path)).as_posix(),for_write=True)
    return path


def _write(path, record):
    from empirical_lawhood.api.configuration import _write_new_file
    _write_new_file(path, record.canonical_bytes())


def register_preparation_commands(campaign_app: typer.Typer) -> None:
    @campaign_app.command("preparation-allocation-prepare")
    @retained_command("campaign preparation-allocation-prepare", closed_output=True, native=False)
    def allocation_prepare(
        allocation_id: Annotated[str, typer.Option("--allocation-id")],
        cohort_namespace: Annotated[str, typer.Option("--cohort-namespace")],
        phase: Annotated[str, typer.Option("--phase", help="ordinary D/E or constructed Q/E")],
        master_seed: Annotated[str, typer.Option("--master-seed", help="Explicit unsigned128-bit decimal or0x integer.")],
        output: Annotated[Path, typer.Option("--output")],
        constructed: Annotated[bool, typer.Option("--constructed")] = False,
        proposed_unrun: Annotated[bool, typer.Option("--proposed-unrun", help="Proposed allocation role only; exposure checks still decide eligibility.")] = False,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Derive the complete purpose roster from a selected master and namespace."""
        from empirical_lawhood.api.preparation_inputs import prepare_preparation_allocation
        h=_helpers()
        try:
            _,_,plane=h._storage()
            destination=_new_output(output,plane)
            value=int(master_seed,16 if master_seed.lower().startswith("0x") else 10)
            attempt_stage("explicit_preparation_allocation_derivation",native_contact="none")
            record=prepare_preparation_allocation(allocation_id=allocation_id,cohort_namespace=cohort_namespace,
                phase=phase,master_seed=value,constructed=constructed,proposed_unrun=proposed_unrun)
            _write(destination,record)
        except (OSError,RuntimeError,TypeError,ValueError) as error:
            h._error(error)
        h._report({"status":"ALLOCATED","output":str(destination),"allocation_sha256":record.fingerprint(),
            "independent_roots_assigned":len(record.roots),"evidence_role":record.exposure,"freshness_or_eligibility_granted":False})

    @campaign_app.command("preparation-conformance")
    @retained_command("campaign preparation-conformance", closed_output=True, native=True)
    def conformance(
        allocation: Annotated[Path, typer.Option("--allocation", exists=True, dir_okay=False)],
        conformance_id: Annotated[str, typer.Option("--conformance-id")],
        output: Annotated[Path, typer.Option("--output")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Exercise one native interval for three actual source kinds and both views."""
        from empirical_lawhood.api.preparation_conformance import preparation_native_conformance
        h = _helpers()
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane, directory=True)
            selected = h._configuration(allocation, "matrix-allocation")
            if selected.exposure != "EXPOSED_DEVELOPMENT_NONPROMOTABLE":
                raise ValueError("Software conformance uses explicit exposed development allocations")
            attempt_stage("bounded_preparation_native_conformance", native_contact="software_conformance")
            record = preparation_native_conformance(roots=selected.roots, output_dir=destination, conformance_id=conformance_id)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "SOFTWARE_CONFORMANCE", "output": str(destination), "report_sha256": record.fingerprint(), "scientific_qualification_performed": False})

    @campaign_app.command("preparation-exposure-inspect")
    @retained_command("campaign preparation-exposure-inspect", closed_output=True, native=False)
    def exposure_inspect(
        original_f: Annotated[Path, typer.Option("--original-f", exists=True, dir_okay=False)],
        inspection_id: Annotated[str, typer.Option("--inspection-id")],
        output: Annotated[Path, typer.Option("--output")],
        prior_input: Annotated[list[Path] | None, typer.Option("--prior-input", exists=True, file_okay=False, help="Repeat for every declared prior saved-current operand directory.")] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Inspect known public recipes and all explicitly supplied current banks."""
        from empirical_lawhood.api.preparation_inputs import inspect_preparation_exposure
        h = _helpers()
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane)
            lower = h._configuration(original_f, "original-f")
            selected = tuple((h._selected_input(bank / "arrays.canonical.json", maximum_bytes=4*1024**2, copy=False),
                h._selected_input(bank / "arrays.npz", maximum_bytes=64*1024**2, copy=False),
                h._selected_input(bank / "allocation.canonical.json", maximum_bytes=4*1024**2, copy=False)) for bank in (prior_input or ()))
            attempt_stage("current_effective_exposure_inspection", native_contact="none")
            record = inspect_preparation_exposure(prior_inputs=selected, inspection_id=inspection_id, original_f=lower, artifact_writer=plane)
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "INSPECTED", "output": str(destination), "inspection_sha256": record.fingerprint(), "prior_banks": len(selected), "eligibility_authority_granted": False})

    @campaign_app.command("preparation-input-publish")
    @retained_command("campaign preparation-input-publish", closed_output=True, native=False)
    def input_publish(
        key: Annotated[str, typer.Option("--key", help="lower, exposure or conformance")],
        input: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
        artifact_id: Annotated[str, typer.Option("--artifact-id")],
        relative_root: Annotated[str, typer.Option("--relative-root")],
        receipt_id: Annotated[str, typer.Option("--receipt-id")],
        output: Annotated[Path, typer.Option("--output")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Publish one actual outcome-blind input and its separate import receipt."""
        from empirical_lawhood.api.preparation_inputs import publish_preparation_input
        h = _helpers()
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane)
            consumers = {"lower": "original-f", "exposure": "preparation-exposure-inspection", "conformance": "preparation-native-conformance"}
            if key not in consumers:
                raise ValueError("Preparation input publication accepts lower, exposure or conformance")
            selected = h._configuration(input, consumers[key])
            attempt_stage("outcome_blind_preparation_input_publication", native_contact="none")
            record = publish_preparation_input(record=selected, key=key, logical_artifact_id=artifact_id,
                relative_root=relative_root, receipt_id=receipt_id, artifact_writer=plane)
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PUBLISHED_INPUT", "output": str(destination), "publication_sha256": record.fingerprint(), "scheduler_receipt_created": False, "scientific_qualification_performed": False})

    @campaign_app.command("preparation-selection-prepare")
    @retained_command("campaign preparation-selection-prepare", closed_output=True, native=False)
    def selection_prepare(
        config_id: Annotated[str, typer.Option("--config-id")],
        phase: Annotated[str, typer.Option("--phase", help="ordinary D/E or constructed Q/E")],
        allocation: Annotated[Path, typer.Option("--allocation", exists=True, dir_okay=False)],
        lower: Annotated[Path, typer.Option("--lower", exists=True, dir_okay=False)],
        exposure: Annotated[Path, typer.Option("--exposure", exists=True, dir_okay=False)],
        conformance: Annotated[Path, typer.Option("--conformance", exists=True, dir_okay=False)],
        output: Annotated[Path, typer.Option("--output")],
        constructed: Annotated[bool, typer.Option("--constructed")] = False,
        qualification: Annotated[Path | None, typer.Option("--qualification", exists=True, dir_okay=False)] = None,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Prepare the fixed full-size stage/source from actual published inputs."""
        from empirical_lawhood.api.preparation_inputs import create_preparation_selection
        from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access
        from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
        h = _helpers()
        try:
            root, _, plane = h._storage()
            destination = _new_output(output, plane)
            selected = h._configuration(allocation, "matrix-allocation")
            publications = tuple(h._configuration(path, "preparation-published-input") for path in (lower, exposure, conformance))
            q = None if qualification is None else h._configuration(qualification, "preparation-published-input")
            for value in (*publications, *((q,) if q is not None else ())):
                value.authenticate(plane, ExternalTaskReceiptStore(plane), result_access_authenticator=authenticate_preparation_result_access)
            attempt_stage("fixed_preparation_selection_preparation", native_contact="none")
            record = create_preparation_selection(root=root, config_id=config_id, phase=phase, allocation=selected,
                lower=publications[0], exposure=publications[1], conformance=publications[2], qualification=q, constructed=constructed)
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "PREPARED", "output": str(destination), "selection_sha256": record.fingerprint(), "independent_units": len(selected.roots), "source_contacted": False})

    @campaign_app.command("preparation-q-bind")
    @retained_command("campaign preparation-q-bind", closed_output=True, native=False)
    def q_bind(
        run_id: Annotated[str, typer.Option("--run-id")],
        receipt_id: Annotated[str, typer.Option("--receipt-id")],
        issued_study_id: Annotated[str, typer.Option("--issued-study-id")],
        execution_authority_id: Annotated[str, typer.Option("--execution-authority-id")],
        reveal_authority_id: Annotated[str, typer.Option("--reveal-authority-id")],
        output: Annotated[Path, typer.Option("--output")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Bind the exact current Q terminal report under its issued reveal chain."""
        from empirical_lawhood.api.preparation_inputs import bind_constructed_preparation_q
        from empirical_lawhood.api.preparation_result_access import preparation_result_access
        h = _helpers()
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane)
            access = preparation_result_access(run_id=run_id, issued_study_id=issued_study_id,
                execution_authority_id=execution_authority_id, reveal_authority_id=reveal_authority_id, writer=plane)
            attempt_stage("exact_current_constructor_result_binding", native_contact="none")
            record = bind_constructed_preparation_q(run_id=run_id, receipt_id=receipt_id, artifact_writer=plane, result_access=access)
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "RETAINED", "output": str(destination), "qualified": record.record.constructor_qualified, "disposition": record.record.disposition, "report_sha256": record.record.fingerprint(), "scientific_execution_performed_by_reader": False})

    @campaign_app.command("preparation-panel-bind")
    @retained_command("campaign preparation-panel-bind", closed_output=True, native=False)
    def panel_bind(
        run_id: Annotated[str, typer.Option("--run-id")],
        root_id: Annotated[str, typer.Option("--root-id")],
        prefix_receipt_id: Annotated[str, typer.Option("--prefix-receipt-id")],
        assay_receipt_id: Annotated[str, typer.Option("--assay-receipt-id")],
        lower_receipt_id: Annotated[str, typer.Option("--lower-receipt-id")],
        measurement_receipt_id: Annotated[str, typer.Option("--measurement-receipt-id")],
        issued_study_id: Annotated[str, typer.Option("--issued-study-id")],
        execution_authority_id: Annotated[str, typer.Option("--execution-authority-id")],
        reveal_authority_id: Annotated[str, typer.Option("--reveal-authority-id")],
        output: Annotated[Path, typer.Option("--output")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Retain one current panel with all four exact native/readout task receipts."""
        from empirical_lawhood.api.preparation_inputs import bind_ordinary_preparation_panel
        from empirical_lawhood.api.preparation_result_access import preparation_result_access
        h = _helpers()
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane)
            access = preparation_result_access(run_id=run_id, issued_study_id=issued_study_id,
                execution_authority_id=execution_authority_id, reveal_authority_id=reveal_authority_id, writer=plane)
            attempt_stage("exact_current_ordinary_panel_binding", native_contact="none")
            record = bind_ordinary_preparation_panel(run_id=run_id, root_id=root_id,
                receipt_ids=(prefix_receipt_id, assay_receipt_id, lower_receipt_id, measurement_receipt_id), artifact_writer=plane, result_access=access)
            _write(destination, record)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "RETAINED", "output": str(destination), "root_id": record.prefix.root_id, "panel_sha256": record.panel.fingerprint(), "complete": record.measurement.complete, "scientific_execution_performed_by_reader": False})

    @campaign_app.command("preparation-result")
    @retained_command("campaign preparation-result", closed_output=True, native=False)
    def result_read(
        run_id: Annotated[str, typer.Option("--run-id")],
        receipt_id: Annotated[str, typer.Option("--receipt-id")],
        issued_study_id: Annotated[str, typer.Option("--issued-study-id")],
        execution_authority_id: Annotated[str, typer.Option("--execution-authority-id")],
        reveal_authority_id: Annotated[str, typer.Option("--reveal-authority-id")],
        output: Annotated[Path, typer.Option("--output")],
        constructed: Annotated[bool, typer.Option("--constructed")] = False,
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Retain the exact current scientific report, including negative/stopped states."""
        from empirical_lawhood.api.preparation_inputs import bind_preparation_report, read_preparation_report
        from empirical_lawhood.api.preparation_result_access import preparation_result_access
        h=_helpers()
        try:
            _,_,plane=h._storage()
            destination=_new_output(output,plane)
            access=preparation_result_access(run_id=run_id,issued_study_id=issued_study_id,
                execution_authority_id=execution_authority_id,reveal_authority_id=reveal_authority_id,writer=plane)
            attempt_stage("exact_current_preparation_report_read",native_contact="none")
            record=bind_preparation_report(run_id=run_id,receipt_id=receipt_id,constructed=constructed,
                artifact_writer=plane,result_access=access)
            summary=read_preparation_report(record,artifact_writer=plane)
            _write(destination,record)
        except (OSError,RuntimeError,TypeError,ValueError) as error:
            h._error(error)
        h._report({"status":"RETAINED","output":str(destination),"result":summary})

    @campaign_app.command("preparation-panel-export")
    @retained_command("campaign preparation-panel-export", closed_output=True, native=False)
    def panel_export(
        stage: Annotated[Path, typer.Option("--stage", exists=True, dir_okay=False)],
        original_f: Annotated[Path, typer.Option("--original-f", exists=True, dir_okay=False)],
        panel: Annotated[list[Path], typer.Option("--panel", exists=True, dir_okay=False, help="Repeat in the complete assigned independent-root order.")],
        output: Annotated[Path, typer.Option("--output")],
        attempt_dir: Annotated[Path | None, ATTEMPT_DIRECTORY_OPTION] = None,
    ) -> None:
        """Authenticate all D32 or E64 panels and export complete exposed operands."""
        from empirical_lawhood.api.preparation_screens import ordinary_preparation_operands_export
        h = _helpers()
        try:
            _, _, plane = h._storage()
            destination = _new_output(output, plane, directory=True)
            selected = h._configuration(stage, "preparation-applicability-stage")
            lower = h._configuration(original_f, "original-f")
            panels = tuple(h._configuration(path, "preparation-panel-publication") for path in panel)
            attempt_stage("complete_current_ordinary_panel_export", native_contact="none")
            destination.mkdir(parents=True, exist_ok=False)
            record = ordinary_preparation_operands_export(stage=selected, original_f=lower, panels=panels,
                destination=destination, artifact_writer=plane)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            h._error(error)
        h._report({"status": "EXPOSED_DEVELOPMENT_OPERANDS", "output": str(destination), "manifest_sha256": record.fingerprint(), "independent_units": len(selected.root_ids), "follow_on_status": "UNENTERED"})

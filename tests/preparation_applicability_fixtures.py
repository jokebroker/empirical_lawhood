"""Explicitly synthetic current authoring inputs; no operative eligibility."""
from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation, MatrixRootAllocation, MatrixPurposeSeed, MATRIX_NATIVE_PURPOSES
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.adapters.methods.preparation_applicability.config import ORIGINAL_F_SCHEMA, PreparationApplicabilityUpstream
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import PreparationApplicabilityExposure


def synthetic_identity(name):
    return ObjectIdentity(name, 'empirical-lawhood/tests/exposed-synthetic-input', '1.0.0', 'a'*64)


def allocation(count, *, constructed=False, phase='q'):
    roots = []
    offset = 100000 if phase.lower() == "e" else 0
    for i in range(count):
        independent = [MatrixPurposeSeed(name, offset+100*(i+1)+j) for j,name in enumerate(MATRIX_NATIVE_PURPOSES)]
        conditioning = ()
        if constructed:
            from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import FIXED_PROBE_SEED
            conditioning = (MatrixPurposeSeed('passive-probes',FIXED_PROBE_SEED,'PCG64DXSM'),)
        else:
            independent.append(MatrixPurposeSeed('passive-probes',(offset+10000+i)<<128,'PCG64DXSM'))
        roots.append(MatrixRootAllocation(f'exposed.synthetic.{phase}.unit-{i:03d}', 'constructed' if constructed else 'q2', tuple(sorted(independent,key=lambda s:s.purpose_id)), conditioning_seeds=conditioning))
    return MatrixAllocation(f'exposed.synthetic.{phase}.allocation',tuple(roots),bootstrap_seed=offset+999999,exposure='PROPOSED_UNRUN' if constructed else 'EXPOSED_DEVELOPMENT_NONPROMOTABLE')


def stage_and_source(phase='Q'):
    from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import ConstructedPreparationStage, ConstructedPreparationSource
    from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationDesign
    design = ConstructedPreparationDesign()
    source = ConstructedPreparationSource('1'*64,'2'*64,synthetic_identity('exposed.synthetic.native'),synthetic_identity('exposed.synthetic.conformance'))
    lower = PreparationApplicabilityUpstream('lower',ArtifactIdentity('exposed.synthetic.lower','exposed-test-only',ORIGINAL_F_SCHEMA,'3'*64,'application/json',16),synthetic_identity('exposed.synthetic.lower.receipt'),'exposed.synthetic.lower.run')
    prior = PreparationApplicabilityExposure(synthetic_identity('exposed.synthetic.exposure'),synthetic_identity('exposed.synthetic.exposure.receipt'),('exposed.synthetic.previous-unit',),('seed.pcg64.ffffffffffffffffffffffffffffffff',))
    inputs = (lower,)
    if phase=='E':
        inputs += (PreparationApplicabilityUpstream('qualification',ArtifactIdentity('exposed.synthetic.qualification','exposed-test-only','empirical-lawhood/constructed-preparation-applicability/report','4'*64,'application/json',16),synthetic_identity('exposed.synthetic.q.receipt'),'exposed.synthetic.q.run'),)
    stage = ConstructedPreparationStage(f'exposed.synthetic.{phase.lower()}.stage',design,phase,allocation(8 if phase=='Q' else 32,constructed=True,phase=phase.lower()),inputs,prior,ObjectIdentity.from_record('exposed.synthetic.source',source))
    return stage, source

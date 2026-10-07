"Additive, construct-validity-first contracts for target construct validation.\n\nThis package binds the registered margin structural recurrence forecast\nruntime and keeps target-native construct records separate from later\nmapping and evaluation records.\n"

from .contracts import TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID, TargetConstructValidationStructuralRelationCodebook, TargetConstructValidationTargetRelationBinding, primary_relation_codebook
from .donor import TargetConstructValidationDonorBinding, current_donor_binding

__all__ = [
    "TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID",
    'TargetConstructValidationDonorBinding',
    'TargetConstructValidationStructuralRelationCodebook',
    'TargetConstructValidationTargetRelationBinding',
    "current_donor_binding",
    "primary_relation_codebook",
]

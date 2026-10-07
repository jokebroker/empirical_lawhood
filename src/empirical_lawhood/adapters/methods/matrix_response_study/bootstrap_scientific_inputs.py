"""Fixed complete numerical allocations for distinct paired comparison roles.

These constants preserve source-controlled bootstrap allocations. They do not
identify an observed cohort, authenticate history or grant authority.
"""

FIXED_MATRIX_RESPONSE_BOOTSTRAP_SEEDS = (
    ('transient-intervention-risk', '0083dd4a361b1f0f8e1005f099df2a9a969d05d7f00038e8f3566cd900c9ce77'),
    ('transient-intervention-residence', '40853326be70d6ed0e2f8e166c05015d26dcd06f01ed43f8c9df9faa54fea3bd'),
    ('causal-intersection-full-risk', '9beae165359e72362c9d4142296c99ca91feaeaca6153d094f6bbd533aec5bed'),
    ('causal-intersection-full-residence', 'cd81f733c28cb98f9a6c8298a16aab385de214018a18c7da26284c26eee2b7cf'),
    ('causal-intersection-response-risk', '49d7b5a90679aaca30f3dd3e1dc2339ab5ba7f5caaeee9cd108dde52fe3d1cba'),
    ('causal-intersection-response-residence', '01aeaadb788038b9538febda4c3faf6872d059c6deb5b6fd750f9a34b94ee7cc'),
    ('causal-intersection-structure-risk', '23de96b5c744e061314844fd91743d8b0a19e0aa59cb0d27d2d4c01f2e756004'),
    ('causal-intersection-structure-residence', '279175df5b610b81e9a7b356624a2b3c011adff447e7338636618660981c3c38'),
    ('prospective-control-simultaneous-family', 'b5f27f5191da79ea66f32f33290e66be529cdd58a13c24dd9d8beef924688cc3'),
)


def fixed_matrix_response_bootstrap_seed_sha256(scientific_role: str) -> str:
    for role, seed in FIXED_MATRIX_RESPONSE_BOOTSTRAP_SEEDS:
        if role == scientific_role:
            return seed
    raise ValueError("matrix response bootstrap role is outside its fixed numerical allocation")

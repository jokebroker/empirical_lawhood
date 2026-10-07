# SPDX-License-Identifier: MPL-2.0
"""Invalid direct conformance outputs refuse before entering the native marcher."""

from dataclasses import replace

import pytest

from empirical_lawhood.api.preparation_conformance import preparation_native_conformance
from empirical_lawhood.api.preparation_inputs import prepare_preparation_allocation


@pytest.mark.parametrize('destination_kind', ('file-parent', 'symlink-parent', 'traversal'))
def test_conformance_invalid_output_refuses_before_native_contact(tmp_path, monkeypatch, destination_kind):
    ordinary = prepare_preparation_allocation(allocation_id='synthetic.output.ordinary',
        cohort_namespace='synthetic.output.ordinary', phase='D', master_seed=8541)
    constructed = prepare_preparation_allocation(allocation_id='synthetic.output.constructed',
        cohort_namespace='synthetic.output.constructed', phase='Q', master_seed=8542, constructed=True)
    roots = (ordinary.roots[0], replace(ordinary.roots[1], cohort='cir1'), constructed.roots[0])
    if destination_kind == 'file-parent':
        parent = tmp_path / 'file'
        parent.write_bytes(b'preserved')
        output = parent / 'nested' / 'result'
    else:
        parent = tmp_path / 'parent'
        parent.mkdir()
        if destination_kind == 'symlink-parent':
            link = tmp_path / 'link'
            link.symlink_to(parent, target_is_directory=True)
            output = link / 'result'
        else:
            output = parent / '..' / 'result'

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid output entered native computation')

    monkeypatch.setattr('empirical_lawhood.adapters.simulators.prepared_response.source.march_native_intervals', forbidden)
    with pytest.raises((NotADirectoryError, ValueError)):
        preparation_native_conformance(roots=roots, output_dir=output,
            conformance_id='synthetic.refused-native-conformance')
    assert not (tmp_path / 'result').exists()
    assert not (tmp_path / 'parent' / 'result').exists()

# SPDX-License-Identifier: MPL-2.0
"""Contain publication outputs in explicitly selected mounted external storage."""
from pathlib import Path


def add_storage_arguments(parser):
    parser.add_argument('--storage-root', type=Path,
                        required=True)
    parser.add_argument('--storage-mount', type=Path,
                        required=True)


def external_directory(target, args, parser):
    target, root, mount = target.resolve(), args.storage_root.resolve(), args.storage_mount.resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (not mount.is_mount() or mount not in root.parents or root not in target.parents
            or target == checkout or checkout in target.parents):
        parser.error('Outputs must be outside the checkout and contained in the selected project root under its actual mounted storage.')
    return target

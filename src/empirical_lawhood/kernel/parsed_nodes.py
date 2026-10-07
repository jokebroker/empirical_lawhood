# SPDX-License-Identifier: MPL-2.0
"""Owned structural mapping reuse, bounded to one parsed document."""

from __future__ import annotations


class ParsedNodeInterner:
    """Retain equal mappings within a parse; scalar types remain distinct.

    This owns parsed syntax only, never authority or scientific record results.
    Callers must still check every serialized occurrence and count aliased nodes.
    """

    def __init__(self) -> None:
        self._mappings: dict[object, dict[str, object]] = {}
        self._owned: set[int] = set()

    def _key(self, value: object) -> object:
        if isinstance(value, dict):
            return (dict, id(value))
        if isinstance(value, list):
            return (list, tuple(self._key(item) for item in value))
        return (type(value), value)

    def mapping(self, value: dict[str, object]) -> dict[str, object]:
        structure = tuple(sorted((key, self._key(item)) for key, item in value.items()))
        previous = self._mappings.get(structure)
        if previous is not None:
            return previous
        self._mappings[structure] = value
        self._owned.add(id(value))
        return value

    def owns(self, value: object) -> bool:
        return id(value) in self._owned

    def tree(self, value: object) -> object:
        """Intern an already validated editable tree without changing scalars."""
        if isinstance(value, list):
            return [self.tree(item) for item in value]
        if isinstance(value, dict):
            children = {key: self.tree(item) for key, item in value.items()}
            # YAML permits other scalar key types; leave those malformed record
            # mappings to their existing field/schema validator.
            if all(isinstance(key, str) for key in children):
                return self.mapping(children)
            return children
        return value

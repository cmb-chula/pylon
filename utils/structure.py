import collections


def apply_structure(structure, fn):
    """Apply fn onto a structure and return the transformed structure

    Args:
        structure: list, dict, singleton
        fn: a transformation function
    """
    if isinstance(structure, collections.Mapping):
        # support dict
        return {key: apply_structure(structure[key], fn) for key in structure}
    elif isinstance(structure, collections.Sequence):
        # support list and tuples
        _out = [apply_structure(e, fn) for e in structure]
        if isinstance(structure, tuple):
            _out = tuple(_out)
        return _out
    else:
        # NOTE: single object
        return fn(structure)

"""Selection between SpaGCN 1.2.7 numerics and the corrected modern model."""

SPAGCN_1_2_7 = "spagcn-1.2.7"
MODERN = "modern"
COMPAT_MODES = (SPAGCN_1_2_7, MODERN)
DEFAULT_COMPAT = SPAGCN_1_2_7


def resolve_compat(compat):
    """Return a validated compatibility mode, using the package default for None."""
    if compat is None:
        return DEFAULT_COMPAT
    if compat not in COMPAT_MODES:
        raise ValueError(
            f"compat must be one of {COMPAT_MODES!r} (or None for the default "
            f"{DEFAULT_COMPAT!r}); got {compat!r}"
        )
    return compat


def default_init(compat):
    """Graph initialization used by each mode when ``init`` is not given."""
    return "louvain" if resolve_compat(compat) == SPAGCN_1_2_7 else "leiden"

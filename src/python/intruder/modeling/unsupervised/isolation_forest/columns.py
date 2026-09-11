"""Feature-block registry and output column names for isolation-forest scoring.

A "block" is a named group of feature columns with a shared prerequisite:
``intrinsic`` needs nothing beyond the input table, ``flank`` needs the VCF the
insertions came from (to re-read the ALT sequence), and ``cohort`` needs
multiple samples grouped by ``SVID``. ``mappability`` is registered as a stub
so ``--blocks mappability`` fails with "not implemented" rather than "unknown
block" -- there are no reference tracks in this repo to compute it from yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureBlock:
    """One named group of feature columns and what it needs to compute them."""

    name: str
    columns: tuple[str, ...]
    requires: str | None = None   # None, "vcf", or "cohort"
    implemented: bool = True


# Single-sample intrinsic properties of the call itself.
INTRINSIC_FEATURES = (
    "purity",
    "log_motif_length",
    "log_rep_length",
    "repeat_coverage",
    "log_depth_total",
    "motif_gc",
    "motif_entropy",
)

# `vaf` = DV / (DV + DR), the read-support split. Not in INTRINSIC_FEATURES
# above: it only survives in the raw stage-01 `depth` column ("DV,DR") --
# `filter_ins_trf.py` collapses it to a summed total before novelty annotation
# ever runs, so by the time most candidate tables exist to score, the split is
# already gone. `features.build_features` computes it when the split is still
# there and leaves it out (never imputed) when it isn't -- structural
# missingness like `flank` below, gated by `drop_vaf` rather than a block name
# since it needs no external resource, just an unmangled `depth` column.
VAF_FEATURE = "vaf"

# `rep_start`/`rep_end` are offsets *inside* the insertion, so the flanks are
# `ALT[:rep_start]` and `ALT[rep_end:]`, each scored against the motif with
# `trcore.motifs.tiling_distance`.
FLANK_FEATURES = (
    "flank_5p_identity",
    "flank_3p_identity",
    "flank_5p_len",
    "flank_3p_len",
)

# Cohort recurrence, grouped by `SVID`. Needs multi-sample input.
COHORT_FEATURES = (
    "n_carriers",
    "rep_length_cv",
    "n_distinct_canonical_motifs",
)

FEATURE_BLOCKS: dict[str, FeatureBlock] = {
    "intrinsic": FeatureBlock("intrinsic", INTRINSIC_FEATURES),
    "flank": FeatureBlock("flank", FLANK_FEATURES, requires="vcf"),
    "cohort": FeatureBlock("cohort", COHORT_FEATURES, requires="cohort"),
    "mappability": FeatureBlock("mappability", (), implemented=False),
}

DEFAULT_BLOCKS = ("intrinsic",)

# Structural missingness (see features.py): a call whose repeat fills the
# whole insertion has no flanking sequence to score, so it is fit and scored
# on `intrinsic` alone rather than being handed a sentinel value for `flank`.
UNFLANKED_BLOCKS = ("intrinsic",)
FLANKED_BLOCKS = ("intrinsic", "flank")


def resolve_blocks(names) -> tuple[FeatureBlock, ...]:
    """``--blocks`` values -> :class:`FeatureBlock`\\ s, in the order requested."""
    blocks = []
    for name in names:
        block = FEATURE_BLOCKS.get(name)
        if block is None:
            raise KeyError(f"unknown feature block {name!r}; expected one of "
                           f"{sorted(FEATURE_BLOCKS)}")
        if not block.implemented:
            raise NotImplementedError(f"feature block {name!r} is a registered "
                                      f"stub, not implemented")
        blocks.append(block)
    return tuple(blocks)


def feature_columns(blocks) -> tuple[str, ...]:
    """All feature columns across ``blocks``, in a stable, deduplicated order."""
    seen: dict[str, None] = {}
    for block in blocks:
        for column in block.columns:
            seen.setdefault(column, None)
    return tuple(seen)


def shap_column(feature: str) -> str:
    """Name of the SHAP attribution column for one feature.

    Emitted negated relative to what `shap.TreeExplainer` returns for an
    `IsolationForest` -- see `explain.py` -- so "positive" reads the way a
    biologist expects: positive SHAP = this feature drove the call to look
    *anomalous*.
    """
    return f"shap_anom_{feature}"


# Non-feature output columns appended by `annotate`, in the order they are
# written after the feature/SHAP columns.
OUTPUT_META_COLUMNS = (
    "if_stratum", "if_score", "if_h", "if_percentile", "if_mode", "if_blocks",
)

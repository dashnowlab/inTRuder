"""Re-read ALT sequences from the VCF and shape flank features for scoring.

The flank geometry and identity scoring themselves live in
``trcore.flanks`` -- they are pure sequence arithmetic future modeling steps
may also want. This module's own job is narrower: read the insertion
sequences back out of a VCF (the same REF-trimmed-off-ALT convention
``sv_trfcaller.py`` uses) and lay :class:`trcore.flanks.Flanks` out into
:data:`columns.FLANK_FEATURES`, this package's feature-table schema.
"""

from __future__ import annotations

import os

import pandas as pd
from cyvcf2 import VCF

from intruder.trcore.flanks import flanks_of

from .columns import FLANK_FEATURES


def _insertion_sequences(vcf_path: str | os.PathLike[str]) -> dict[str, str]:
    """``SVID`` -> inserted sequence, for every ``SVTYPE=INS`` record in the VCF."""
    sequences: dict[str, str] = {}
    for variant in VCF(str(vcf_path)):
        if not variant.is_sv or variant.INFO.get("SVTYPE") != "INS":
            continue
        if not variant.ALT:
            continue
        alt = variant.ALT[0]
        ref = variant.REF
        if ref == alt[:len(ref)]:
            alt = alt[len(ref):]
        sequences[variant.ID] = alt
    return sequences


def compute_flank_features(frame: pd.DataFrame,
                           vcf_path: str | os.PathLike[str]) -> pd.DataFrame:
    """:data:`columns.FLANK_FEATURES` for every row of ``frame``, keyed by ``SVID``.

    ``NaN`` for a row whose repeat fills the whole insertion, or whose ``SVID``
    is not in the VCF -- that is not a failure here, the caller
    (:func:`features.build_features`) routes those rows to the intrinsic-only
    partition rather than treating the gap as a feature value.
    """
    required = ("SVID", "motif", "rep_start", "rep_end")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"the 'flank' block needs column(s) {missing}")

    sequences = _insertion_sequences(vcf_path)
    out = pd.DataFrame(index=frame.index, columns=list(FLANK_FEATURES), dtype="float64")
    for idx, row in frame.iterrows():
        sequence = sequences.get(row["SVID"])
        if sequence is None:
            continue
        flanks = flanks_of(sequence, int(row["rep_start"]), int(row["rep_end"]),
                           str(row["motif"]))
        if flanks.five_prime.sequence:
            out.at[idx, "flank_5p_identity"] = flanks.five_prime.identity
            out.at[idx, "flank_5p_len"] = flanks.five_prime.length
        if flanks.three_prime.sequence:
            out.at[idx, "flank_3p_identity"] = flanks.three_prime.identity
            out.at[idx, "flank_3p_len"] = flanks.three_prime.length
    return out

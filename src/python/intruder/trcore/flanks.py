"""The sequence flanking a tandem repeat call, and how repeat-like it is.

``rep_start``/``rep_end`` (as written by ``sv_trfcaller.py``) are 0-based
offsets *inside* an insertion sequence, not genome coordinates, so the flanks
are ``sequence[:rep_start]`` and ``sequence[rep_end:]`` of that same insertion.

A real repeat array typically decays at its edges -- degenerate copies of the
motif bleeding into the flanking sequence -- while an array with unrelated
flanks that start and stop precisely at the call's boundaries is comparatively
suspicious. ``flank_identity`` is the score behind that read: how much of a
flank still looks like the motif, tiled across it.

This module is pure sequence geometry -- no file I/O, no pandas -- so any
future modeling step that reasons about a call's flanks shares one definition
of what a flank is, rather than each re-deriving it from ``rep_start``/
``rep_end`` its own way. Reading the flanking sequence out of a VCF, and
shaping the result into a feature table, is the caller's job -- see
``modeling.unsupervised.isolation_forest.flanks``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .motifs import tiling_distance


def flank_identity(motif: str, flank: str) -> float:
    """Fraction of ``flank`` that matches ``motif`` tiled across it; ``nan`` if empty.

    Built on :func:`motifs.tiling_distance`, which needs at least two whole
    copies of ``motif`` to fit in ``flank``; a shorter flank is inherently
    ambiguous evidence and reads as the worst case (``0.0``), not a missing
    value -- an unrelated-looking flank is exactly what a clean, non-decaying
    boundary looks like.
    """
    if not flank:
        return float("nan")
    distance = tiling_distance(motif, flank, len(flank))
    return 1.0 - (min(distance, len(flank)) / len(flank))


@dataclass(frozen=True)
class Flank:
    """One side of the sequence flanking a repeat call, and how repeat-like it is."""

    sequence: str
    identity: float   # flank_identity(motif, sequence); nan for an empty flank

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class Flanks:
    """Both sides of the sequence flanking one repeat call inside its insertion."""

    five_prime: Flank
    three_prime: Flank

    @property
    def has_flank(self) -> bool:
        """Whether there is any flanking sequence at all.

        False when the repeat call fills the whole insertion (``rep_start ==
        0`` and ``rep_end == len(sequence)``) -- structurally, not by a
        sentinel value; see the caller for how that case is handled.
        """
        return bool(self.five_prime.sequence or self.three_prime.sequence)


def flanks_of(sequence: str, rep_start: int, rep_end: int, motif: str) -> Flanks:
    """The :class:`Flanks` around ``sequence[rep_start:rep_end]`` in ``sequence``."""
    five, three = sequence[:rep_start], sequence[rep_end:]
    return Flanks(Flank(five, flank_identity(motif, five)),
                 Flank(three, flank_identity(motif, three)))

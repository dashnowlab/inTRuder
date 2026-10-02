"""An in-memory interval index over a reference tandem-repeat catalogue.

Rows are held column-wise in numpy arrays, globally sorted by ``(chrom, start)``;
``_bounds`` maps a contig onto its slice. Overlap search is a ``searchsorted`` to
the last candidate start followed by a walk left that stops once ``max_end`` (a
per-contig prefix maximum of the interval ends) can no longer reach the query.

Motif strings are interned: ``_seq_id`` indexes into ``_motifs``/``_canon``, so
the 1.05M rows of hg38 ``simpleRepeat`` store only their 516k distinct sequences
and each distinct sequence is canonicalised exactly once.

Given a reference coordinate and a motif, a locus is classified as:

    known        a nearby reference repeat carries a matching motif -- exactly,
                 or within the :class:`trcore.motifs.MotifTolerance` in force
    novel_motif  reference repeats are annotated nearby, but none with this motif
    novel_locus  no reference repeat is annotated within the search window
    unscreened   this catalogue has no rows on this contig at all, so it has no
                 opinion either way

``unscreened`` exists because the two are not the same claim. A catalogue that
stops at the primary assembly (TRExplorer v2 carries 25 contigs; UCSC
``simpleRepeat`` carries 702) would otherwise report every alt, random and decoy
contig as ``novel_locus`` -- novelty manufactured out of missing coverage.

The catalogue can come from any platform (see :mod:`novelty.platforms`); this
module only assumes the normalised schema, and carries whatever optional
annotations the platform supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from intruder.trcore.coords import interval_distance, normalize_chrom
from intruder.trcore.motifs import (
    DEFAULT_EQUIVALENCE,
    DEFAULT_TOLERANCE,
    MATCH_NONE,
    MotifEquivalence,
    MotifTolerance,
    canonical_motif,
)

from .platforms import (
    ANNOTATION_COLUMNS,
    canonical_motifs,
    normalize_chroms,
)

# Ordered least to most novel, which is also the order a combined verdict
# resolves in: the first status any catalogue reports wins. ``unscreened`` is
# last because it is not a claim about the locus at all -- it says this
# catalogue has no rows on this contig, so it has no opinion, and any catalogue
# that does have one outranks it.
STATUSES = ("known", "novel_motif", "novel_locus", "unscreened")

# The status meaning "this catalogue cannot speak about this contig".
UNSCREENED = "unscreened"

# Columns :meth:`RepeatCatalog.screen_frame` produces, before the platform prefix.
RESULT_COLUMNS = (
    "novelty", "n_nearby", "start", "end", "distance", "motif", "canonical",
    "motif_edits", "match",
)

_INT_ANNOTATIONS = ("period", "consensus_size", "per_match", "per_indel")


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReferenceRepeat:
    """One catalogue row, with 0-based half-open coordinates.

    ``annotations`` holds whichever of :data:`novelty.platforms.ANNOTATION_COLUMNS`
    the platform provided -- a BED catalogue supplies none of them.
    """

    chrom: str
    start: int
    end: int
    motif: str
    canonical: str
    annotations: dict[str, float] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Hit:
    """A reference repeat near the query, with how far off it is."""

    repeat: ReferenceRepeat
    distance: int      # bp from the query point to the interval; 0 when inside
    motif_edits: int   # 0 when the motifs are equivalent
    match: str = MATCH_NONE   # which tolerance rule accepted the motif, if any

    @property
    def motif_matches(self) -> bool:
        return self.match != MATCH_NONE


@dataclass(frozen=True)
class Verdict:
    """Result of screening one (chrom, pos, motif) against one catalogue."""

    chrom: str
    point: int                     # 0-based query coordinate
    motif: str
    canonical: str
    status: str                    # one of STATUSES
    n_nearby: int                  # reference repeats within the window
    best: Hit | None               # best motif match, else nearest repeat

    @property
    def is_novel(self) -> bool:
        return self.status != "known"


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #

class RepeatCatalog:
    """Interval index over one reference tandem-repeat catalogue."""

    def __init__(self, equivalence: MotifEquivalence = DEFAULT_EQUIVALENCE,
                 platform: str = "bed") -> None:
        self.equivalence = equivalence
        self.platform = platform
        self._bounds: dict[str, tuple[int, int]] = {}
        self._starts = np.empty(0, dtype=np.int64)
        self._ends = np.empty(0, dtype=np.int64)
        self._max_end = np.empty(0, dtype=np.int64)
        self._seq_id = np.empty(0, dtype=np.int32)
        self._annots: dict[str, np.ndarray] = {}
        self._motifs: list[str] = []
        self._canon: list[str] = []

    def __len__(self) -> int:
        return int(self._starts.size)

    @property
    def annotations(self) -> tuple[str, ...]:
        """Optional annotation columns this catalogue carries, in schema order."""
        return tuple(c for c in ANNOTATION_COLUMNS if c in self._annots)

    def __repr__(self) -> str:
        return (f"<RepeatCatalog {self.platform} {len(self):,} repeats "
                f"{len(self._bounds):,} contigs "
                f"[{self.equivalence.describe()}]>")

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *,
                   equivalence: MotifEquivalence = DEFAULT_EQUIVALENCE,
                   platform: str = "bed") -> RepeatCatalog:
        """Build from a normalised catalogue frame (see :mod:`novelty.platforms`)."""
        index = cls(equivalence=equivalence, platform=platform)
        index._build(frame)
        return index

    def _build(self, frame: pd.DataFrame) -> None:
        """Populate the columnar arrays from a normalised catalogue frame."""
        frame = frame.reset_index(drop=True)
        chrom = pd.Categorical(frame["chrom"].astype(object))
        starts = frame["start"].to_numpy(dtype=np.int64)

        # Sort by (chrom, start) so each contig is one contiguous slice. Sorting
        # the category codes avoids a million object-dtype string comparisons.
        codes = chrom.codes.astype(np.int64)
        order = np.lexsort((starts, codes))
        codes = codes[order]

        self._starts = starts[order]
        self._ends = frame["end"].to_numpy(dtype=np.int64)[order]

        self._annots = {}
        for column in ANNOTATION_COLUMNS:
            if column not in frame.columns:
                continue
            values = frame[column].to_numpy(dtype=np.float64)[order]
            if column in _INT_ANNOTATIONS:
                self._annots[column] = np.nan_to_num(values, nan=-1).astype(np.int32)
            else:
                self._annots[column] = values.astype(np.float32)

        # Intern motifs: canonicalise each distinct sequence exactly once.
        motif = frame["motif"].to_numpy(dtype=object)[order]
        seq_id, uniques = pd.factorize(pd.Series(motif, dtype=object), sort=False)
        self._motifs = [str(m) for m in uniques]
        self._canon = [canonical_motif(m, self.equivalence) for m in self._motifs]
        self._seq_id = seq_id.astype(np.int32)

        # Contig slice boundaries, and a prefix max of ends within each slice.
        self._max_end = np.empty_like(self._ends)
        self._bounds = {}
        if codes.size:
            edges = np.flatnonzero(codes[1:] != codes[:-1]) + 1
            lows = np.concatenate(([0], edges))
            highs = np.concatenate((edges, [codes.size]))
            names = chrom.categories
            for lo, hi in zip(lows, highs):
                self._bounds[str(names[codes[lo]])] = (int(lo), int(hi))
                np.maximum.accumulate(self._ends[lo:hi], out=self._max_end[lo:hi])

    # -- overlap search ----------------------------------------------------- #

    def _record(self, chrom: str, i: int) -> ReferenceRepeat:
        # Cast out of numpy scalars: these are formatted straight into the TSV,
        # and repr of a numpy scalar is not the repr of the Python value.
        annotations: dict[str, float] = {}
        for name, values in self._annots.items():
            value = values[i]
            annotations[name] = (round(float(value), 2) if name == "copy_num"
                                 else int(value))
        return ReferenceRepeat(
            chrom=chrom,
            start=int(self._starts[i]),
            end=int(self._ends[i]),
            motif=self._motifs[self._seq_id[i]],
            canonical=self._canon[self._seq_id[i]],
            annotations=annotations,
        )

    def _overlap_batch(self, chrom: str, q_start: np.ndarray,
                       q_end: np.ndarray) -> list[np.ndarray]:
        """Global row indices intersecting each half-open query interval.

        One ``searchsorted`` covers every query on the contig; the walk left from
        each landing point stops as soon as the running maximum end can no longer
        reach the query, which is what makes nested intervals safe.
        """
        empty = np.empty(0, dtype=np.int64)
        bounds = self._bounds.get(chrom)
        if bounds is None:
            return [empty] * len(q_start)
        lo, hi = bounds
        if hi <= lo:
            return [empty] * len(q_start)

        starts = self._starts[lo:hi]
        ends = self._ends[lo:hi]
        max_end = self._max_end[lo:hi]
        # Every candidate has start < q_end; walk left from there.
        right = np.searchsorted(starts, q_end, side="left") - 1

        out: list[np.ndarray] = []
        for k in range(len(q_start)):
            i = int(right[k])
            floor = int(q_start[k])
            hits: list[int] = []
            while i >= 0 and max_end[i] > floor:
                if ends[i] > floor:
                    hits.append(lo + i)
                i -= 1
            hits.reverse()
            out.append(np.asarray(hits, dtype=np.int64) if hits else empty)
        return out

    def overlapping(self, chrom: str, start: int, end: int) -> list[ReferenceRepeat]:
        """Repeats intersecting the 0-based half-open interval ``[start, end)``."""
        chrom = normalize_chrom(chrom)
        if end <= start:                 # zero-width query: probe a single base
            end = start + 1
        (indices,) = self._overlap_batch(chrom, np.array([start]), np.array([end]))
        return [self._record(chrom, int(i)) for i in indices]

    # -- screening ---------------------------------------------------------- #

    def _best_hit(self, indices: np.ndarray, chrom: str, point: int, motif: str,
                  canonical: str, tolerance: MotifTolerance) -> Hit | None:
        """The nearby repeat that best explains the query, or ``None``."""
        best: Hit | None = None
        best_key: tuple[int, int, int, float] | None = None
        for i in indices:
            repeat = self._record(chrom, int(i))
            distance = interval_distance(point, point + 1, repeat.start, repeat.end)
            found = tolerance.compare(motif, repeat.motif, self.equivalence,
                                      query_canonical=canonical,
                                      target_canonical=repeat.canonical)
            # Prefer any motif match over none, then the closest match, then
            # proximity, then the largest repeat. Edit counts are not comparable
            # across candidates of different lengths once a proportional budget
            # is in play, so whether it matched at all is the first key.
            key = (0 if found.matched else 1, found.edits, distance,
                   -repeat.annotations.get("copy_num", 0.0))
            if best_key is None or key < best_key:
                best_key = key
                best = Hit(repeat, distance, found.edits, found.kind)
        return best

    def _status(self, chrom: str, n_nearby: int, best: Hit | None) -> str:
        if chrom not in self._bounds:
            # No rows at all on this contig, so "nothing annotated nearby" is a
            # statement about the catalogue's coverage, not about the locus.
            # Reporting it as novel_locus would manufacture novelty for every
            # alt/random/decoy contig: TRExplorer v2 carries 25 contigs where
            # UCSC simpleRepeat carries 702.
            return UNSCREENED
        if not n_nearby:
            return "novel_locus"
        if best is not None and best.motif_matches:
            return "known"
        return "novel_motif"

    def covers(self, chrom: str) -> bool:
        """Whether this catalogue has any rows on the given contig."""
        return normalize_chrom(chrom) in self._bounds

    def screen(self, chrom: str, point: int, motif: str, *, window: int = 10,
               tolerance: MotifTolerance = DEFAULT_TOLERANCE) -> Verdict:
        """Classify a 0-based reference coordinate + motif as known or novel.

        ``window`` is how far off the coordinates may be, in bp: SV breakpoints
        are rarely placed exactly on the annotated repeat boundary, so a repeat
        counts as "here" if it comes within ``window`` bases of ``point``.
        """
        chrom = normalize_chrom(chrom)
        canonical = canonical_motif(motif, self.equivalence)
        (indices,) = self._overlap_batch(
            chrom, np.array([point - window]), np.array([point + window + 1]))
        best = self._best_hit(indices, chrom, point, motif, canonical, tolerance)
        return Verdict(
            chrom=chrom,
            point=point,
            motif=motif.strip().upper(),
            canonical=canonical,
            status=self._status(chrom, len(indices), best),
            n_nearby=len(indices),
            best=best,
        )

    def screen_frame(self, chroms, points, motifs, *, window: int = 10,
                     tolerance: MotifTolerance = DEFAULT_TOLERANCE,
                     prefix: str = "") -> pd.DataFrame:
        """Screen many loci at once; returns one row of results per query.

        Coordinates in and out are 0-based. Column names are
        :data:`RESULT_COLUMNS` plus this catalogue's annotations, each with
        ``prefix`` prepended so several platforms can sit side by side in one
        table.
        """
        chrom_values = normalize_chroms(chroms).to_numpy()
        point_values = pd.to_numeric(pd.Series(points), errors="raise").to_numpy(np.int64)
        motif_values = (pd.Series(motifs, dtype=object).fillna("")
                        .str.strip().str.upper().to_numpy())
        canonicals = canonical_motifs(motif_values, self.equivalence)

        n = len(point_values)
        novelty: list[str] = [""] * n
        n_nearby = np.zeros(n, dtype=np.int64)
        cells: dict[str, list] = {c: [None] * n for c in RESULT_COLUMNS[2:]}
        annotation_cells: dict[str, list] = {c: [None] * n for c in self.annotations}

        positions = pd.Series(np.arange(n))
        for chrom, group in positions.groupby(pd.Series(chrom_values), sort=False):
            rows = group.to_numpy()
            batch = self._overlap_batch(
                chrom, point_values[rows] - window, point_values[rows] + window + 1)
            for row, indices in zip(rows, batch):
                point = int(point_values[row])
                best = self._best_hit(indices, chrom, point, motif_values[row],
                                      canonicals[row], tolerance)
                n_nearby[row] = len(indices)
                novelty[row] = self._status(chrom, len(indices), best)
                if best is None:
                    continue
                cells["start"][row] = best.repeat.start
                cells["end"][row] = best.repeat.end
                cells["distance"][row] = best.distance
                cells["motif"][row] = best.repeat.motif
                cells["canonical"][row] = best.repeat.canonical
                cells["motif_edits"][row] = best.motif_edits
                cells["match"][row] = best.match
                for name, value in best.repeat.annotations.items():
                    annotation_cells[name][row] = value

        out = pd.DataFrame(index=pd.RangeIndex(n))
        out[f"{prefix}novelty"] = novelty
        out[f"{prefix}n_nearby"] = n_nearby
        for name in ("start", "end", "distance"):
            out[f"{prefix}{name}"] = pd.array(cells[name], dtype="Int64")
        out[f"{prefix}motif"] = cells["motif"]
        out[f"{prefix}canonical"] = cells["canonical"]
        out[f"{prefix}motif_edits"] = pd.array(cells["motif_edits"], dtype="Int64")
        out[f"{prefix}match"] = cells["match"]
        for name in self.annotations:
            dtype = "Int64" if name in _INT_ANNOTATIONS else "Float64"
            out[f"{prefix}{name}"] = pd.array(annotation_cells[name], dtype=dtype)
        return out

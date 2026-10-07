"""Reading reference tandem-repeat catalogues, and loading one or more of them.

Every catalogue is normalised to one schema (``chrom, start, end, motif`` plus
whatever of :data:`ANNOTATION_COLUMNS` it carries) -- this is what
:mod:`novelty.catalog` builds its interval index from.

Only plain BED4 (``chrom start end motif``) is read here; no format sniffing,
no UCSC ``simpleRepeat``/TRGT readers. A catalogue is given explicitly, as a
local path or a URL -- no platform registry, no named/bundled catalogues, no
auto-discovery.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from intruder.trcore.coords import normalize_chrom
from intruder.trcore.fetch import download_file
from intruder.trcore.motifs import DEFAULT_EQUIVALENCE, MotifEquivalence, canonical_motif
from intruder.trcore.paths import repo_root

# The normalised schema every reader produces.
CATALOG_COLUMNS = ("chrom", "start", "end", "motif")

# Optional per-repeat annotations, kept when a catalogue provides them:
#   period          length of the repeat unit (e.g. 6 -> 6 bp motif)
#   copy_num        mean number of copies of the unit in the reference
#   consensus_size  length of the consensus sequence (usually == period)
#   per_match       % identity between the perfect repeat and the genome
#   per_indel       % indel between the perfect repeat and the genome
ANNOTATION_COLUMNS = ("period", "copy_num", "consensus_size", "per_match", "per_indel")

DOWNLOAD_DIR = Path(repo_root(__file__) / "data" / "novelty")


# --------------------------------------------------------------------------- #
# normalization -- what catalog.py's RepeatCatalog.screen_frame calls
# --------------------------------------------------------------------------- #

def canonical_motifs(values,
                     equivalence: MotifEquivalence = DEFAULT_EQUIVALENCE) -> np.ndarray:
    """Vectorised :func:`canonical_motif` over an array-like of motif strings.

    Each distinct string is canonicalised exactly once and the result is
    broadcast back through the factor codes.
    """
    series = pd.Series(values, dtype=object).fillna("")
    codes, uniques = pd.factorize(series, sort=False)
    if len(uniques) == 0:
        return np.empty(len(series), dtype=object)
    table = np.empty(len(uniques) + 1, dtype=object)
    table[:-1] = [canonical_motif(str(u), equivalence) for u in uniques]
    table[-1] = ""                      # factorize marks missing values as -1
    return table[[c if c >= 0 else len(uniques) for c in codes]]


def normalize_chroms(values) -> pd.Series:
    """Vectorised :func:`normalize_chrom`; contig names repeat, so map the uniques."""
    series = pd.Series(values, dtype=object).fillna("")
    codes, uniques = pd.factorize(series, sort=False)
    lookup = [normalize_chrom(u) for u in uniques] + [""]
    return pd.Series(
        [lookup[c if c >= 0 else -1] for c in codes], index=series.index, dtype=object
    )


# --------------------------------------------------------------------------- #
# reading -- BED4 only
# --------------------------------------------------------------------------- #

def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _finalize(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Normalise contigs, coordinates and motifs; drop unusable rows."""
    out = pd.DataFrame(index=frame.index)
    out["chrom"] = normalize_chroms(frame["chrom"])
    out["start"] = pd.to_numeric(frame["start"], errors="coerce").astype("Int64")
    out["end"] = pd.to_numeric(frame["end"], errors="coerce").astype("Int64")
    out["motif"] = frame["motif"].astype(object).fillna("").str.strip().str.upper()
    for column in ANNOTATION_COLUMNS:
        if column in frame.columns:
            out[column] = pd.to_numeric(frame[column], errors="coerce")

    usable = out["start"].notna() & out["end"].notna() & (out["motif"].str.len() > 0)
    dropped = int((~usable).sum())
    if dropped:
        print(f"[novelty] {path}: skipped {dropped:,} row(s) with no motif or "
              f"coordinates", file=sys.stderr)
    out = out.loc[usable].reset_index(drop=True)
    out["start"] = out["start"].astype("int64")
    out["end"] = out["end"].astype("int64")
    return out


def read_bed(path: str | Path) -> pd.DataFrame:
    """Read a BED4 catalogue: ``chrom start end motif``, extra columns ignored."""
    path = Path(path)
    frame = pd.read_csv(
        path, sep="\t", compression="gzip" if _is_gzip(path) else None,
        header=None, comment="#", usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "motif"],
        dtype={0: "string", 3: "string"}, na_filter=False,
    )
    return _finalize(frame, path)


# UCSC's raw simpleRepeat.txt(.gz) table dump: no header, these columns in this
# order. `name` is NOT the repeat unit -- it's always the literal string "trf"
# (a label meaning "Tandem Repeats Finder produced this row"), a historical
# artefact of the track's origin. The actual consensus sequence is the last
# column, `sequence`.
_SIMPLEREPEAT_COLUMNS = ("bin", "chrom", "chromStart", "chromEnd", "name",
                         "period", "copyNum", "consensusSize", "perMatch",
                         "perIndel", "score", "A", "C", "G", "T", "entropy",
                         "sequence")


def read_simplerepeat(path: str | Path) -> pd.DataFrame:
    """Read UCSC's raw ``simpleRepeat.txt(.gz)`` table dump, as downloaded from
    e.g. ``hgdownload.soe.ucsc.edu/goldenPath/<assembly>/database/simpleRepeat.txt.gz``.

    Not BED: no header, starts with a ``bin`` indexing column UCSC uses
    internally, and the repeat unit is the ``sequence`` column, not ``name``
    (which is always the literal string ``"trf"``, not a motif). Converted here
    into the same normalised schema :func:`read_bed` produces, picking up
    ``period``/``copyNum``/``consensusSize``/``perMatch``/``perIndel`` as the
    catalogue's :data:`ANNOTATION_COLUMNS`.
    """
    path = Path(path)
    frame = pd.read_csv(
        path, sep="\t", compression="gzip" if _is_gzip(path) else None,
        header=None, comment="#", usecols=range(len(_SIMPLEREPEAT_COLUMNS)),
        names=_SIMPLEREPEAT_COLUMNS,
        dtype={"chrom": "string", "sequence": "string"}, na_filter=False,
    )
    renamed = pd.DataFrame({
        "chrom": frame["chrom"],
        "start": frame["chromStart"],
        "end": frame["chromEnd"],
        "motif": frame["sequence"],
        "period": frame["period"],
        "copy_num": frame["copyNum"],
        "consensus_size": frame["consensusSize"],
        "per_match": frame["perMatch"],
        "per_indel": frame["perIndel"],
    })
    return _finalize(renamed, path)


def read_catalog(path: str | Path, fmt: str = "bed") -> pd.DataFrame:
    """Read a catalogue file into the normalised schema.

    ``fmt`` is explicit, never sniffed: ``"bed"`` (plain BED4, the default) or
    ``"ucsc"`` (a raw UCSC ``simpleRepeat.txt(.gz)`` table dump).
    """
    if fmt == "bed":
        return read_bed(path)
    if fmt == "ucsc":
        return read_simplerepeat(path)
    raise ValueError(f"unsupported catalogue format {fmt!r}; use 'bed' or 'ucsc' "
                     f"(no format sniffing/registry)")


# --------------------------------------------------------------------------- #
# multi-source loading -- explicit path or URL, no registry, no auto-discovery
# --------------------------------------------------------------------------- #

def is_url(spec: str) -> bool:
    return urlparse(spec).scheme in ("http", "https")


def resolve_source(name: str, source: str) -> Path:
    """A local path as given, or a URL downloaded once to a local cache dir."""
    if not is_url(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"--repeats {name}={source}: file not found")
        return path
    filename = Path(urlparse(source).path).name or f"{name}.bed"
    target = DOWNLOAD_DIR / filename
    if target.exists():
        return target
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return download_file(source, target, label="novelty")


def parse_catalogs(specs: list[str]) -> dict[str, str]:
    """``NAME=PATH`` or ``NAME=URL`` specs, in the order given; names unique."""
    sources: dict[str, str] = {}
    for spec in specs:
        name, sep, source = spec.partition("=")
        if not sep or not name or not source:
            raise ValueError(f"--repeats {spec!r} must be NAME=PATH or NAME=URL")
        if name in sources:
            raise ValueError(f"--repeats name {name!r} given twice")
        sources[name] = source
    return sources


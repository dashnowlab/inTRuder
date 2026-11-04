"""Novelty assessment for tandem repeats found in SV insertions.

A tandem repeat called inside an SV insertion is *novel* when the reference
genome has nothing like it at that locus, against one or more explicitly-given
reference catalogues (BED4, UCSC simpleRepeat or STRchive files, local or URL).

The ``detect`` command runs the steps below, on a VCF of SVs:

    novelty.trfcaller   finding tandem repeats inside each insertion (pytrf)
    novelty.filter      dropping weak repeat calls -- a judgment on the SV/TRF
                        call, separate from whether the locus is known
    novelty.platforms   reading + normalising a catalogue; locating its source
    novelty.catalog     the interval index and the known/novel verdict
    novelty.motifs      motif comparison: equivalence, and tolerance
    novelty.coords      the coordinate conventions every step shares
    novelty.fetch       downloading a catalogue and caching it on disk
    novelty.paths       finding the repository checkout, for the cache location
    novelty.cli         the `python -m intruder.pipeline.novelty` command line,
                        which also combines the per-catalogue verdicts
"""

from .coords import interval_distance, normalize_chrom, to_external, to_internal
from .motifs import (
    DEFAULT_EQUIVALENCE,
    DEFAULT_TOLERANCE,
    MATCH_KINDS,
    STR_MAX_MOTIF,
    MotifEquivalence,
    MotifMatch,
    MotifTolerance,
    canonical_motif,
    edit_budget,
    least_rotation,
    motif_distance,
    primitive_unit,
    tiling_distance,
)

from .catalog import (
    STATUSES,
    UNSCREENED,
    Hit,
    ReferenceRepeat,
    RepeatCatalog,
    Verdict,
)

from .platforms import (
    ANNOTATION_COLUMNS,
    CATALOG_COLUMNS,
    canonical_motifs,
    is_url,
    #load_catalogs,
    normalize_chroms,
    parse_catalogs,
    read_catalog,
    resolve_source,
)

__all__ = [
    "ANNOTATION_COLUMNS",
    "CATALOG_COLUMNS",
    "DEFAULT_EQUIVALENCE",
    "DEFAULT_TOLERANCE",
    "MATCH_KINDS",
    "STATUSES",
    "STR_MAX_MOTIF",
    "UNSCREENED",
    "Hit",
    "MotifEquivalence",
    "MotifMatch",
    "MotifTolerance",
    "ReferenceRepeat",
    "RepeatCatalog",
    "Verdict",
    "canonical_motif",
    "canonical_motifs",
    "edit_budget",
    "interval_distance",
    "is_url",
    "least_rotation",
    #"load_catalogs",
    "motif_distance",
    "normalize_chrom",
    "normalize_chroms",
    "parse_catalogs",
    "primitive_unit",
    "read_catalog",
    "resolve_source",
    "tiling_distance",
    "to_external",
    "to_internal",
]
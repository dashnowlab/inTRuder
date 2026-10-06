"""Novelty assessment for tandem repeats found in SV insertions.

A tandem repeat called inside an SV insertion is *novel* when the reference
genome has nothing like it at that locus, against one or more explicitly-given
reference catalogues (BED4 files, local or URL).

    trcore.motifs       motif comparison: equivalence, and tolerance
    trcore.coords       the coordinate conventions both steps share
    novelty.platforms   reading + normalising a catalogue; loading catalogue(s)
    novelty.catalog     the interval index and the known/novel verdict
    novelty.verdicts    combining verdicts across more than one catalogue
    novelty.insertions  purity of the insertion itself, and row-level QC --
                        a judgment on the SV/TRF call, separate from whether
                        the locus is known; wired into `annotate`, not `query`
    novelty.cli         the `python -m intruder.pipeline.novelty` command line
"""

from intruder.trcore.coords import interval_distance, normalize_chrom, to_external, to_internal
from intruder.trcore.motifs import (
    DEFAULT_EQUIVALENCE,
    DEFAULT_TOLERANCE,
    MATCH_KINDS,
    MAX_FUZZY_MOTIF,
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
    "MAX_FUZZY_MOTIF",
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
"""Command line for screening tandem repeats against reference catalogue(s).

Just argument parsing and orchestration: turning flags into the domain
objects (`MotifEquivalence`, `MotifTolerance`), calling into
`platforms` to load catalogues and `catalog.RepeatCatalog` to screen, and
`verdicts` to combine results. No source resolution, no download, no
verdict-combining logic lives here -- see those modules for that.

    # a whole table, against two catalogues (a local file and a URL)
    python -m intruder.pipeline.novelty annotate \\
        --repeats ucsc=catalog.bed \\
        --repeats trexplorer=https://example.org/trexplorer.bed.gz \\
        input.tsv output.tsv

    # the same, against both local files
    python -m intruder.pipeline.novelty annotate \\
        --repeats ucsc=catalog.bed --repeats trexplorer=trexplorer.bed.gz \\
        input.tsv output.tsv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from intruder.trcore.coords import to_external, to_internal

from .catalog import STATUSES, RepeatCatalog
from .motifs import MotifEquivalence, MotifTolerance
from .platforms import parse_catalogs, resolve_source, read_catalog


# known < novel_motif < novel_locus < unscreened -- the most conservative
# verdict wins when catalogues disagree: a locus is only novel if none of
# them has it. unscreened ranks last since it's an absence of coverage, not
# an opinion -- any catalogue with an actual verdict outranks it.
PRECEDENCE = {status: rank for rank, status in enumerate(STATUSES)}
BY_RANK = dict(enumerate(STATUSES))


# --------------------------------------------------------------------------- #
# screening parameters -- Group A: change the known/novel verdict itself
# --------------------------------------------------------------------------- #

def _add_screen_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("screening")
    group.add_argument(
        "--window", type=int, default=10, metavar="BP",
        help="how far a reference repeat may sit from the query coordinate "
             "and still count as the same locus (default: %(default)s)")
    group.add_argument(
        "--max-motif-edits", type=int, default=0, metavar="N",
        help="accept a reference motif within N edits of the query motif "
             "(default: %(default)s, exact only)")
    group.add_argument(
        "--max-motif-edit-fraction", type=float, default=None, metavar="FRAC",
        help="also accept a reference motif within FRAC x its length, for "
             "long motifs (default: off)")
    group.add_argument(
        "--min-subrepeat-motif", type=int, default=None, metavar="BP",
        help="also accept a reference motif that the query motif tiles, e.g. "
             "ACC against ACCATC, when the tiling unit is at least BP long. "
             "Only has an effect together with --max-motif-edits or "
             "--max-motif-edit-fraction (default: off)")
    group.add_argument(
        "--circular", action=argparse.BooleanOptionalAction, default=True,
        help="CAG == AGC == GCA, i.e. rotation-equivalent (default: on)")


def _add_catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeats", required=True, action="append", metavar="NAME=PATH|URL",
        help="a reference catalogue, as NAME=PATH or NAME=URL (BED4: "
             "chrom start end motif). Repeat this flag for more than one "
             "catalogue; NAME becomes the output column prefix.")
    parser.add_argument(
        "--format", action="append", default=[], metavar="NAME=FMT",
        help="format override for one --repeats catalogue, as NAME=FMT: "
             "'bed' (default, plain BED4) or 'ucsc' (a raw UCSC "
             "simpleRepeat.txt(.gz) table dump). Repeat for more than one "
             "catalogue needing a non-default format.")


def _parse_formats(specs: list[str]) -> dict[str, str]:
    """``--format NAME=FMT``, in the order given; names unique."""
    formats: dict[str, str] = {}
    for spec in specs:
        name, sep, fmt = spec.partition("=")
        if not sep or not name or not fmt:
            raise ValueError(f"--format {spec!r} must be NAME=FMT")
        if name in formats:
            raise ValueError(f"--format name {name!r} given twice")
        formats[name] = fmt
    return formats


def _equivalence(args: argparse.Namespace) -> MotifEquivalence:
    return MotifEquivalence(circular=args.circular)


def _tolerance(args: argparse.Namespace) -> MotifTolerance:
    return MotifTolerance(max_edits=args.max_motif_edits or 0,
                          max_edit_fraction=args.max_motif_edit_fraction,
                          min_subrepeat_motif=args.min_subrepeat_motif)


# --------------------------------------------------------------------------- #
# file ops
# --------------------------------------------------------------------------- #


def load_catalogs(specs: list[str], *, equivalence: MotifEquivalence,
                  formats: dict[str, str] | None = None,
                  verbose: bool = True) -> dict[str, RepeatCatalog]:
    """Build one RepeatCatalog per ``--repeats`` spec, in the order given.

    ``formats`` maps a catalogue's name to an explicit format ('bed'/'ucsc');
    any name not in it defaults to 'bed'. Never sniffed from the file itself.
    """
    formats = formats or {}
    catalogs: dict[str, RepeatCatalog] = {}
    for name, source in parse_catalogs(specs).items():
        path = resolve_source(name, source)
        fmt = formats.get(name, "bed")
        frame = read_catalog(path, fmt=fmt)
        catalogs[name] = RepeatCatalog.from_frame(frame, equivalence=equivalence,
                                                   platform=name)
        if verbose:
            print(f"[novelty] {name}: {len(catalogs[name]):,} repeat(s) from "
                  f"{path} (fmt={fmt})", file=sys.stderr)
    return catalogs



# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _locus_counts(out: pd.DataFrame, chrom_col: str, pos_col: str) -> tuple[pd.Series, int]:
    """One verdict per (chrom, pos) locus, not per row.

    Multiple rows can share a locus -- different samples carrying the same
    insertion, or multiple TRF calls at one site -- and row-level counts
    overweight whichever loci happen to have more duplicate rows. This takes
    the most conservative verdict across a locus's rows (same precedence
    `combine_verdicts` uses across catalogues, just applied across rows here)
    so each genomic locus is counted exactly once.
    """
    ranks = out["novelty"].map(PRECEDENCE)
    per_locus_rank = ranks.groupby([out[chrom_col], out[pos_col]], sort=False).min()
    per_locus_status = per_locus_rank.map(BY_RANK)
    return per_locus_status.value_counts(), len(per_locus_status)


def _report(out: pd.DataFrame, args: argparse.Namespace, catalogs: dict) -> None:
    counts = out["novelty"].value_counts()
    total = len(out) or 1
    print(f"[novelty] {args.output}: {len(out):,} row(s), "
          f"catalogues: {', '.join(catalogs)}", file=sys.stderr)
    for status in STATUSES:
        n = int(counts.get(status, 0))
        print(f"[novelty]   {status:<12} {n:>8,}  ({100 * n / total:5.1f}%)",
              file=sys.stderr)

    locus_counts, n_loci = _locus_counts(out, args.chrom_col, args.pos_col)
    print(f"[novelty] per locus ({args.chrom_col}+{args.pos_col}) (n={n_loci:,})",
          file=sys.stderr)
    for status in STATUSES:
        n = int(locus_counts.get(status, 0))
        print(f"[novelty]   {status:<12} {n:>8,}  ({100 * n / (n_loci or 1):5.1f}%)",
              file=sys.stderr)


def combine_verdicts(statuses: pd.DataFrame) -> pd.Series:
    """One verdict per row, across catalogues' status columns."""
    ranks = statuses.apply(lambda column: column.map(PRECEDENCE))
    return ranks.min(axis=1).map(BY_RANK)   


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def _cmd_annotate(args: argparse.Namespace) -> int:
    catalogs = load_catalogs(args.repeats, equivalence=_equivalence(args),
                             formats=_parse_formats(args.format))
    tolerance = _tolerance(args)

    frame = pd.read_csv(args.input, sep="\t",
                        dtype={args.chrom_col: "string", args.motif_col: "string"})
    for column in (args.chrom_col, args.pos_col, args.motif_col):
        if column not in frame.columns:
            raise KeyError(f"column {column!r} not in {args.input} header: "
                           f"{list(frame.columns)}")


    points = to_internal(pd.to_numeric(frame[args.pos_col], errors="raise"),
                         args.coord_base)

    blocks: list[pd.DataFrame] = []
    statuses = pd.DataFrame(index=frame.index)
    for name, catalog in catalogs.items():
        block = catalog.screen_frame(
            frame[args.chrom_col], points, frame[args.motif_col],
            window=args.window, tolerance=tolerance, prefix=f"{name}_")
        block[f"{name}_start"], block[f"{name}_end"] = to_external(
            block[f"{name}_start"], block[f"{name}_end"], args.coord_base)
        statuses[name] = block[f"{name}_novelty"]
        blocks.append(block)

    novelty = combine_verdicts(statuses) if len(catalogs) > 1 else statuses.iloc[:, 0]
    out = pd.concat([frame, pd.DataFrame({"novelty": novelty}), *blocks], axis=1)
    out.to_csv(args.output, sep="\t", index=False, na_rep="NA")

    _report(out, args, catalogs)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novelty-min",
        description="screen tandem repeats against one or more explicitly-"
                    "given reference catalogues -- core algorithm only")
    parser.add_argument(
        "--coord-base", type=int, choices=(0, 1), default=1,
        help="coordinate base of the input table's position column "
             "(default: %(default)s, VCF-style)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="screen a whole table")
    annotate.add_argument("input")
    annotate.add_argument("output")
    annotate.add_argument("--chrom-col", default="chrom")
    annotate.add_argument("--pos-col", default="ins_coord",
                          help="column with the SV's reference-genome coordinate "
                               "(default: %(default)s, the sv_trfcaller.py "
                               "insertion site -- not rep_start/rep_end, which "
                               "are offsets within the inserted sequence, not "
                               "reference coordinates)")
    annotate.add_argument("--motif-col", default="motif")
    _add_catalog_args(annotate)
    _add_screen_args(annotate)
    annotate.set_defaults(func=_cmd_annotate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


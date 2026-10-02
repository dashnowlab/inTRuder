"""Command line for screening tandem repeats against reference catalogue(s).

Just argument parsing and orchestration: turning flags into the domain
objects (`MotifEquivalence`, `MotifTolerance`), calling into
`platforms` to load catalogues and `catalog.RepeatCatalog` to screen, and
`verdicts` to combine results. No source resolution, no download, no
verdict-combining logic lives here -- see those modules for that.

    # one locus, one catalogue
    python -m intruder.pipeline.novelty query \\
        --repeats ucsc=catalog.bed --chrom chr1 --pos 10772 --motif GC

    # one locus, two catalogues (a local file and a URL)
    python -m intruder.pipeline.novelty query \\
        --repeats ucsc=catalog.bed \\
        --repeats trexplorer=https://example.org/trexplorer.bed.gz \\
        --chrom chr1 --pos 10772 --motif GC

    # a whole table, against both
    python -m intruder.pipeline.novelty annotate \\
        --repeats ucsc=catalog.bed --repeats trexplorer=trexplorer.bed.gz \\
        input.tsv output.tsv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from intruder.trcore.coords import to_external, to_internal
from intruder.trcore.motifs import MAX_FUZZY_MOTIF, MotifEquivalence, MotifTolerance

from .catalog import STATUSES, RepeatCatalog
from .platforms import parse_repeats, resolve_source, read_catalog
from .verdicts import PRECEDENCE, combine_verdicts


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
        "--max-fuzzy-motif", type=int, default=MAX_FUZZY_MOTIF, metavar="BP",
        help="longest motif near-miss matching is attempted on "
             "(default: %(default)s)")
    group.add_argument(
        "--circular", action=argparse.BooleanOptionalAction, default=True,
        help="CAG == AGC == GCA, i.e. rotation-equivalent (default: on)")
    group.add_argument(
        "--reverse-complement", action="store_true",
        help="CAG == CTG, i.e. opposite-strand equivalent (default: off)")


def _equivalence(args: argparse.Namespace) -> MotifEquivalence:
    return MotifEquivalence(circular=args.circular,
                            reverse_complement=args.reverse_complement)


def _tolerance(args: argparse.Namespace) -> MotifTolerance:
    return MotifTolerance(max_edits=args.max_motif_edits or 0,
                          max_edit_fraction=args.max_motif_edit_fraction,
                          max_fuzzy_motif=args.max_fuzzy_motif or MAX_FUZZY_MOTIF)


def _add_catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeats", required=True, action="append", metavar="NAME=PATH|URL",
        help="a reference catalogue, as NAME=PATH or NAME=URL (BED4: "
             "chrom start end motif). Repeat this flag for more than one "
             "catalogue; NAME becomes the output column prefix.")


# --------------------------------------------------------------------------- #
# file ops
# --------------------------------------------------------------------------- #


def load_catalogs(specs: list[str], *, equivalence: MotifEquivalence,
                  verbose: bool = True) -> dict[str, RepeatCatalog]:
    """Build one RepeatCatalog per ``--repeats`` spec, in the order given."""
    catalogs: dict[str, RepeatCatalog] = {}
    for name, source in parse_repeats(specs).items():
        path = resolve_source(name, source)
        frame = read_catalog(path, fmt="bed")
        catalogs[name] = RepeatCatalog.from_frame(frame, equivalence=equivalence,
                                                   platform=name)
        if verbose:
            print(f"[novelty] {name}: {len(catalogs[name]):,} repeat(s) from {path}",
                  file=sys.stderr)
    return catalogs



# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def _cmd_query(args: argparse.Namespace) -> int:
    catalogs = load_catalogs(args.repeats, equivalence=_equivalence(args))
    tolerance = _tolerance(args)
    point = to_internal(args.pos, args.coord_base)

    print(f"{args.chrom}:{args.pos} ({args.coord_base}-based)  "
          f"motif={args.motif.strip().upper()}")

    verdicts = {}
    for name, catalog in catalogs.items():
        verdict = catalog.screen(args.chrom, point, args.motif, window=args.window,
                                 tolerance=tolerance)
        verdicts[name] = verdict
        print(f"  {name}")
        print(f"    status : {verdict.status}")
        print(f"    nearby (+/-{args.window}bp): {verdict.n_nearby} "
              f"reference repeat(s)")
        if verdict.best is not None:
            repeat = verdict.best.repeat
            start, end = to_external(repeat.start, repeat.end, args.coord_base)
            label = verdict.best.match if verdict.best.motif_matches else "nearest"
            print(f"    best {label:<9}: {repeat.chrom}:{start}-{end}  "
                  f"motif={repeat.motif} (canonical={repeat.canonical}, "
                  f"{verdict.best.motif_edits} edit(s), "
                  f"{verdict.best.distance}bp away)")

    if len(verdicts) > 1:
        combined = min(verdicts.values(), key=lambda v: PRECEDENCE[v.status])
        print(f"  combined : {combined.status}")
    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    catalogs = load_catalogs(args.repeats, equivalence=_equivalence(args))
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

    counts = out["novelty"].value_counts()
    total = len(out) or 1
    print(f"[novelty] {args.output}: {len(out):,} row(s), "
          f"catalogues: {', '.join(catalogs)}", file=sys.stderr)
    for status in STATUSES:
        n = int(counts.get(status, 0))
        print(f"[novelty]   {status:<12} {n:>8,}  ({100 * n / total:5.1f}%)",
              file=sys.stderr)

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
        help="coordinate base of --pos / the input table's position column "
             "(default: %(default)s, VCF-style)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="screen one locus")
    query.add_argument("--chrom", required=True)
    query.add_argument("--pos", type=int, required=True)
    query.add_argument("--motif", required=True)
    _add_catalog_args(query)
    _add_screen_args(query)
    query.set_defaults(func=_cmd_query)

    annotate = subparsers.add_parser("annotate", help="screen a whole table")
    annotate.add_argument("input")
    annotate.add_argument("output")
    annotate.add_argument("--chrom-col", default="chrom")
    annotate.add_argument("--pos-col", default="ins_coord")
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


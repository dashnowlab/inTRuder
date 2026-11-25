from pathlib import Path
from datetime import date
import argparse
import gzip
import tempfile
import itertools

from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds

from intruder.coords import to_external, to_internal
from intruder.catalog import RepeatCatalog
from intruder.platforms import _finalize, _is_gzip


def load_novelty_results(path: Path | str, cols: list) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=cols)


def prep_validation(args, samples: list, strip_hp = True) -> None:
    """One row per (repeat, sample): chrom, start, end, motif, sample, stat.

    Read in chunks and keep only ``samples``: splitting the sample column
    multiplies the rows by the number of samples carrying each repeat.
    Each chunk is appended to ``args.val_out`` instead of held in memory.
    """
    columns = [
        "#chrom",
        "start",
        "end",
        "motif",
        "copy_numbers",
        # "sizes",  # unused in output, removed to save memory
        # "motifs",  # unused in output, removed to save memory
        # "max_change",
        # "num_samples",
        # "num_calls",
        # "motif_frequency",
        # "feature",
    ]
    smps = tuple(samples) # turn samples into tuple to work with str.startswith

    open_func = gzip.open if _is_gzip(args.validation) else open
    with open_func(args.validation, "rt", encoding="utf-8") as file:
        sample_list = file.readline().replace("#", "").strip().split(";")

    keep_positions = [
        position for position, name in enumerate(sample_list)
            if name.startswith(smps)
    ]

    if not keep_positions:
        raise ValueError("[prep] No header samples match the requested sample prefixes")
    #kept_sample_names = [sample_list[position] for position in keep_positions]

    # fixed categories so every chunk has the same dtype
    kept_names = pd.Series([sample_list[position] for position in keep_positions])
    if strip_hp:
        sample_dtype = pd.CategoricalDtype(kept_names.str.split(".").str[0].unique())
        hp_dtype = pd.CategoricalDtype(kept_names.str.split(".").str[1].dropna().unique())
    else:
        sample_dtype = pd.CategoricalDtype(kept_names.unique())

    batch_count = 0 # chunks written so far
    for df in pd.read_csv(args.validation, sep="\t", usecols=columns, chunksize=100000, skiprows=1):
        # filter down to only insertions
        df = df[df["start"] == df["end"] - 1 ] if not args.all_loci else df

        if df.empty: # if the chunk contains no insertions, then skip
            continue
       
        df = df.rename(columns={"#chrom": "chrom"})

        # format sample-data rows
        df["copy_numbers"] = df["copy_numbers"].str.split(";")
        #df["motifs"] = df["motifs"].str.split(";")

        # per row, header positions that match a requested sample and have a copy number
        valid_positions = [
            [position for position in keep_positions if copy_numbers[position] != "-"]
            for copy_numbers in df["copy_numbers"]
        ]

        # slice copy numbers and sample names to those positions, before exploding
        df["copy_numbers"] = [
            [copy_numbers[position] for position in positions]
            for copy_numbers, positions in zip(df["copy_numbers"], valid_positions)
        ]
        df["sample"] = [
            [sample_list[position] for position in positions]
            for positions in valid_positions
        ]

        # drop repeats with no valid samples (explode would turn [] into a NaN row)
        df = df[df["copy_numbers"].str.len() > 0]
        if df.empty:
            continue

        df = df[["chrom", "start", "end", "motif", "copy_numbers", "sample"]].explode(["sample", "copy_numbers"])
        df = df.reset_index(drop=True)

        # remove haplotype sample tags so they match with novelty sample strings
        if strip_hp:
            df["val_hp"] = df["sample"].str.split(".").str[1].astype(hp_dtype)
            df["sample"] = df["sample"].str.split(".").str[0]

        # compact dtypes before the chunk is written
        df["sample"] = df["sample"].astype(sample_dtype)
        df["copy_numbers"] = pd.to_numeric(df["copy_numbers"], downcast="float")

        # first chunk creates the file with a header, later chunks append
        df.to_csv(args.val_out, sep="\t", index=False,
                  mode="w" if batch_count == 0 else "a", header=(batch_count == 0))
        batch_count += 1
        print(f"\r[prep] Batches complete: {batch_count}", end="", flush=True)

    if batch_count == 0:
        raise ValueError("[prep] No validation rows kept, nothing was written")


def run_prep(args) -> int:
    # only samples present in the novelty results are kept
    frame = load_novelty_results(args.input, ["sample"])
    print(f"[prep] Prepping validation set...")
    prep_validation(args, set(frame["sample"]))
    print(f"\n[prep] Wrote prepped validation set to {args.val_out}")
    return 0


def filter_batches(reader, sample_values: pa.Array):
    """Yield each batch reduced to rows whose sample is requested."""
    for batch in reader:
        yield batch.filter(pc.is_in(batch["sample"], value_set=sample_values))


def split_validation_by_sample(path: Path | str, samples: set, out_dir: Path) -> int:
    """Pass over prepped file, writing a Parquet partition per sample.

    Returns the number of sample partitions written.
    """
    locus_columns = ["chrom", "start", "end", "motif"]
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(use_threads=True, block_size=32 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            include_columns=locus_columns + ["sample"],
            column_types={"chrom": pa.string(), "start": pa.int64(),
                          "end": pa.int64(), "motif": pa.string(),
                          "sample": pa.string()},
        ),
    )

    files_written = itertools.count(1)
    ds.write_dataset(
            filter_batches(reader, pa.array(sorted(samples))),
            out_dir, schema=reader.schema, format="parquet",
            partitioning=["sample"], partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
            file_visitor=lambda written: print("files written:", next(files_written), flush=True),
    )
    return len(list(out_dir.glob("sample=*")))


def run_validate(args) -> int:
    # read in novelty results
    frame_cols = ["chrom", "ins_coord", "sample", "motif", "insert_size", "purity","novelty"]
    frame = load_novelty_results(args.input, frame_cols)
    #frame[args.sample_col] = frame[args.sample_col].astype(str)

    # format 
    points = to_internal(pd.to_numeric(frame["ins_coord"], errors="raise"), 1)

    blocks: list[pd.DataFrame] = []

    print("Reading validation file...")
    # each pass is (validation repeats to screen against, novelty rows to screen)
    if args.within_sample: 
        # one pass over the big file, then each sample reads only its own small file
        tmp_dir = tempfile.TemporaryDirectory()

        written_count = split_validation_by_sample(
            args.validation, set(frame["sample"]), Path(tmp_dir.name))

        # print("written:", written_count, flush=True)
        # print("on disk:", sorted(p.name for p in Path(tmp_dir.name).iterdir())[:5], flush=True)
        # print("novelty:", sorted(map(repr, set(frame["sample"])))[:5], flush=True)

        # samples that have a validation file on disk
        matched_samples = {
            sample for sample in frame["sample"].unique()
            if (Path(tmp_dir.name) / f"sample={sample}").exists()
        }
        matched_mask = frame["sample"].isin(matched_samples)

        # every call without a matching sample is unscreened, filled in one shot
        blocks.append(pd.DataFrame({"val_novelty": "unscreened"}, index=frame.index[~matched_mask]))

        # only matched samples are looped over
        passes = (
            (pd.read_parquet(Path(tmp_dir.name) / f"sample={sample}").drop_duplicates(), rows)
            for sample, rows in frame[matched_mask].groupby("sample", sort=False)
        )
    else: 
        # single pass: all calls against every validation repeat, regardless of sample 
        # unique loci only, deduplicated per chunk to bound memory
        locus_columns = ["chrom", "start", "end", "motif"]
        unique_loci = pd.concat(
            chunk.drop_duplicates() for chunk in
            pd.read_csv(args.validation, sep="\t", usecols=locus_columns, chunksize=1000000)
        ).drop_duplicates()
        passes = [(unique_loci, frame)]

    # for each sample in the novelty results
    for val_rows, rows in tqdm(passes, total=len(matched_samples), unit="samples", desc="[novelty]"): 

        # screen this sample's calls
        ref = _finalize(val_rows[["chrom", "start", "end", "motif"]], args.validation) 
        catalog = RepeatCatalog.from_frame(ref, platform="val")
        block = catalog.screen_frame(
            rows["chrom"], points.loc[rows.index], rows["motif"],
            window=args.window, prefix="val_"
            )

        # # reset index to match input block (screen changes them)
        block.index = rows.index

        # a chrom(contig) with none of this sample's repeats is novel, not unscreened
        block["val_novelty"] = block["val_novelty"].replace("unscreened", "val_novel")
        block["val_novelty"] = block["val_novelty"].replace("novel_locus", "val_novel")
        blocks.append(block)

    # ensure coords line up with input format
    val_block = pd.concat(blocks)
    if "val_start" in val_block:
        val_block["val_start"], val_block["val_end"] = to_external(
            val_block["val_start"], val_block["val_end"], 1)

    # combine validation info with original columns, then output to tsv
    out = frame.join(val_block)
    out.to_csv(args.output, sep="\t", index=False, na_rep="NA")
    # [out["novelty"] == "novel_locus"]
    # run's verdict against the validation verdict, one row per distinct call
    calls = out.drop_duplicates(["chrom", "ins_coord", "sample", "motif"])

    screened = calls[calls["val_novelty"] != "unscreened"]
    print(f"[validate] output {len(out)} rows to {args.output}")
    print(f"[validate] {len(screened)} of {len(calls)}"
            f"unique calls screened against their own sample")
    print(f"[validate] {(((screened['val_novelty'] == 'known') | (screened['val_novelty'] == 'novel_motif')) & (screened['novelty'] == 'novel_locus')).sum()} of {len(screened[screened['novelty'] == 'novel_locus'])} "
          f"unique screened novel calls supported in their own sample")
    print(f"[validate] {((screened['val_novelty'] == 'known') | (screened['val_novelty'] == 'novel_motif')).sum()} of {len(screened)} "
        f"unique screened calls supported in their own sample")

    return 0


def main() -> int:
    dt = date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # shared args inherited by both subcommands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--input", default="/projects/garner1@xsede.org/dashnow_lab/inTRuder/data/hprc_multisample_novelty_20261027.tsv")
    common_parser.add_argument("--validation", default="/projects/garner1@xsede.org/dashnow_lab/inTRuder/data/hg38_frmd.tsv.gz")

    # prep: stream the validation catalog to a per-(repeat, sample) tsv
    prep_parser = subparsers.add_parser("prep", help="prep the validation catalog", parents=[common_parser])
    prep_parser.add_argument("--val-out", default="/projects/garner1@xsede.org/dashnow_lab/inTRuder/data/novelty_val_within-sample.tsv")
    prep_parser.add_argument("--all-loci", action='store_true', default=False)
    prep_parser.set_defaults(func=run_prep)

    # validate: screen novelty calls against the prepped validation tsv
    validate_parser = subparsers.add_parser("validate", help="screen novelty calls", parents=[common_parser])
    validate_parser.add_argument("--output", default=f"/projects/garner1@xsede.org/dashnow_lab/inTRuder/data/validation_results_{dt}.tsv")
    validate_parser.add_argument("--window", type=int, default=10)
    validate_parser.add_argument("--within-sample", action=argparse.BooleanOptionalAction, default=True)
    validate_parser.set_defaults(func=run_validate)
    args = parser.parse_args()

    return args.func(args)

if __name__ == "__main__":
    main()
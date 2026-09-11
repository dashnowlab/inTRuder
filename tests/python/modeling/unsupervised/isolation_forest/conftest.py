"""Shared fixtures: a synthetic candidate table, and a matching tiny VCF."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

REQUIRED_COLUMNS = (
    "chrom", "ins_coord", "SVID", "sample", "depth", "insert_size", "motif",
    "purity", "motif_length", "rep_start", "rep_end", "rep_length", "rep_units",
)


@pytest.fixture
def make_candidates():
    """Build a synthetic candidate table with every column `features.py` needs.

    ``n`` rows, motif lengths split between short (2-4bp) and long (8-12bp) so
    the default ``strata_bounds=(6,)`` produces two non-trivial strata; every
    row's repeat fills its whole insertion by default (``rep_start=0``), so
    tests that want a flanked row pass ``flanked=True`` for a subset.
    """

    def _make(n: int = 400, *, seed: int = 0, flanked: bool = False,
             svid_prefix: str = "SV") -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        motifs_short = ["AC", "AAT", "AGAT"]
        motifs_long = ["ACGTACGA", "AACCGGTTAACC", "ACACGTGACCTG"]

        rows = []
        for i in range(n):
            short = i % 2 == 0
            motif = rng.choice(motifs_short if short else motifs_long)
            motif_length = len(motif)
            rep_units = rng.randint(3, 20)
            rep_length = motif_length * rep_units
            flank_len = rng.randint(5, 20) if flanked else 0
            insert_size = rep_length + 2 * flank_len
            rep_start = flank_len
            rep_end = flank_len + rep_length
            rows.append({
                "chrom": "chr1",
                "ins_coord": 1000 + i,
                "SVID": f"{svid_prefix}.{i}",
                "sample": f"sample{i % 5}",
                "depth": f"{rng.randint(10, 20)},{rng.randint(10, 20)}",
                "insert_size": insert_size,
                "motif": motif,
                "purity": round(float(rng.uniform(0.8, 1.0)), 3),
                "motif_length": motif_length,
                "rep_start": rep_start,
                "rep_end": rep_end,
                "rep_length": rep_length,
                "rep_units": float(rep_units),
                "repeat_coverage": rep_length / insert_size,
            })
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def write_vcf(tmp_path):
    """Write a minimal SVTYPE=INS VCF whose ALT reproduces each candidate's insertion.

    Mirrors ``sv_trfcaller.py``'s convention: ALT = REF + inserted sequence, so
    ``flanks.py`` can trim REF back off and recover the same sequence the
    candidate table's ``rep_start``/``rep_end`` index into.
    """

    def _write(frame: pd.DataFrame, sequences: dict[str, str], name="candidates.vcf"):
        path = tmp_path / name
        header = (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=chr1,length=248956422>\n"
            "##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"Type of SV\">\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        )
        lines = []
        for svid, seq in sequences.items():
            row = frame.loc[frame["SVID"] == svid].iloc[0]
            lines.append(f"chr1\t{row['ins_coord']}\t{svid}\tA\tA{seq}\t60\tPASS\tSVTYPE=INS\n")
        path.write_text(header + "".join(lines), encoding="utf-8")
        return path

    return _write

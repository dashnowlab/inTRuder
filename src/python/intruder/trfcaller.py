#! /usr/bin/env python3

"""Calling tandem repeats inside the sequence of SV insertions.

Runs pytrf's approximate tandem repeat finder (``ATRFinder``) over each inserted
sequence and returns one row per repeat call. That table is what ``filter``
quality-checks and what the novelty screen in ``catalog`` then compares against
the reference catalogues.

What it reads
-------------
Only insertions: records with ``SVTYPE=INS``, an ``SVLEN`` INFO field and an ALT
allele. Only the first ALT allele is used. If it begins with REF (the usual
anchored-base form) that REF prefix is stripped, so only the inserted bases are
searched. Records whose SVLEN falls outside ``min_sv_length``/``max_sv_length``
are skipped. 

Every sample is searched separately and, within a sample, every non-reference
allele among the first two genotype entries. A homozygous-alt sample therefore
runs the finder twice on the same sequence and contributes two identical rows.

What it returns
---------------
A DataFrame with one row per repeat call, and nothing written to disk:

    chrom, ins_coord    where the insertion sits in the reference: the VCF CHROM
                        and POS (POS is 1-based)
    SVID                the VCF record ID
    depth               ``"<DV>,<DR>"``: variant-supporting then
                        reference-supporting read count for that sample
    insert_size         SVLEN
    sample, allele      the sample name and the genotype allele index
    rep_start, rep_end  the repeat's position inside the inserted sequence
                        (pytrf start - 1 and end - 1). These are offsets within
                        the insertion, not reference coordinates, so they cannot
                        be used to place the repeat in the genome
    motif               the repeat unit pytrf reported
    purity              pytrf's percent identity / 100, rounded to 3 places
    motif_length        len(motif)
    rep_length          rep_end - rep_start
    rep_units           rep_length // motif_length

When no insertion yields a repeat the frame has no columns at all, so a caller
should check for an empty result before indexing it.

Parameters
----------
``run_trf_on_insertions`` takes the parsed ``args`` namespace and reads ``input``
(the VCF), ``min_sv_length``, ``max_sv_length``, ``min_motif``, ``max_motif``,
``min_identity``, ``min_rep_units``, ``min_rep_length`` and ``max_rep_length``.
They reach pytrf as ``min_motif``, ``max_motif``, ``min_identity``,
``min_seedrep``, ``min_seedlen`` and ``max_extend``. ``min_identity`` is a
fraction (0.8) on the command line and is multiplied by 100 here, because pytrf
takes a percentage.

``suppress_pytrf_errors`` silences stdout and stderr at the file-descriptor
level (``os.dup2`` onto ``/dev/null``), which also catches output written by
compiled code, not just Python ``print`` calls.
"""
from __future__ import annotations

import argparse
import os
import io
import sys
from contextlib import contextmanager

import pytrf
from cyvcf2 import VCF
from tqdm import tqdm
import pandas as pd



@contextmanager
def suppress_pytrf_errors():
    """
    Suppress output from pytrf (stdout and stderr) within the context.
    This is useful when calling functions that may produce unwanted output to the console.
    Usage:
        with suppress_pytrf_errors():
            # code that calls pytrf functions
    """
    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Duplicate original file descriptors so they can be restored later
    save_stdout_fd = os.dup(original_stdout_fd)
    save_stderr_fd = os.dup(original_stderr_fd)

    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # Flush standard Python buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Overwrite original descriptors with devnull
    os.dup2(devnull_fd, original_stdout_fd)
    os.dup2(devnull_fd, original_stderr_fd)

    # Close the extra devnull handle
    os.close(devnull_fd)

    try:
        yield
    finally:
        # Flush again before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original descriptors
        os.dup2(save_stdout_fd, original_stdout_fd)
        os.dup2(save_stderr_fd, original_stderr_fd)

        # Close duplicated handles
        os.close(save_stdout_fd)
        os.close(save_stderr_fd)


def run_trf_on_insertions(args):
    """
    Run tandem repeat calling on the insertion variants of an input VCF and return the calls.

    Args:
        args: Parsed command-line arguments. Attributes read: `input` (the VCF),
            `min_sv_length`, `max_sv_length`, `min_motif`, `max_motif`,
            `min_identity` (a 0-1 fraction), `min_rep_units`, `min_rep_length` and
            `max_rep_length`. The VCF is expected to contain structural variant
            records with `SVTYPE=INS`, an `SVLEN` INFO field, and `DR`/`DV` sample
            FORMAT fields used to report read support.

    Returns:
        A DataFrame with one row per detected repeat call: chromosome, insertion
        coordinate, variant ID, depth, insertion size, sample, allele, repeat
        boundaries, motif, purity, motif length, repeat length, and repeat unit
        count. The frame has no columns when no repeat
        was found.
    """

    # cols = ['chrom', 'ins_coord', 'SVID', 'depth', 
    #         'insert_size', 'sample', 'allele', 'rep_start', 
    #         'rep_end', 'motif', 'purity','motif_length', 
    #         'rep_length', 'rep_units']
     
    rows = []
    vcf = VCF(args.input)

    samples = vcf.samples
    for variant in tqdm(vcf, ncols=80, smoothing=0.1, unit=' variants', desc="[trf] "):
        if variant.is_sv and variant.INFO.get("SVTYPE") == "INS":
            ID    = variant.ID
            REF   = variant.REF
            if variant.ALT is None or len(variant.ALT) == 0:
                continue
            ALT   = variant.ALT[0]
            if REF == ALT[:len(variant.REF)]:  # Get the inserted sequence
                ALT = ALT[len(variant.REF):]
            DR    = variant.format('DR')
            DV    = variant.format('DV')    
            if variant.INFO.get("SVLEN") is None:
                continue
            LEN   = variant.INFO.get("SVLEN")  # length of the insertion
            if LEN < args.min_sv_length or LEN > args.max_sv_length:
                continue

            for s, sample in enumerate(samples):
                genotype = variant.genotypes[s]
                for gt in genotype[:2]:
                    if gt == 0 or gt == '.' or gt == -1: continue  # Skip reference alleles
                    var_depth = DV[s][0] # depth of reads supporting variant allele
                    ref_depth = DR[s][0] # depth of reads supporting referebce allele
                    #ref_depth = DR[s][0] 
                    
                    n_trf = 0
                    trf_repeats    = []

                    with suppress_pytrf_errors():
                        for repeat in pytrf.ATRFinder(sample, ALT, min_motif=args.min_motif,
                                                                max_motif=args.max_motif,
                                                                min_identity=args.min_identity * 100,
                                                                min_seedrep=args.min_rep_units,
                                                                min_seedlen=args.min_rep_length,
                                                                max_extend=args.max_rep_length):

                            rep_start = repeat.start - 1
                            rep_end   = repeat.end - 1
                            motif     = repeat.motif
                            motif_length = len(motif)

                            rep_length = rep_end - rep_start
                            rep_units  = rep_length // motif_length
                            purity       = float(repeat.identity)/100

                            trf_repeats.append({
                                        'chrom': variant.CHROM,
                                        'ins_coord': variant.POS,
                                        'SVID': ID,
                                        'depth': f'{var_depth},{ref_depth}',
                                        'insert_size': LEN,
                                        'sample': sample,
                                        'allele': gt,
                                        'rep_start': rep_start,
                                        'rep_end': rep_end,
                                        'motif': motif,
                                        'purity': round(purity, 3),
                                        'motif_length': motif_length,
                                        'rep_length': rep_length,
                                        'rep_units': rep_units
                                    })
                            
                            n_trf += 1

                    for rep in trf_repeats:
                        rows.append(rep)
                        #print(*rep, sep='\t', file=out)


    # create & return pandas df of trs
    return pd.DataFrame(rows)
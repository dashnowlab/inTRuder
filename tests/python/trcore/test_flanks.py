"""Flank geometry and identity scoring: pure sequence arithmetic, no I/O."""

from __future__ import annotations

import pytest

from intruder.trcore.flanks import Flank, Flanks, flank_identity, flanks_of


def test_flank_identity_is_nan_for_an_empty_flank():
    assert flank_identity("AC", "") != flank_identity("AC", "")  # NaN != NaN


def test_flank_identity_is_zero_when_too_short_for_two_motif_copies():
    # "AC" needs >= 4bp of flank to tile twice; a 3bp flank is ambiguous and
    # reads as the worst case rather than a missing value.
    assert flank_identity("AC", "ACA") == 0.0


def test_flank_identity_is_perfect_for_a_clean_tiling():
    assert flank_identity("AC", "ACACAC") == pytest.approx(1.0)


def test_flank_identity_degrades_with_mismatches():
    perfect = flank_identity("AC", "ACACAC")
    degraded = flank_identity("AC", "ACACAT")
    assert degraded < perfect


def test_flanks_of_splits_around_the_repeat_call():
    # "XX" + "ACAC" (the repeat) + "YYY"
    sequence = "XXACACYYY"
    flanks = flanks_of(sequence, rep_start=2, rep_end=6, motif="AC")
    assert flanks.five_prime.sequence == "XX"
    assert flanks.three_prime.sequence == "YYY"


def test_flanks_of_has_flank_is_false_when_the_repeat_fills_the_sequence():
    flanks = flanks_of("ACACACAC", rep_start=0, rep_end=8, motif="AC")
    assert flanks.five_prime.sequence == ""
    assert flanks.three_prime.sequence == ""
    assert flanks.has_flank is False


def test_flanks_of_has_flank_is_true_with_either_side_present():
    flanks = flanks_of("XXACACAC", rep_start=2, rep_end=8, motif="AC")
    assert flanks.has_flank is True


def test_flank_length_is_the_sequence_length():
    flank = Flank("XXX", identity=0.0)
    assert flank.length == 3


def test_flanks_dataclasses_are_frozen():
    flank = Flank("A", 1.0)
    with pytest.raises(AttributeError):
        flank.sequence = "C"
    flanks = Flanks(flank, flank)
    with pytest.raises(AttributeError):
        flanks.five_prime = flank

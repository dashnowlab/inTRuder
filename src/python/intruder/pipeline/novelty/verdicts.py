"""Combining known/novel verdicts across more than one reference catalogue.

Kept separate from `catalog.py` itself so the core RepeatCatalog stays
single-catalogue-only; this is what turns several catalogues' independent
answers into one.
"""

from __future__ import annotations

import pandas as pd

from .catalog import STATUSES

# known < novel_motif < novel_locus < unscreened -- the most conservative
# verdict wins when catalogues disagree: a locus is only novel if none of
# them has it. unscreened ranks last since it's an absence of coverage, not
# an opinion -- any catalogue with an actual verdict outranks it.
PRECEDENCE = {status: rank for rank, status in enumerate(STATUSES)}
BY_RANK = dict(enumerate(STATUSES))


def combine_verdicts(statuses: pd.DataFrame) -> pd.Series:
    """One verdict per row, across catalogues' status columns."""
    ranks = statuses.apply(lambda column: column.map(PRECEDENCE))
    return ranks.min(axis=1).map(BY_RANK)

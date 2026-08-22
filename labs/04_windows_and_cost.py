"""Lab 4 — The series, and what it costs.

Why this lab exists: a single reading of speed says almost nothing, and the two
cheapest ways to flatter a model are a window that reaches past its own edge and
a split that puts a reading's own neighbour in the test set. You prove here that
you can summarise a window honestly, split a table so that no test row precedes
a training row, and price a join instead of guessing at a tolerance.
Where it sits: Block four — "What the join costs", and the definition slides
"Definition — the six window features", "Definition — lag-k sample
autocorrelation", "Definition — a split by time, never at random" and
"Definition — a nearest match within a tolerance".
What the check grades: the six features against its own arithmetic over exactly
the last `width` values, on the shipped speeds and on a planted window whose
answers are arithmetic; a split of a shuffled table whose last training time is
at or before the first test time, at the asked fraction and losing nothing;
matched shares that never fall as the tolerance widens and that agree with a
reference join within 0.2 percentage points against a vehicle thinned to one
reading every thirty seconds; and the hand-off table itself — one row per phone
per window, a mask beside every filled value and beside no other, the split
point recorded as an instant with no window straddling it, and the fitted
transform and the ledger stored unchanged.
Needs: pandas, numpy, plotly, and the loaders in lab_support.

Twenty-five minutes.

A single reading of speed says almost nothing. A window of readings says how
fast, how variable, which way it is heading, and whether anything unusual
happened -- and those are the features a model can actually use.

Then two costs, both measured rather than assumed: what a time-ordered split
costs you in training rows, and what a join costs you as the tolerance widens.

What you write: window_features(series, width), split_by_time(frame, fraction),
join_growth(phones, bus, tolerances), and
assemble(aligned, fitted, ledger, target, train_fraction).

    window_features(series, width) -> dict
        Over the last `width` values of `series`, return mean, std, minimum,
        maximum, slope and autocorrelation_1. Use only the last `width` values.
        A window that quietly reaches further back is a small leak of exactly
        the kind Lab 3 was about.

    split_by_time(frame, train_fraction) -> (train, test)
        Sort by the coordinated universal time column, then cut. Everything
        before the cut trains, everything after tests. No shuffling, no overlap,
        and no row in both. Module 1 measured why: consecutive readings correlate
        at about 0.997, so a random split tests the model on paraphrases of its
        own training data.

    join_growth(phones, bus, tolerances) -> dict
        For each tolerance in seconds, match each phone row to the nearest
        vehicle reading within that tolerance, and return
        {tolerance: matched_share_percent}. Widening the window buys matches and
        spends accuracy -- the vehicle reading you matched is further away in
        time. Measure the trade rather than picking a number and hoping.

        On the archive this runs 95.3 per cent at one second to 97.4 at thirty:
        2.1 percentage points for thirty times the tolerance. Report what you
        find here and compare.

    assemble(aligned, fitted, ledger, target, train_fraction) -> (table, manifest)
        The last twenty minutes of the module, and the only part of it another
        person will ever run. Everything so far produced a number; this produces
        the object Modules 3, 4 and 5 open. It is one table plus one manifest
        describing it, and the manifest is the half people forget: a table
        without its split point, its stored transform and its ledger is a table
        nobody can reproduce a result from.
"""
from __future__ import annotations

import sys
import pathlib

# Every library the reference solution uses is imported here, so that the work
# in front of you is the statistics and not the import lines.
import json                                                        # noqa: F401

import numpy as np
import pandas as pd
import plotly.graph_objects as go                                    # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_bus, load_phones  # noqa: E402
from _narrate import narrator, show_table, save_figure     # noqa: E402,F401

LAB = 4
TOLERANCES = (1, 2, 5, 10, 30)
DDOF = 0                   # population standard deviation, the same choice as Lab 3
TRAIN_FRACTION = 0.7       # the share of rows that train, stated once
TARGET_COLUMN = "aboard"   # what the next three modules are asked to predict
HANDOFF_SCHEMA = "1.0"     # the version Module 3 refuses to read anything but


def window_features(series, width: int) -> dict:
    """Six numbers describing the last `width` readings.

    Definition graded by the check:
        over the last w readings: x̄, σ with ddof = 0, min, max, slope b = Σ_i
        (i − ī)(x_i − x̄) / Σ_i (i − ī)², and r_1
        (Kuhn & Johnson, 2019, ch. 5). ī is the mean position, so the slope is
        the least-squares gradient against position and is in units of the
        series per reading. The standard deviation is the population one,
        ddof = 0. Slide: "Definition — the six window features".
        r_k = Σ_{t=k+1}^{n} (x_t − x̄)(x_{t−k} − x̄) / Σ_{t=1}^{n} (x_t − x̄)²
        (Box, Jenkins, Reinsel & Ljung, 2015, §2.1.4). One mean over the whole
        window and the whole window's sum of squares underneath. This is not
        what pandas computes -- that is the correlation between
        the series and its shifted copy, with two means and two variances, and
        the two disagree on every finite window. Module 1 grades the same
        formula, so the two modules' numbers are the same quantity. Slide:
        "Definition — lag-k sample autocorrelation".
    Needs: pandas, numpy

    Returns:
        {"mean", "std", "minimum", "maximum", "slope", "autocorrelation_1"}.
    """
    # TODO: take the last `width` values and compute the six.
    raise NotSolved("window_features(series, width) still raises instead of returning features")


def split_by_time(frame, train_fraction: float = 0.7):
    """Split in time order. Everything before the cut trains; everything after tests.

    Definition graded by the check:
        train = {x_t : t < t_c}, test = {x_t : t ≥ t_c} for one cut instant t_c,
        the ⌊f·n⌋-th time in sorted order — never a random permutation of the
        rows
        (Roberts et al., 2017; Bergmeir & Benítez, 2012). The rows you are handed
        are out of order, as the archive's own rows are, so sorting is part of
        the job and not an assumption you may make. Slide: "Definition — a split
        by time, never at random".
    Needs: pandas, int, len

    Returns:
        (train, test).
    """
    # TODO: sort by timestamp_utc, then cut at the fraction.
    raise NotSolved("split_by_time(frame, train_fraction) still raises instead of returning two frames")


def join_growth(phones, bus, tolerances=TOLERANCES) -> dict:
    """What each tolerance buys: {seconds: per cent of phone rows matched}.

    Definition graded by the check:
        match(p) = argmin_b |t_b − t_p| when min_b |t_b − t_p| ≤ τ, else none;
        matched share = |{p : match(p) exists}| / n_p
        (McKinney, 2022). t_p is a phone row's time and t_b a vehicle reading's,
        both in coordinated universal time; τ is the tolerance in seconds. The
        share is a percentage of the phone rows. Note that merge_asof refuses two
        time columns of different resolution, and pandas infers the resolution
        from the text it parsed -- force both to the same unit before joining.
        Slide: "Definition — a nearest match within a tolerance".
    Needs: pandas

    Returns:
        {tolerance in seconds: matched share in per cent}.
    """
    # TODO: nearest match within each tolerance; report the share matched.
    raise NotSolved("join_growth(phones, bus, tolerances) still raises instead of returning shares")


def assemble(aligned, fitted: dict, ledger: dict, target: str = TARGET_COLUMN,
                  train_fraction: float = TRAIN_FRACTION):
    """The table this module hands to Module 3, and the manifest that describes it.

    Definition graded by the check:
        one row per (phone_id, window); mask_c = 1[x_c absent] beside x_c for
        every filled column c and for no other; split_point t_c = the window of
        the ⌊f·n⌋-th row in time order, train = {t < t_c}, test = {t ≥ t_c}; θ
        and the ledger stored unchanged
        (Huyen, 2022; Kuhn & Johnson, 2019, ch. 8). f is `train_fraction` and n
        the rows of the table. The split point is an instant, not a row number,
        which is why the cut is moved to the start of the window the ⌊f·n⌋-th
        row sits in: rows sharing an instant cannot end up on opposite sides of
        a split by time. Slide: "Definition — the table this module hands to the
        next three".
    Needs: pandas, int, len, sorted

    Arguments:
        aligned         one row per phone per window, from Lab 1: `phone_id`,
                        `window`, the feature columns, and the target.
        fitted          the transform from Lab 3: medians, means, stds and the
                        column order. `fitted["columns"]` names the columns that
                        are filled and scaled, in the order they are stored.
        ledger          the conservation ledger from Lab 1, recorded verbatim.
        target          the name of the target column in `aligned`.
        train_fraction  the share of rows that train.

    Returns:
        (table, manifest).

        table    `phone_id`, `window`, then for each stored column its filled and
                 scaled value immediately followed by its mask where anything was
                 filled, then the target, then `split` holding "train" or "test".
                 No gaps are left anywhere in the stored columns.
        manifest a dict carrying at least: schema_version, rows, columns, key,
                 feature_columns, mask_columns, target, split_point, train_rows,
                 test_rows, transform and ledger. The split point is written as
                 text so that the manifest survives a trip through a file.
    """
    # TODO: fill with the stored constants, put the mask beside what you filled,
    # cut at an instant rather than at a row, and write the manifest.
    raise NotSolved("assemble(aligned, fitted, ledger, ...) still raises instead of "
                    "returning a table and a manifest")


if __name__ == "__main__":
    say = narrator(LAB)
    phones = load_phones()
    say.info("window features over the last 60 speeds: %s",
             window_features(phones["speed"], 60))
    train, test = split_by_time(phones, 0.7)
    say.info("train %s  test %s", f"{len(train):,}", f"{len(test):,}")
    show_table(pd.DataFrame({"per cent matched": join_growth(phones, load_bus())})
               .rename_axis("tolerance (s)"), "nearest match within a tolerance",
               logger=say)

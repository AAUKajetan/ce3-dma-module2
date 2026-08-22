"""Lab 2 — Missingness is a mechanism.

Why this lab exists: more than half of the beacon readings in this archive are
absent, and the absence is caused by the very thing the reading would have told
you — distance. You prove here that you can name the mechanism, fill a gap
without destroying the record that you filled it, and measure in decibels how
much each fill invented.
Where it sits: Block two — "The mask, and why it must survive", and the
definition slides "Definition — the three missing-data mechanisms",
"Definition — the absence mask and the masked exponential moving average" and
"Definition — imputation bias".
What the check grades: for all five beacons and all three methods the mask marks
exactly the rows where the signal strength is empty or the proximity column holds
−1; "drop" leaves those rows empty, "mean" fills them with the mean of the
readings that were heard, and "ema_masked" reproduces the stated recursion on a
planted two-phone gap; imputation_bias() equals the mean of fill − truth over the
filled rows on three planted triples and on the day; and
fills_are_biased_which_way() returns "too strong".
Needs: pandas, numpy, plotly, and the loader in lab_support.

Twenty-five minutes.

A beacon reading is absent on most rows. Before you fill anything in, one
question decides everything that follows: *why* is it absent?

  If a packet was lost in transit, the absence tells you nothing, and filling
  it with a plausible value costs little.

  If the beacon was out of range, the absence tells you where the phone was.
  Filling it with the average signal strength does not recover information --
  it invents a phone standing next to a beacon it could not hear, and it does
  so on the majority of your rows.

The second is what happens here, and in the archive: signal strength falls with
distance, so the readings that are missing are exactly the far ones. That is
missing not at random, and it is the case where imputation lies.

There is a second trap, and it is in the real file too. The same absence is
encoded twice: `rssiA` is empty and `proxA` is -1, on exactly the same rows. A
mean taken over `proxA` without recognising the sentinel averages in -1 as
though it were a real proximity band.

What you write: impute_with_mask(frame, method), imputation_bias(filled, truth,
missing), and fills_are_biased_which_way().

impute_with_mask returns a copy of `frame` with two new columns per beacon:

    <beacon>_filled     the values after imputation
    <beacon>_missing    True where the original was absent, before you filled it

The mask is not decoration. It is the only record that the value is a guess,
and every model downstream needs it. Lose it and nobody can tell a measurement
from an invention.

Three methods to support:

    "drop"        do not fill; leave the absent values absent. The mask still
                  has to be right.
    "mean"        fill with the mean of the readings that are present.
    "ema_masked"  the exponential moving average of the readings that were
                  heard, carried forward across the gaps, with the mask kept
                  (Servizi et al., 2023 -- the method this study used).

Then measure what each one invented, against the true values the check holds,
and say which way the fills are wrong before you look.
"""
from __future__ import annotations

import sys
import pathlib

# Every library the reference solution uses is imported here, so that the work
# in front of you is the statistics and not the import lines.
import numpy as np                                                   # noqa: F401
import pandas as pd
import plotly.graph_objects as go                                    # noqa: F401
from plotly.subplots import make_subplots                            # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_phones          # noqa: E402
from _narrate import narrator, show_table, save_figure   # noqa: E402,F401

LAB = 2
BEACONS = ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]
PROXIMITY = {"rssiA": "proxA", "rssiB": "proxB", "rssiC": "proxC",
             "rssi1": "prox1", "rssi2": "prox2"}
SENTINEL = -1
SMOOTHING = 0.3   # alpha, the weight of the newest heard reading -- a stated choice


def impute_with_mask(frame, method: str = "ema_masked"):
    """Fill the absent beacon readings, and keep a record that you did.

    Definition graded by the check:
        m_t = 1 when rssi_t is null or prox_t = −1, else 0
        (van Buuren, 2018). The same absence is written down twice in this
        archive and a mask that recognises one encoding and not the other is
        wrong on every row where they differ.
        s_t = α·x_t + (1−α)·s_{t−1} over the heard readings only, α = 0.3,
        carried forward across a gap
        (Servizi et al., 2023). α = SMOOTHING = 0.3, stated rather than tuned;
        s_t is the plain recursion, not the adjusted one; one series per phone,
        so a gap in one volunteer's trace is filled from that volunteer's own
        last heard readings and from nobody else's. Slide: "Definition — the
        absence mask and the masked exponential moving average".
    Needs: pandas

    Args:
        frame:  phone traces, with beacon columns absent and proximity columns
                carrying the sentinel on the same rows.
        method: "drop", "mean" or "ema_masked".

    Returns:
        A copy with <beacon>_filled and <beacon>_missing for each beacon.
    """
    # TODO: recognise both encodings of absence, build the mask, then fill.
    raise NotSolved("impute_with_mask(frame, method) still raises instead of returning a frame")


def imputation_bias(filled, truth, missing) -> float:
    """How far the values you invented sit from what was really there, in decibels.

    Definition graded by the check:
        bias = (1/|F|) Σ_{t ∈ F} (fill_t − truth_t), F the rows the method
        filled, in decibels
        (Little & Rubin, 2019; van Buuren, 2018). Positive means the fills sit
        above the truth -- "too strong". F is the set of rows the mask marks
        absent *and* the method actually filled, so "drop" invents nothing and
        scores nought by construction. Slide: "Definition — imputation bias".
    Needs: pandas, float

    Args:
        filled:   the column after imputation.
        truth:    the values that were really there, which only the check has.
        missing:  the mask, True where the original reading was absent.

    Returns:
        The mean of (fill − truth) over the filled rows, in decibels; 0.0 when
        the method filled nothing.
    """
    # TODO: the rows that were masked AND filled, then the mean of the difference.
    raise NotSolved("imputation_bias(filled, truth, missing) still raises instead of "
                    "returning a number")


def fills_are_biased_which_way() -> str:
    """Do the filled values come out too strong, or too weak?

    Definition graded by the check:
        MCAR: P(M | Y_obs, Y_mis) = P(M); MAR: P(M | Y_obs, Y_mis) = P(M |
        Y_obs); MNAR: neither
        (Rubin, 1976; Little & Rubin, 2019). M is the indicator that a reading is
        absent, Y_obs what you can see and Y_mis what you cannot. Signal strength
        falls with distance and a beacon is heard only when it is near, so the
        absence depends on the missing value itself: the third case. Slide:
        "Definition — the three missing-data mechanisms".
    Needs: nothing but the mechanism -- return "too strong" or "too weak"

    Reason from the mechanism rather than from habit: the missing readings are
    the far ones. What does filling them with the average of the near ones do to
    the distribution?
    """
    # TODO: return "too strong" or "too weak".
    raise NotSolved("fills_are_biased_which_way() still raises instead of returning "
                    "a direction")


if __name__ == "__main__":
    say = narrator(LAB)
    phones = load_phones()
    truth = load_phones(with_truth=True)
    rows = {}
    for name in ("drop", "mean", "ema_masked"):
        filled = impute_with_mask(phones, name)
        rows[name] = {"masked rows": int(filled["rssi1_missing"].sum()),
                      "bias (decibels)": imputation_bias(
                          filled["rssi1_filled"], truth["rssi1_true"],
                          filled["rssi1_missing"])}
    show_table(pd.DataFrame(rows).T.rename_axis("method"), "rssi1", logger=say)
    say.info("the fills come out: %s", fills_are_biased_which_way())

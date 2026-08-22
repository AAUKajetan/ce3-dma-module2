"""Lab 1 — Audit and align.

Why this lab exists: the vehicle and the phones arrive on two different clocks at
two different rates, and every number the rest of the course produces is computed
from the table you make by putting them together. You prove here that you can
choose a grain, join on the right clock, account for every input row, and tell
the difference between a comparison of two days and a comparison of two fleets.
Where it sits: Block one — "Bronze to silver — what changes, and what must not",
and the definition slides "Definition — a tumbling window on the coordinated
universal time grain", "Definition — the conservation ledger" and
"Definition — Simpson's paradox".
What the check grades: the ledger balances exactly and its reasons sum to the
dropped count, on the shipped pair and on a day where the vehicle stops half way
through; every window starts on the five-second grid in coordinated universal
time inside the span the phones cover, one row per phone per window; the aligned
table is consistent with the machine-readable profile Module 1 produced for the
same slice, which ships here as data/module1_profile.json, and the ledger records
what that profile says about the frame you were handed — graded on three frames
whose answers differ, so neither an empty list nor a remembered complaint will
do; and on the planted two-day
frame the pooled difference, every per-shuttle difference, the reversal and its
name are right — and on two frames that do not reverse, the verdict says so.
Needs: pandas, numpy, plotly, and the loaders in lab_support.

Twenty-five minutes.

Two sources, two clocks, two rates. The vehicle reports about twice a second;
the phones about once. Neither lines up with the other, and the moment you put
them in one table you have made a decision about time whether you meant to or
not.

What you write: align(bus, phones, grain_seconds) and
pooled_versus_by_group(frame, outcome, group, day).

The job: put both sources on one grain -- tumbling windows of `grain_seconds`
-- and account for every input row. Not "most" rows. Every one. Then compare the
two days, once pooled and once inside each shuttle, and say what you find.

Three traps, all of them in the real archive as well as here:

  the wrong clock    Both frames carry a column named `timestamp` and a column
                     holding coordinated universal time. The one named
                     `timestamp` is local time, one hour ahead in January. Use
                     it and your two sources are silently an hour apart, which
                     looks like data rather than like an error.

  the silent drop    A window with no vehicle reading, or a phone row outside
                     the vehicle's operating hours, has nowhere to go. Dropping
                     it quietly is how a table comes to describe a day that did
                     not happen. Count it, and say why.

  the grain itself   Five seconds is a choice. It is not the only one, and the
                     ledger has to record which you made, because every number
                     computed downstream depends on it.

The table Module 1 handed over comes with a description of itself. It is in
data/module1_profile.json, schema `aau-ce3/data-profile/1`, and it is not prose:
per column a declared type, unit, range and tolerated absence, plus the time
column and the rate the source is expected to report at. It is what Module 1's
`declare_profile()` wrote, byte for byte.

So do not read it -- run it. `load_module1_profile()` and `check_against(frame,
profile)` are both in lab_support, the second copied unchanged from Module 1,
and `align` calls them **before it joins anything** and puts the answer in the
ledger under `profile_breaches`. An empty list is a result: it says the frame
arrived as promised. Try the same call on one day of the slice instead of both
and watch it stop being empty -- the pooled frame satisfies a declaration that
its own first day does not, which is the paradox in the other half of this lab
wearing a validation layer's clothes.

The check also insists that you aligned the whole table Module 1
profiled, that your grain is not finer than the interval the vehicle actually
reports at, and that the negative speeds Module 1's declared range admits are
still there. They
are a property of the instrument, and the previous module wrote them down so
that this one would not quietly tidy them away.

Return (aligned, ledger):

    aligned   one row per (phone_id, window_start), with the phone's readings
              in that window and the vehicle's mean speed and mean payload in
              the same window. Window start is a timestamp in coordinated
              universal time.

    ledger    a dict recording, at minimum:
                grain_seconds          the choice you were given
                bus_rows_in            rows of `bus` you received
                phone_rows_in          rows of `phones` you received
                windows_out            rows in `aligned`
                phone_rows_used        phone rows that landed in an output row
                phone_rows_dropped     phone rows that did not
                drop_reasons           {reason: count}, summing to phone_rows_dropped
                profile_breaches       what Module 1's profile says about `bus`,
                                       as `check_against` returns it: the list of
                                       complaints, empty when the frame satisfies
                                       the declaration it arrived with

The check asserts conservation: phone_rows_used + phone_rows_dropped must equal
phone_rows_in, exactly. A ledger that does not balance is the whole lesson.
`drop_reasons` may be empty on the shipped pair -- nothing was lost, so there is
nothing to explain -- and may not be on the interrupted day.

The second function is the one that changes how you read every two-day table
after today. Two shuttles ran on 22 January and one on 23 January, so a pooled
comparison of the days is partly a comparison of fleets. Report both.
"""
from __future__ import annotations

import sys
import pathlib

# Every library the reference solution uses is imported here, so that the work
# in front of you is the statistics and not the import lines.
import numpy as np                                                   # noqa: F401
import pandas as pd
import plotly.graph_objects as go                                    # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (NotSolved, load_bus, load_phones, load_simpson,   # noqa: E402
                         load_module1_profile, check_against)
from _narrate import narrator, show_table, save_figure                  # noqa: E402,F401

LAB = 1


def align(bus, phones, grain_seconds: int = 5):
    """Put two sources on one grain, and account for every row.

    Definition graded by the check:
        w_j = [t_0 + j·Δ, t_0 + (j+1)·Δ), Δ = 5 s, t_0 = 00:00:00 in coordinated
        universal time, from utc_time
        (Akidau et al., 2015). Δ is `grain_seconds`; flooring a timestamp to the
        grain computes j. Slide: "Definition — a tumbling window on the
        coordinated universal time grain".
        used + dropped = received, with Σ_reason drop_reasons[reason] = dropped
        (Wang & Strong, 1996). Every phone row you were handed is either used or
        dropped with a reason; nothing is both and nothing is neither. Slide:
        "Definition — the conservation ledger".
        breaches(X, P) = the complaints P makes about X, over presence, type,
        range, tolerated absence and sampling step, in that order; X arrives fit
        for use exactly when breaches(X, P) is empty
        (Wang & Strong, 1996; Module 1's `declare_profile` writes P). P is the
        profile Module 1 handed over, `load_module1_profile()`; run
        `check_against(bus, P)` before you align anything and put the answer in
        the ledger, because a complaint that is not recorded on arrival is a
        complaint that will be argued about later. Slide: "Definition — the
        upstream profile, and checking a frame against it".
    Needs: pandas, len, load_module1_profile, check_against

    Returns:
        (aligned, ledger) -- see the module docstring for both shapes.
    """
    # TODO: run `bus` against Module 1's profile first, then parse the right time
    # column, build the windows, join, and keep the books.
    raise NotSolved("align(bus, phones, grain_seconds) still raises instead of returning a table")


def pooled_versus_by_group(frame, outcome: str, group: str, day: str) -> dict:
    """Compare the two days pooled, and inside each group, and say whether they disagree.

    Definition graded by the check:
        P(Y|D=1) > P(Y|D=0) while P(Y|D=1,G=g) < P(Y|D=0,G=g) for every group g
        (Simpson, 1951; the name is Blyth's, 1972; the causal reading is Pearl's,
        2014). Here Y is `outcome` (aboard), D is `day` and G is `group` (the
        shuttle ridden). The reversal needs every group to move, and to move
        against the pool. Slide: "Definition — Simpson's paradox".
    Needs: pandas, numpy

    Returns a dict with:
        days      the two values of `day`, sorted, earlier first
        pooled    the later day's share of `outcome` minus the earlier day's,
                  over the whole frame. A share, so 0.05 is five points.
        by_group  the same difference inside each value of `group`
        shares    {group or "pooled": {day: share}}, for a figure
        reversal  True when every group moves against the pooled direction
        name      what that is called, when it is one, else None
    """
    # TODO: shares by day pooled and by group, the two differences, then the verdict.
    raise NotSolved("pooled_versus_by_group(frame, outcome, group, day) still raises "
                    "instead of returning the two comparisons")


if __name__ == "__main__":
    say = narrator(LAB)
    aligned, ledger = align(load_bus(), load_phones(), 5)
    say.info("aligned: %s rows", f"{len(aligned):,}")
    say.info("what Module 1's profile says about the vehicle frame we were handed: %s",
             ledger["profile_breaches"] or "nothing — it satisfies the declaration")
    show_table(pd.DataFrame({"count": ledger})
               .drop(index=["drop_reasons", "profile_breaches"])
               .rename_axis("ledger"), "the ledger", logger=say)
    verdict = pooled_versus_by_group(load_simpson(), "aboard", "bus", "day")
    say.info("pooled %+.3f, by group %s, reversal %s, named %r", verdict["pooled"],
             verdict["by_group"], verdict["reversal"], verdict["name"])

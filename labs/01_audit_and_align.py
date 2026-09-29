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
SIMPSON = "Simpson's paradox"

def align(bus, phones, grain_seconds: int = 5):
    """Put two sources on one grain, and account for every row.

    The three decisions, made explicitly rather than by accident:

    1. **Which clock.** Both frames carry a column named `timestamp` that is
       local time and a column that is coordinated universal time. In January
       Copenhagen is one hour ahead, so using the friendly-looking name puts the
       two sources an hour apart -- and nothing complains, because both columns
       parse and both look like times. Always join on the coordinated column.

    2. **The grain.** Five seconds is a choice, not a fact. The vehicle reports
       twice a second and the phones once, so any common grain throws some
       resolution away; a coarser one throws away more and matches more. The
       ledger records which was used, because every downstream number depends
       on it. Tumbling windows w_j = [t_0 + j·Δ, t_0 + (j+1)·Δ), Δ the grain,
       t_0 midnight in coordinated universal time -- which is what flooring a
       timestamp to the grain computes.

    3. **What happens to the leftovers.** A phone row in a window with no
       vehicle reading has nowhere to go. Dropping it silently is how a table
       comes to describe a day that did not happen, so it is counted and given
       a reason. The check adds the buckets up and insists they balance:
       used + dropped = received, and the reasons sum to dropped.
    """
    grain = f"{grain_seconds}s"

    # Before anything is joined: what does the module upstream say about the
    # frame that just arrived? Module 1 declared this table's columns, their
    # ranges, the absence each one tolerates and the rate it reports at, and
    # `check_against` is Module 1's own function applied to what we were handed.
    #
    # Recording the answer is the point, not passing it. An empty list is a
    # measurement too: it says the frame arrived as promised, and it is the
    # sentence that makes the non-empty case worth anything. Note what happens
    # on the whole slice against one day of it -- the pooled frame satisfies the
    # declaration and the first day alone does not, which is the same shape as
    # the paradox in the other half of this lab.
    breaches = check_against(bus, load_module1_profile())

    bus = bus.copy()
    bus["_t"] = pd.to_datetime(bus["utc_time"], utc=True)
    bus["window"] = bus["_t"].dt.floor(grain)
    per_window_bus = bus.groupby("window").agg(
        bus_speed=("speed", "mean"),
        bus_payload=("payload", "mean"),
        bus_readings=("speed", "size"),
    )

    phones = phones.copy()
    phones["_t"] = pd.to_datetime(phones["timestamp_utc"], utc=True, errors="coerce")

    # Reason one: a row whose clock will not parse cannot be placed in time.
    unparseable = int(phones["_t"].isna().sum())
    phones = phones.dropna(subset=["_t"])
    phones["window"] = phones["_t"].dt.floor(grain)

    per_window_phone = phones.groupby(["phone_id", "window"]).agg(
        phone_readings=("speed", "size"),
        phone_speed=("speed", "mean"),
        rssi1=("rssi1", "mean"),
        rssi2=("rssi2", "mean"),
        aboard=("label2", lambda values: (values == "IN").mean()),
    ).reset_index()

    aligned = per_window_phone.merge(
        per_window_bus, left_on="window", right_index=True, how="inner")

    # Reason two: a window with no vehicle reading at all. The inner join above
    # removed those rows, so they are counted here rather than lost.
    kept_windows = set(zip(aligned["phone_id"], aligned["window"]))
    used = int(per_window_phone.apply(
        lambda row: (row["phone_id"], row["window"]) in kept_windows, axis=1)
        .mul(per_window_phone["phone_readings"]).sum())

    dropped_no_vehicle = int(per_window_phone["phone_readings"].sum() - used)

    ledger = {
        "grain_seconds": grain_seconds,
        "bus_rows_in": int(len(bus)),
        "phone_rows_in": int(len(phones) + unparseable),
        "windows_out": int(len(aligned)),
        "phone_rows_used": used,
        "phone_rows_dropped": dropped_no_vehicle + unparseable,
        "drop_reasons": {
            "timestamp would not parse": unparseable,
            "no vehicle reading in the window": dropped_no_vehicle,
        },
        "profile_breaches": breaches,
    }
    return aligned.reset_index(drop=True), ledger


    
    raise NotSolved("align(bus, phones, grain_seconds) still raises instead of returning a table")

def pooled_versus_by_group(frame, outcome: str, group: str, day: str) -> dict:
    """The two-day comparison, pooled and inside each group -- and whether they disagree.

    Before you compare two days you have to know what each day is made of. The
    archive's two days are not the same fleet: two shuttles ran on 22 January
    and one on 23 January, so a pooled comparison of the days is partly a
    comparison of shuttles. In the generated phones the same thing is planted on
    purpose: the second day has fewer volunteers and most of them rode the
    shuttle whose riders spend the smaller share of their time aboard.

    So the aboard share *rises* on each shuttle and *falls* pooled. Both numbers
    are correct; they answer different questions. When the sign inside every
    group is the opposite of the pooled sign, that is Simpson's paradox
    (Simpson, 1951; the name is Blyth's, 1972), and the honest report says which
    comparison you made -- Pearl (2014) is the reading on which one to trust,
    and the answer depends on what caused the mix to change.

    Returns a dict with `days` (the two day values, sorted), `pooled` (the
    later day's share minus the earlier day's, pooled), `by_group` (the same
    difference inside each group), `shares` (the shares themselves, for a
    figure), `reversal` (True when every group moves against the pool) and
    `name` (Simpson's paradox when it is one, else None). Differences are in
    shares, so 0.05 is five percentage points.
    """
    days = sorted(pd.unique(frame[day]))
    assert len(days) == 2, f"expected two values of {day!r}, found {len(days)}"
    earlier, later = days
    aboard = frame[outcome].astype(bool)

    def share(rows) -> float:
        return float(aboard[rows].mean()) if rows.any() else float("nan")

    on_day = {d: frame[day] == d for d in days}
    shares = {"pooled": {d: share(on_day[d]) for d in days}}
    for g in sorted(pd.unique(frame[group])):
        shares[g] = {d: share(on_day[d] & (frame[group] == g)) for d in days}

    pooled = shares["pooled"][later] - shares["pooled"][earlier]
    by_group = {g: shares[g][later] - shares[g][earlier]
                for g in shares if g != "pooled"}

    # A reversal needs every group to move, and to move against the pool. A
    # group that did not move, or was absent on one day, settles nothing.
    moves = [d for d in by_group.values() if not np.isnan(d)]
    reversal = bool(
        pooled != 0 and len(moves) == len(by_group) and moves
        and all(d != 0 and np.sign(d) == -np.sign(pooled) for d in moves))

    return {"days": (earlier, later), "pooled": pooled, "by_group": by_group,
            "shares": shares, "reversal": reversal,
            "name": SIMPSON if reversal else None}



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

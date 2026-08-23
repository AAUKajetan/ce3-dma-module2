"""Lab 1, solved — with the reasoning, not only the code."""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (NotSolved, load_bus, load_phones, load_simpson,
                         load_module1_profile, check_against)  # noqa: E402,F401
from _narrate import narrator, show_table, save_figure                  # noqa: E402

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
    say.info("Lab 1 — one grain, a ledger that balances, and a two-day comparison "
             "that reverses when it is pooled")

    bus, phones = load_bus(), load_phones()
    say.info("vehicle telemetry: %s rows (archive slice, shuttle VJRD1A10224000055); "
             "phones: %s rows (generated, seed 20200122, first day)", f"{len(bus):,}",
             f"{len(phones):,}")
    say.info("grain 5 s on utc_time, because the vehicle reports twice a second and "
             "the phones once — neither rate divides the other")
    aligned, ledger = align(bus, phones, 5)
    say.info("aligned table: %s rows, one per phone per 5-second window; "
             "%s of %s phone rows landed", f"{len(aligned):,}",
             f"{ledger['phone_rows_used']:,}", f"{ledger['phone_rows_in']:,}")
    say.info("Module 1's profile, run over the vehicle frame before anything was "
             "joined: %s", ledger["profile_breaches"] or "no complaints — the frame "
             "arrived as the declaration promised")
    show_table(pd.DataFrame({"count": ledger})
               .drop(index=["drop_reasons", "profile_breaches"])
               .rename_axis("ledger"), "the ledger", logger=say)
    say.info("drop reasons: %s — every dropped row has one", ledger["drop_reasons"])
    say.info("conservation: %s used + %s dropped = %s received: %s",
             ledger["phone_rows_used"], ledger["phone_rows_dropped"],
             ledger["phone_rows_in"],
             ledger["phone_rows_used"] + ledger["phone_rows_dropped"] == ledger["phone_rows_in"])

    # A day where the vehicle stops half way through, so that the ledger has
    # something to account for -- the same day the check uses.
    utc = pd.to_datetime(phones["timestamp_utc"], utc=True)
    midpoint = utc.min() + (utc.max() - utc.min()) / 2
    interrupted = bus[pd.to_datetime(bus["utc_time"], utc=True) < midpoint]
    _, short = align(interrupted, phones, 5)
    say.info("vehicle stopping at %s: %s phone rows dropped, reasons %s — the books "
             "still balance: %s", midpoint.strftime("%H:%M:%S"),
             f"{short['phone_rows_dropped']:,}", short["drop_reasons"],
             short["phone_rows_used"] + short["phone_rows_dropped"] == short["phone_rows_in"])

    # The same declaration, on one day of the frame instead of both. It is the
    # module's own paradox in a validation layer: the aggregate check passes and
    # the per-group check fails, so an examination made only on the pool would
    # have reported nothing wrong.
    day_one = bus[pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str)
                  == "2020-01-22"]
    _, per_day = align(day_one, phones, 5)
    say.info("profile over both days: %s; over 22 January alone: %s — an examination "
             "made on the pool only would have found nothing",
             ledger["profile_breaches"] or "no complaints", per_day["profile_breaches"])

    simpson = load_simpson()
    say.info("both generated days: %s rows, %s phones on %s and %s on %s",
             f"{len(simpson):,}",
             simpson.loc[simpson['day'] == '2020-01-22', 'phone_id'].nunique(), "2020-01-22",
             simpson.loc[simpson['day'] == '2020-01-23', 'phone_id'].nunique(), "2020-01-23")
    verdict = pooled_versus_by_group(simpson, "aboard", "bus", "day")
    table = pd.DataFrame(verdict["shares"]).T.mul(100).round(1)
    table.columns = [f"{d} (per cent aboard)" for d in table.columns]
    show_table(table, "aboard share by day, pooled and per shuttle", logger=say)
    say.info("pooled, later day minus earlier: %+.1f points; per shuttle: %s points",
             verdict["pooled"] * 100,
             {g: f"{d * 100:+.1f}" for g, d in verdict["by_group"].items()})
    say.info("every shuttle moves against the pool: reversal=%s, named %r — the pooled "
             "line compares fleets, not days", verdict["reversal"], verdict["name"])

    earlier, later = verdict["days"]
    groups = [g for g in verdict["shares"] if g != "pooled"] + ["pooled"]
    fig = go.Figure()
    for d, colour in ((earlier, "#2A78D6"), (later, "#E07B39")):
        fig.add_bar(name=d, x=groups, y=[verdict["shares"][g][d] * 100 for g in groups],
                    marker_color=colour,
                    text=[f"{verdict['shares'][g][d] * 100:.1f}" for g in groups],
                    textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="aboard share (per cent of readings)",
                      xaxis_title="shuttle ridden (generated phones, both days)",
                      yaxis_range=[0, 100],
                      title="Up on each shuttle, down when pooled — Simpson's paradox")
    save_figure(fig, "simpson", LAB, logger=say)

    say.info("what the check grades: used + dropped = received with the reasons summing to "
             "dropped, on the shipped pair and on the interrupted day; windows on the "
             "5-second UTC grid inside the phones' span; and on the planted frame the "
             "pooled difference, every group difference, reversal=True and the name")

#!/usr/bin/env python3
"""Check 1 — one grain, a ledger that balances, and a comparison that reverses."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, not_ready, explain                          # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from lab_support import (load_bus, load_phones, load_simpson,     # noqa: E402
                             load_module1_profile, check_against)
    import json                                                       # noqa: E402
    import pandas as pd                                               # noqa: E402
except ImportError as unready:
    not_ready(unready)

GRAIN = 5

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
MODULE1_PROFILE = REPOSITORY / "data" / "module1_profile.json"

# The shape Module 1's machine-readable profile has to have for this module to
# consume it. Written out here rather than imported, because the whole point of a
# hand-off is that the consumer states what it needs: a schema that lives only in
# the producer is a promise nobody can check.
#
# These are the keys `declare_profile()` writes in Module 1's Lab 3, and the
# shipped fixture is byte-for-byte that function's output on this same slice.
PROFILE_SCHEMA = ("schema", "module", "dataset", "rows", "time_column",
                  "expected_step_seconds", "step_tolerance_share", "columns")
PROFILE_VERSION = "aau-ce3/data-profile/1"

# The day the declaration does not hold on. Module 1 declared the absence each
# column tolerates over BOTH days pooled; on 22 January alone `emergency_stop`
# is absent more often than that. The frame passes and its first day fails --
# the same shape as the paradox this lab's other half is about, in a validation
# layer. It is measured here, not typed: `check_against` is called on each frame.
DAY_THE_PROFILE_FIRES_ON = "2020-01-22"


def expand(counts) -> "pd.DataFrame":
    """Rows from a table of (group, day, aboard rows, rows), for the controls.

    Writing the control as counts rather than as rows keeps the arithmetic on
    the page: a reader can add the columns up and see for themselves that the
    frame does or does not reverse.
    """
    rows = []
    for group, day, aboard, total in counts:
        rows += [{"bus": group, "day": day, "aboard": True}] * aboard
        rows += [{"bus": group, "day": day, "aboard": False}] * (total - aboard)
    return pd.DataFrame(rows)


def shares_and_differences(frame, outcome, group, day):
    """The check's own arithmetic: the pooled difference and the per-group ones."""
    days = sorted(pd.unique(frame[day]))
    earlier, later = days
    aboard = frame[outcome].astype(bool)

    def share(rows):
        return float(aboard[rows].mean()) if rows.any() else float("nan")

    pooled = share(frame[day] == later) - share(frame[day] == earlier)
    by_group = {}
    for g in sorted(pd.unique(frame[group])):
        inside = frame[group] == g
        by_group[g] = (share(inside & (frame[day] == later))
                       - share(inside & (frame[day] == earlier)))
    return pooled, by_group


def grade_pooled_versus_by_group(lab):
    """The planted reversal, and two frames that do not reverse.

    Three calls, because one is not enough. On the planted frame a student who
    returns `reversal=True` and the name unconditionally is right by accident;
    the two controls are frames where the honest answer is False, and they are
    what separates a measurement from a memory.
    """
    planted = load_simpson()
    verdict = lab.pooled_versus_by_group(planted, "aboard", "bus", "day")
    for key in ("days", "pooled", "by_group", "shares", "reversal", "name"):
        assert key in verdict, f"pooled_versus_by_group() returned no {key!r}"

    pooled, by_group = shares_and_differences(planted, "aboard", "bus", "day")
    assert abs(verdict["pooled"] - pooled) <= 1e-9, (
        f"the pooled difference reads {verdict['pooled']:+.4f}; over the whole frame, "
        f"later day minus earlier, it is {pooled:+.4f}. Shares, not percentages: "
        "0.05 is five percentage points.")
    assert set(verdict["by_group"]) == set(by_group), (
        f"by_group covers {sorted(verdict['by_group'])}; the frame has "
        f"{sorted(by_group)}. Every group gets its own difference.")
    for name, difference in by_group.items():
        assert abs(verdict["by_group"][name] - difference) <= 1e-9, (
            f"inside {name!r} the difference reads {verdict['by_group'][name]:+.4f}; "
            f"it is {difference:+.4f}.")

    assert verdict["reversal"] is True or verdict["reversal"] == 1, (
        f"the pooled difference is {pooled:+.4f} and every group moves the other way "
        f"({ {k: round(v, 4) for k, v in by_group.items()} }). That is the reversal "
        "this lab is about, and reversal is False.")
    # Naming it is the examinable half, so the message must not do the naming.
    assert "simpson" in str(verdict["name"]).lower(), explain(
        "pooled_versus_by_group:name",
        f"the reversal is reported and the verdict is named {verdict['name']!r}, which "
        "is not what this is called",
        "It has a name because it is a known shape, and a verdict that describes the "
        "shape without naming it cannot be looked up by whoever reads your report. "
        "The definition slide for it is in this block, and the name is the surname of "
        "the statistician who set the two-by-two-by-two table out in 1951 -- the paper "
        "is on the References slide and in READING.md.")

    # Control one: one shuttle only, so the pooled comparison and the group
    # comparison are the same comparison and nothing can reverse.
    one_bus = planted[planted["bus"] == sorted(pd.unique(planted["bus"]))[0]]
    single = lab.pooled_versus_by_group(one_bus, "aboard", "bus", "day")
    assert not single["reversal"], (
        "with a single shuttle in the frame the pooled difference IS the group "
        "difference, so nothing can reverse — and reversal came back True. The answer "
        "has to be measured from the frame in front of you.")
    assert single["name"] is None, (
        f"nothing reversed and the verdict is still named {single['name']!r}.")

    # Control two: both groups and the pool move the same way. Counts, so the
    # arithmetic is on the page: 30/100 to 50/100 and 60/100 to 80/100, both up,
    # and the pool up with them.
    agreeing = expand([("Bus 1", "2020-01-22", 30, 100), ("Bus 1", "2020-01-23", 50, 100),
                       ("Bus 2", "2020-01-22", 60, 100), ("Bus 2", "2020-01-23", 80, 100)])
    quiet = lab.pooled_versus_by_group(agreeing, "aboard", "bus", "day")
    agreeing_pooled, agreeing_by_group = shares_and_differences(
        agreeing, "aboard", "bus", "day")
    assert abs(quiet["pooled"] - agreeing_pooled) <= 1e-9, (
        f"on a frame that rises everywhere the pooled difference reads "
        f"{quiet['pooled']:+.4f}; it is {agreeing_pooled:+.4f}.")
    for name, difference in agreeing_by_group.items():
        assert abs(quiet["by_group"][name] - difference) <= 1e-9, (
            f"on a frame that rises everywhere the difference inside {name!r} reads "
            f"{quiet['by_group'][name]:+.4f}; it is {difference:+.4f}.")
    assert not quiet["reversal"], (
        "every group rises by twenty percentage points and the pool rises with them. "
        "There is no reversal here, and reversal came back True — a verdict that is "
        "always True says nothing about any frame.")
    assert quiet["name"] is None, (
        f"nothing reversed and the verdict is still named {quiet['name']!r}.")


def _interrupted_vehicle(bus, phones):
    """The vehicle stopping half way through the phones' day.

    One definition, called from both graders, so that the frame the ledger is
    graded on and the frame the profile is graded on are the same frame.
    """
    phone_utc = pd.to_datetime(phones["timestamp_utc"], utc=True)
    midpoint = phone_utc.min() + (phone_utc.max() - phone_utc.min()) / 2
    return bus[pd.to_datetime(bus["utc_time"], utc=True) < midpoint]


def grade_against_module1_profile(lab, bus, phones, aligned, ledger):
    """The frame was run against Module 1's profile, and nothing was tidied away.

    Module 1's last lab writes two profiles of the vehicle slice: DATA_PROFILE.md
    for a person, and `out/data_profile.json` for a program. The second is a data
    dictionary with rules in it, and it is the hand-off into this module. A
    hand-off nobody reads is a document rather than a contract, so this grades
    three things: that the declaration is the one Module 2 was written for, that
    Lab 1 ran the incoming frame against it and recorded the answer, and that the
    align did not quietly repair what the declaration says is allowed.

    Shipped as a fixture rather than read out of `Module 1/exercises/out/`: a
    check that reaches into a sibling module's output directory is green only on
    a machine where that module has been run, and red for reasons that have
    nothing to do with the student's own work. `data/prepare.py` buys back what
    the live dependency was worth, by running the declaration over the slice on
    every setup and refusing to prepare when it no longer holds.
    """
    if not MODULE1_PROFILE.exists():
        raise AssertionError(
            f"{MODULE1_PROFILE.name} is not in data/. It is Module 1's profile of the "
            "vehicle slice, shipped with this module; run  bash setup.sh  to restore it.")
    profile = load_module1_profile()

    # 1. The contract itself. A consumer states what it needs and refuses a
    #    version it has not been written for.
    missing = [key for key in PROFILE_SCHEMA if key not in profile]
    assert not missing, (
        f"the data profile is missing {missing}. Module 2 consumes "
        f"{', '.join(PROFILE_SCHEMA)}; a profile without them cannot be graded against.")
    assert profile["schema"] == PROFILE_VERSION, (
        f"the data profile says schema {profile['schema']!r} and this module was "
        f"written for {PROFILE_VERSION!r}. A hand-off that changes shape without "
        "changing version is how two modules come to disagree in silence.")

    # 2. You aligned the table Module 1 profiled, and all of it.
    assert ledger["bus_rows_in"] == profile["rows"], (
        f"the ledger received {ledger['bus_rows_in']:,} vehicle rows; Module 1 profiled "
        f"{profile['rows']:,} of them in {profile['dataset']}. Aligning a subset of the "
        "table the previous module described means every rule it declared — the ranges, "
        "the tolerated absences, the reporting rate — no longer applies to what you "
        "built.")

    # 3. The grain cannot be finer than the source's own reporting interval.
    #    Module 1 declared that interval; asking for a window smaller than it
    #    manufactures windows the vehicle never reported into.
    interval = float(profile["expected_step_seconds"])
    assert ledger["grain_seconds"] >= interval, explain(
        "align:grain_below_interval",
        f"the ledger records a grain of {ledger['grain_seconds']} s, and Module 1 "
        f"declared the vehicle reporting every {interval} s",
        "A window shorter than the source's own reporting interval is a window most "
        "of whose rows have no reading in them at all. The aggregate is then either "
        "empty or a single reading wearing the word 'mean', and the ledger fills up "
        "with drops that describe your choice of grain rather than the data.")

    # 4. The frame was run against the declaration, and the answer was recorded.
    #    Graded on three frames whose answers differ, and each expected answer is
    #    computed here by calling the same function the lab is told to call: a
    #    hard-coded empty list fails on the third, and a hard-coded complaint
    #    fails on the first two. Nothing here can drift from the fixture, because
    #    nothing here is typed out.
    day_one = bus[pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str)
                  == DAY_THE_PROFILE_FIRES_ON]
    interrupted = _interrupted_vehicle(bus, phones)
    for label, frame, given in (
            ("the whole slice", bus, ledger),
            ("the interrupted day", interrupted, None),
            (f"{DAY_THE_PROFILE_FIRES_ON} alone", day_one, None)):
        recorded = given if given is not None else lab.align(frame, phones, GRAIN)[1]
        assert "profile_breaches" in recorded, explain(
            "align:no_profile_breaches",
            "the ledger has no 'profile_breaches'. Lab 1 runs the vehicle frame it "
            "was handed against Module 1's profile before it aligns anything, and "
            "records what came back",
            "A frame that arrives is a frame somebody upstream made promises about. "
            "Reading those promises costs one call — load_module1_profile() and "
            "check_against(bus, profile) — and not reading them means the first "
            "person to notice the breach is whoever is on call for the model.")
        expected = check_against(frame, profile)
        assert list(recorded["profile_breaches"]) == expected, explain(
            f"align:profile_breaches:{label}",
            f"on {label} the ledger records profile_breaches = "
            f"{list(recorded['profile_breaches'])!r} and Module 1's declaration says "
            f"{expected!r}",
            "The answer depends on the frame, which is the entire point: the whole "
            "slice satisfies the declaration and its first day alone does not, "
            "because the absence Module 1 tolerated was measured over both days "
            "pooled. Call check_against on the frame you were handed, and record "
            "what it returns — do not decide in advance what it ought to be.")

    # 5. Module 1 declared a negative minimum for the vehicle's speed, which is a
    #    statement that the reversals in this archive are a property of the
    #    instrument rather than a defect to be tidied away. A table whose vehicle
    #    speed never goes below nought has quietly repaired it on the way through.
    declared_minimum = float(profile["columns"]["speed"]["minimum"])
    if declared_minimum < 0:
        negative = int((bus["speed"] < 0).sum())
        assert float(aligned["bus_speed"].min()) < 0, explain(
            "align:negative_speed_lost",
            f"Module 1 declared a minimum vehicle speed of {declared_minimum:+.4g} m/s, "
            f"there are {negative:,} rows below nought in the slice, and the lowest "
            f"vehicle speed in your aligned table is "
            f"{float(aligned['bus_speed'].min()):+.4g}",
            "Those rows are a property of the instrument, and the previous module "
            "declared a range that admits them so that this one would not silently "
            "lose them. Filtering them out, clipping them at nought, or taking an "
            "absolute value all produce a table that describes a vehicle that never "
            "reversed.")

    # 6. One row per phone per window, and no window holding a phone twice.
    assert not aligned.duplicated(subset=["phone_id", "window"]).any(), (
        "the grain is one row per phone per window, so nothing in the source can "
        "justify two rows for one phone in one window in your output.")


def body(lab):
    bus, phones = load_bus(), load_phones()
    aligned, ledger = lab.align(bus, phones, GRAIN)

    for key in ("grain_seconds", "bus_rows_in", "phone_rows_in", "windows_out",
                "phone_rows_used", "phone_rows_dropped", "drop_reasons"):
        assert key in ledger, f"the ledger has no '{key}'"

    assert ledger["grain_seconds"] == GRAIN, (
        f"the ledger records grain {ledger['grain_seconds']}, you were asked for {GRAIN}")
    assert ledger["phone_rows_in"] == len(phones), (
        f"the ledger says {ledger['phone_rows_in']} phone rows came in; there were "
        f"{len(phones)}. Count what you were given, not what survived.")

    # The whole point of the lab: the books balance, exactly.
    total = ledger["phone_rows_used"] + ledger["phone_rows_dropped"]
    assert total == ledger["phone_rows_in"], (
        f"{ledger['phone_rows_used']} used + {ledger['phone_rows_dropped']} dropped "
        f"= {total}, but {ledger['phone_rows_in']} came in. "
        f"{abs(ledger['phone_rows_in'] - total)} rows are unaccounted for — that is "
        "exactly the silent loss this lab exists to prevent.")

    assert sum(ledger["drop_reasons"].values()) == ledger["phone_rows_dropped"], (
        "drop_reasons does not add up to phone_rows_dropped. Every dropped row needs "
        "a reason, and every reason needs a count.")
    # An empty dict is the honest answer when nothing was dropped, and on this
    # pair nothing is: the vehicle reports right across the phone window, so
    # every phone row has somewhere to go. Insisting on a reason here would force
    # a student either to invent a loss or to write a bucket holding nought,
    # which says less than saying nothing.
    assert ledger["drop_reasons"] or ledger["phone_rows_dropped"] == 0, (
        f"{ledger['phone_rows_dropped']} rows were dropped and drop_reasons is "
        "empty. A row that vanished without a reason is the silent loss this lab "
        "exists to prevent.")

    assert len(aligned) == ledger["windows_out"], "windows_out disagrees with the table"
    assert len(aligned) > 0, "the aligned table is empty"

    for column in ("phone_id", "window", "bus_speed"):
        assert column in aligned.columns, f"the aligned table has no '{column}' column"

    assert not aligned.duplicated(subset=["phone_id", "window"]).any(), (
        "the aligned table has more than one row for the same phone in the same "
        "window. One row per phone per window is what 'one grain' means.")

    # The clock trap. Windows must sit on the coordinated universal time grid.
    windows = pd.to_datetime(aligned["window"], utc=True)
    seconds = windows.dt.second + windows.dt.minute * 60
    assert (seconds % GRAIN == 0).all(), (
        f"window starts are not on a {GRAIN}-second boundary in coordinated universal "
        "time. Did you floor the column named `timestamp`? That one is local time — "
        "one hour ahead in January — and joining on it puts your two sources an hour "
        "apart while everything still parses.")

    # Every window must fall inside the span the phones actually cover, in
    # coordinated universal time. Flooring the local column shifts everything an
    # hour later and lands the whole table outside the data it claims to
    # describe -- while every value still parses and every boundary still looks
    # right. Bound it on both sides, or the error passes.
    phone_utc = pd.to_datetime(phones["timestamp_utc"], utc=True)
    first, last = phone_utc.min().floor(f"{GRAIN}s"), phone_utc.max()
    outside = int(((windows < first) | (windows > last)).sum())
    assert outside == 0, (
        f"{outside:,} of {len(windows):,} windows fall outside the span the phones "
        f"cover ({first} to {last}); yours run {windows.min()} to {windows.max()}. "
        "An offset of about an hour means you floored the column named `timestamp`, "
        "which is local time. Both columns parse and both look like times, which is "
        "exactly why this one is worth a check.")

    # Conservation is the headline lesson of this lab, and on the shipped pair it
    # is satisfied by doing nothing at all: the vehicle reports across the whole
    # phone window, so 10,800 rows arrive, 10,800 are used, and a ledger of
    # hard-coded numbers balances exactly as well as a measured one. So run it
    # once more on a day where the vehicle stopped reporting half way through.
    # Now roughly half the phone rows have no vehicle reading to join to, and the
    # books have to be kept for the sum to come out right.
    interrupted = _interrupted_vehicle(bus, phones)

    _, short_ledger = lab.align(interrupted, phones, GRAIN)
    assert short_ledger["phone_rows_in"] == len(phones), (
        f"the ledger says {short_ledger['phone_rows_in']} phone rows came in; there "
        f"were {len(phones)}. The vehicle stopping early changes what can be used, "
        "not what arrived.")
    assert short_ledger["phone_rows_dropped"] > 0, (
        "the vehicle stops reporting half way through the phone window here, so the "
        "phone rows in the later windows have nothing to join to and cannot reach the "
        "output. Your ledger dropped none of them. On the shipped pair nothing is "
        "ever droppable, so a dropped count that is always nought balances there and "
        "is wrong here — count what actually failed to land.")
    short_total = (short_ledger["phone_rows_used"]
                   + short_ledger["phone_rows_dropped"])
    assert short_total == short_ledger["phone_rows_in"], (
        f"{short_ledger['phone_rows_used']} used + "
        f"{short_ledger['phone_rows_dropped']} dropped = {short_total}, but "
        f"{short_ledger['phone_rows_in']} came in on the interrupted day. "
        f"{abs(short_ledger['phone_rows_in'] - short_total)} rows are unaccounted "
        "for.")
    assert (sum(short_ledger["drop_reasons"].values())
            == short_ledger["phone_rows_dropped"]), (
        "on the interrupted day drop_reasons does not add up to phone_rows_dropped. "
        "Every dropped row needs a reason, and every reason needs a count.")
    assert short_ledger["drop_reasons"], (
        "rows were dropped on the interrupted day and not one reason was recorded. "
        "'Where did the other half of the day go?' is asked months later, by someone "
        "who has only your ledger to answer it with.")

    grade_against_module1_profile(lab, bus, phones, aligned, ledger)

    grade_pooled_versus_by_group(lab)


run(1, "01_audit_and_align", "align", body)

"""What every Module 2 lab needs: the unsolved marker and the data sources.

The vehicle telemetry is real. The phone traces are generated, because the real
ones are the position records of sixteen identifiable people. Which is which is
never left to guesswork -- `load_bus()` says archive, `load_phones()` says
generated, and the module's slides carry the same tag on every number.

The generated tables are written once by `data/prepare.py` (which `setup.sh`
runs) and read from `data/*.parquet` here, so that a lab never depends on the
generator running at import time and check 0 can verify the files before any
grading starts. When a Parquet file is missing the loader falls back to the
generator and says so once, on the standard error stream: the answer is the same
table, only slower, and the fix is `bash setup.sh`.
"""
from __future__ import annotations

import json
import logging
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
BUS_SLICE = DATA / "bus_slice.csv.gz"
MODULE1_PROFILE = DATA / "module1_profile.json"
sys.path.insert(0, str(DATA))

_log = logging.getLogger("lab_support")
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    _log.addHandler(_handler)
    _log.propagate = False
_warned: set[str] = set()


class NotSolved(Exception):
    """A lab stub raises this. The check turns it into exit code 2.

    It is not an error. It means "you have not written this yet", which is a
    different state from "you wrote it and it is wrong", and the checks say so.
    """


class EnvironmentNotReady(Exception):
    """The tools or the data this module needs are missing. The checks exit 3.

    A third state, separate from the other two, because "this machine is not set
    up" is not "your code is wrong". A student told the second while the first is
    true goes hunting for a bug that was never there. Every check imports it from
    here, so the modules share one class rather than several with the same name.
    """


def _fallback_warning(name: str) -> None:
    """One WARNING per missing table per process, not one per call."""
    if name not in _warned:
        _warned.add(name)
        _log.warning("data/%s is not there -- generating it in memory instead. "
                     "Run  bash setup.sh  once to write it.", name)


def load_bus() -> pd.DataFrame:
    """Real vehicle telemetry -- shuttle VJRD1A10224000055, 22-23 January 2020.

    The archive. It identifies nobody, which is why you have it. Readings arrive
    about twice a second. Ships with the repository, so it needs no preparing.
    """
    return pd.read_csv(BUS_SLICE, low_memory=False)


def load_phones(day: str = "2020-01-22", with_truth: bool = False) -> pd.DataFrame:
    """Generated phone traces, calibrated from the archive.

    Every calibrated parameter -- sampling rate, the absence rate of each beacon
    per aboard state, the aboard balance -- was measured from the real
    passengers.csv and is recorded in data/calibration.json. The magnitudes are
    real; the people are not.

    Readings arrive about once a second, against the vehicle's twice a second.
    That gap is Lab 1. `with_truth=True` returns the hidden columns as well:
    true signal, true distance to each beacon in metres, the aboard state and
    the shuttle ridden. The graded functions never reach for them -- the check
    hands the truth to `imputation_bias` as an argument -- but you may open the
    frame yourself while you work, which is how Lab 2's own demonstration
    measures what each fill invented.
    """
    name = f"phones_truth_{day}.parquet" if with_truth else f"phones_{day}.parquet"
    path = DATA / name
    if path.exists():
        return pd.read_parquet(path)
    _fallback_warning(name)
    from make_phones import generate
    return generate(day=day, with_truth=with_truth)


def load_simpson() -> pd.DataFrame:
    """Both generated days in one table: phone, time, day, shuttle ridden, aboard.

    The frame Lab 1's `pooled_versus_by_group` reads. The aboard share moves one
    way on each shuttle and the other way when the shuttles are pooled; the
    generator (data/make_phones.py, RIDERS) plants that on purpose, and the
    check grades whether you noticed.
    """
    path = DATA / "simpson.parquet"
    if path.exists():
        return pd.read_parquet(path)
    _fallback_warning("simpson.parquet")
    from make_phones import generate
    from prepare import simpson_frame
    return simpson_frame(generate)


# --------------------------------------------------------------------------
# What Module 1 hands over
# --------------------------------------------------------------------------
# Module 1's Lab 3 writes two things: DATA_PROFILE.md, for a person, and
# out/data_profile.json, for a program. The second is what this module consumes.
# The copy under data/module1_profile.json is byte-for-byte what Module 1's own
# `declare_profile()` produced from this same slice -- MODULE 1 IS ITS SOURCE,
# and `python3 solutions/lab_03.py` in Module 1's exercises regenerates it.
#
# It is shipped here rather than read out of `Module 1/exercises/out/` on
# purpose. A check that reaches into a sibling module's output directory is
# green only on a machine where that module has already been run, and red for
# reasons that have nothing to do with the student in front of it. What the live
# dependency was really buying is the guarantee that the fixture still describes
# the data -- and `data/prepare.py` buys that directly, on every setup, by
# running the declaration back over the slice and refusing to prepare if it no
# longer holds.
#
# `check_against` below is Module 1's function, copied. It is not re-derived and
# it is not improved: two modules that validate the same table by two slightly
# different rules is precisely the failure this hand-off exists to prevent.


def load_module1_profile() -> dict:
    """Module 1's machine-readable profile of the vehicle slice this module ships.

    A data dictionary with rules in it: per column a declared type, unit, range
    and tolerated absence, plus the time column and the sampling step the source
    is expected to report at. Schema `aau-ce3/data-profile/1`.
    """
    if not MODULE1_PROFILE.exists():
        raise EnvironmentNotReady(
            f"{MODULE1_PROFILE.name} is not in data/. It is Module 1's profile of "
            "the slice this module aligns, and Lab 1 runs the incoming frame "
            "against it before anything else. Run  bash setup.sh")
    return json.loads(MODULE1_PROFILE.read_text())


def check_against(frame: pd.DataFrame, profile: dict) -> list:
    """Apply Module 1's declaration to a frame. An empty list means it holds.

    Copied unchanged from `Module 1/exercises/solutions/lab_03.py`, where it is
    graded. Presence, then type, then range, then missing share, then the
    sampling step; each complaint begins with the field it concerns. The order
    matters: a numeric column that arrived as text would raise rather than
    report if compared with a number, so the range rule stays silent exactly
    where the type rule has already spoken.
    """
    breaches: list = []
    columns = profile.get("columns", {})
    typed = {}

    for name, rule in columns.items():
        if name not in frame.columns:
            breaches.append(f"{name}: declared in the profile and not in this frame.")
            continue
        numeric = pd.api.types.is_numeric_dtype(frame[name])
        typed[name] = (rule.get("type") == "number") == numeric
        if not typed[name]:
            arrived = "number" if numeric else "text"
            breaches.append(f"{name}: declared as {rule.get('type')}, arrived as "
                            f"{arrived} (stored as {frame[name].dtype}).")

    for name, rule in columns.items():
        if not typed.get(name) or rule.get("type") != "number":
            continue
        values = frame[name]
        if rule.get("minimum") is not None:
            below = int((values < rule["minimum"]).sum())
            if below:
                breaches.append(f"{name}: {below} row(s) below the declared minimum "
                                f"{rule['minimum']}.")
        if rule.get("maximum") is not None:
            above = int((values > rule["maximum"]).sum())
            if above:
                breaches.append(f"{name}: {above} row(s) above the declared maximum "
                                f"{rule['maximum']}.")

    for name, rule in columns.items():
        if name not in frame.columns or rule.get("max_missing_share") is None:
            continue
        absent = 1.0 - float(frame[name].notna().mean())
        if absent > float(rule["max_missing_share"]):
            breaches.append(f"{name}: absent on {absent:.4f} of rows, above the declared "
                            f"{rule['max_missing_share']}.")

    time_column = profile.get("time_column")
    expected = profile.get("expected_step_seconds")
    if time_column in frame.columns and expected is not None:
        stamps = pd.to_datetime(frame[time_column], errors="coerce").sort_values()
        steps = stamps.groupby(stamps.dt.date).diff().dt.total_seconds().dropna()
        if len(steps):
            median = float(steps.median())
            tolerance = float(profile.get("step_tolerance_share", 0.0)) * float(expected)
            if abs(median - float(expected)) > tolerance:
                breaches.append(
                    f"{time_column}: median step {median:.3f} s within a day, declared "
                    f"{expected} s with {tolerance:.3f} s of tolerance.")
    return breaches

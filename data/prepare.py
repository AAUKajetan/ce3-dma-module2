#!/usr/bin/env python3
"""Prepare every dataset and artefact the labs read, and write data/MANIFEST.json.

    python3 data/prepare.py          called by setup.sh; safe to re-run

Generated data is generated here, once, deterministically (seed 20200122), and
written to data/ as Parquet, so that (a) a lab never depends on a generator
running at import time, (b) `make check` can verify the data is present before
grading anything (verify/check_00_data.py), and (c) a teacher can open the file
a student worked on. Shipped data (the archive slice) is verified, not rewritten.

Module 2 prepares, per day, the student's view of the phones and the truth-kept
view beside it (true signal strength, true distance, the aboard state, the
shuttle ridden), and one more table across both days: `simpson.parquet`, the
frame in which the aboard share moves one way on each shuttle and the other way
pooled. Lab 1 grades that reversal; Lab 2's check and figure read the truth.

The manifest records, per file, the row count, the column count and a hash of
the values -- not of the bytes, because Parquet metadata carries a writer string.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent          # exercises/data
EXERCISES = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXERCISES))

MANIFEST = HERE / "MANIFEST.json"
SLICE = HERE / "bus_slice.csv.gz"

# Module 1's machine-readable data profile of the same slice, shipped with this
# module rather than read out of Module 1's output directory. The file is
# byte-for-byte what Module 1's own `declare_profile()` wrote to
# `Module 1/exercises/out/data_profile.json` from this slice -- MODULE 1 IS ITS
# SOURCE -- and `python3 solutions/lab_03.py` there regenerates it.
#
# A check that reaches into a sibling module's `out/` is green only on a machine
# where that module has been run, and red for reasons that have nothing to do
# with the student's work. So the profile is a fixture here, and prepare.py's job
# is to prove that the fixture still describes the slice this module ships --
# which is the part a live dependency was ever really buying.
MODULE1_PROFILE = HERE / "module1_profile.json"
PROFILE_SCHEMA = "aau-ce3/data-profile/1"

# The columns of simpson.parquet, in the order the lab sees them.
SIMPSON_COLUMNS = ["phone_id", "timestamp_utc", "day", "bus", "aboard"]


def content_hash(frame: pd.DataFrame) -> str:
    return f"{int(pd.util.hash_pandas_object(frame, index=False).sum()):x}"


def table_entry(path: pathlib.Path, frame: pd.DataFrame, source: str, note: str) -> dict:
    return {"path": str(path.relative_to(EXERCISES)), "kind": "table", "source": source,
            "rows": int(len(frame)), "columns": int(frame.shape[1]),
            "content_hash": content_hash(frame), "note": note}


def check_module1_profile(frame: pd.DataFrame) -> str | None:
    """Prove the shipped declaration still holds on the shipped slice.

    Returns None when it does, and the disagreement as a sentence when it does
    not. A fixture that has drifted from the data it claims to describe is worse
    than no fixture, because every check built on it goes on passing.

    The test is the declaration's own: run Module 1's `check_against` over the
    slice and require silence. That is stronger than comparing a stored number
    with a recomputed one, because it exercises every rule in the file -- the
    types, the ranges, the tolerated absences and the reporting rate -- with the
    same code the labs and the checks use.
    """
    if not MODULE1_PROFILE.exists():
        return f"{MODULE1_PROFILE.name} is missing from the checkout"
    from lab_support import check_against
    shipped = json.loads(MODULE1_PROFILE.read_text())
    if shipped.get("schema") != PROFILE_SCHEMA:
        return (f"{MODULE1_PROFILE.name} declares schema {shipped.get('schema')!r}; "
                f"this module consumes {PROFILE_SCHEMA!r}")
    if shipped.get("rows") != int(len(frame)):
        return (f"{MODULE1_PROFILE.name} profiles {shipped.get('rows')!r} rows; the "
                f"slice in this checkout has {len(frame)}")
    breaches = check_against(frame, shipped)
    if breaches:
        return (f"{MODULE1_PROFILE.name} no longer holds on {SLICE.name}: "
                + "; ".join(breaches))
    return None


def simpson_frame(generate) -> pd.DataFrame:
    """Both days, one row per reading: who, when, which day, which shuttle, aboard.

    Built from the truth-kept frames rather than from the labels, so that the
    outcome is a boolean the student did not have to derive. The reversal is
    the generator's (data/make_phones.py, RIDERS); this only lays it out.
    """
    from make_phones import RIDERS
    parts = []
    for day in RIDERS:
        truth = generate(day=day, with_truth=True)
        parts.append(pd.DataFrame({
            "phone_id": truth["phone_id"], "timestamp_utc": truth["timestamp_utc"],
            "day": day, "bus": truth["bus"], "aboard": truth["aboard_truth"].astype(bool)}))
    return pd.concat(parts, ignore_index=True)[SIMPSON_COLUMNS]


def main() -> int:
    files = []

    # 1. The archive slice ships with every module; verify it, never rewrite it.
    if not SLICE.exists():
        print(f"prepare failed: {SLICE.relative_to(EXERCISES)} is missing from the checkout")
        return 1
    bus = pd.read_csv(SLICE, low_memory=False)
    files.append(table_entry(SLICE, bus, "archive",
                             "vehicle VJRD1A10224000055, 22-23 January 2020, as shipped"))
    print(f"verified  {SLICE.name}: {len(bus)} rows x {bus.shape[1]} columns")

    # 1b. Module 1's profile of that slice, shipped and verified, never rewritten.
    drift = check_module1_profile(bus)
    if drift is not None:
        print("prepare failed: the shipped data profile no longer describes the shipped "
              "slice --")
        print(f"  {drift}")
        return 1
    files.append({"path": str(MODULE1_PROFILE.relative_to(EXERCISES)), "kind": "artefact",
                  "source": "archive", "bytes": MODULE1_PROFILE.stat().st_size,
                  "note": "Module 1's machine-readable data profile of "
                          "data/bus_slice.csv.gz (schema aau-ce3/data-profile/1), shipped "
                          "as a fixture and re-run over the slice on every prepare; Lab 1 "
                          "runs the incoming frame against it and its check grades what "
                          "came back"})
    print(f"verified  {MODULE1_PROFILE.name}: the declaration still holds on {SLICE.name}")

    # 2. Generated phone traces, the student's view and the truth-kept view.
    if (HERE / "make_phones.py").exists():
        from make_phones import generate, CALIBRATION
        for day in CALIBRATION["phones_per_day"]:
            frame = generate(day=day, with_truth=False)
            out = HERE / f"phones_{day}.parquet"
            frame.to_parquet(out, index=False)
            files.append(table_entry(out, frame, "generated",
                                     f"make_phones.generate(day={day!r}), seed 20200122"))
            print(f"generated {out.name}: {len(frame)} rows x {frame.shape[1]} columns")

            truth = generate(day=day, with_truth=True)
            out = HERE / f"phones_truth_{day}.parquet"
            truth.to_parquet(out, index=False)
            files.append(table_entry(
                out, truth, "generated",
                f"make_phones.generate(day={day!r}, with_truth=True), seed 20200122 -- "
                "true signal, true distance, aboard state and shuttle kept; the check "
                "and the figures read this, the student's loader does not hand it over"))
            print(f"generated {out.name}: {len(truth)} rows x {truth.shape[1]} columns")

        # 3. The planted reversal, both days in one table.
        simpson = simpson_frame(generate)
        out = HERE / "simpson.parquet"
        simpson.to_parquet(out, index=False)
        files.append(table_entry(
            out, simpson, "generated",
            "both days of make_phones.generate(with_truth=True), seed 20200122: "
            "phone, time, day, shuttle ridden, aboard -- the aboard share rises on "
            "each shuttle and falls pooled (Simpson's paradox, graded by Lab 1)"))
        print(f"generated {out.name}: {len(simpson)} rows x {simpson.shape[1]} columns")

    # 4. The generated stream, where the module has a world.
    if (EXERCISES / "service" / "world.py").exists():
        from service import world
        frame = world.stream()
        out = HERE / "stream.parquet"
        frame.to_parquet(out, index=False)
        files.append(table_entry(out, frame, "generated",
                                 "service.world.stream(), 28 days, seed 20200122"))
        print(f"generated {out.name}: {len(frame)} rows x {frame.shape[1]} columns")

    # 5. Trained artefacts, where the module serves a model.
    if (EXERCISES / "service" / "models.py").exists():
        result = subprocess.run([sys.executable, str(EXERCISES / "service" / "models.py")],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("prepare failed: service/models.py did not train --")
            print(result.stdout[-1500:], result.stderr[-1500:])
            return 1
        artefacts = EXERCISES / "service" / "artefacts"
        for path in sorted(artefacts.glob("*")):
            if path.is_file():
                files.append({"path": str(path.relative_to(EXERCISES)), "kind": "artefact",
                              "source": "generated", "bytes": path.stat().st_size,
                              "note": "written by service/models.py"})
        print(f"trained   {len([f for f in files if f['kind'] == 'artefact'])} artefact(s) "
              f"in service/artefacts/")

    MANIFEST.write_text(json.dumps({"seed": 20200122, "files": files}, indent=1))
    print(f"wrote     data/MANIFEST.json -- {len(files)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate phone traces with the archive's shape and none of its people.

    python3 data/make_phones.py

The real `passengers.csv` is the position trace of sixteen identifiable
volunteers. Under Article 4 of the General Data Protection Regulation that is
personal data, and it is not in this repository and will not be.

What is here instead is a generator whose every calibrated parameter was
measured from the real file by `Module 2/slides/measure.py` and written into
`calibration.json`. The magnitudes ship; the rows never do. A generator with
invented parameters would be a fabricated experimental result, so nothing below
is invented — where a calibrated number appears it came from the archive, and
`calibration.json` says so. The one thing that is a construction rather than a
measurement, the rider mix in RIDERS below, says so where it is written.

What is faithfully reproduced, because the labs turn on it:

  * two sampling rates -- phones about once a second, vehicles twice a second,
    which is the alignment problem in Lab 1
  * beacons absent at the measured rates, and absent *for a reason*: out of
    range, not lost in transit. That is a mechanism, not noise (Lab 2)
  * the same absence encoded twice -- an empty signal strength and a proximity
    of -1 mark exactly the same rows, as they do in the archive
  * a `bus_id` present exactly when the passenger is aboard, which is the leak
    Lab 3 hunts. In the archive this is real: 7,723 rows aboard all carry it,
    6,000 rows not aboard all lack it, with no exceptions
  * two shuttles, "Bus 1" and "Bus 2", as the archive's own labels name them,
    and a second day with fewer volunteers and a different mix of riders -- so
    that the aboard share moves one way on each shuttle and the other way when
    the two are pooled. That reversal by composition is Simpson's paradox, and
    it is planted here on purpose (Lab 1)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CALIBRATION = json.loads((HERE / "calibration.json").read_text())

SEED = 20200122
BEACONS = ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]
PROXIMITY = {"rssiA": "proxA", "rssiB": "proxB", "rssiC": "proxC",
             "rssi1": "prox1", "rssi2": "prox2"}

# A beacon is heard when it is near. Signal strength falls with distance, so the
# absence is a consequence of geometry rather than of a dropped packet -- which
# is exactly what makes it missing-not-at-random.
RANGE_METRES = 25.0
STRENGTH_AT_ONE_METRE = -45.0
PATH_LOSS = 2.4

PER_PHONE = 900  # fifteen minutes each, enough for windows without being slow

# Who rode which shuttle, and for how long, in readings out of PER_PHONE.
#
# This table is a construction, not a measurement, and it is the only one in
# the generator. It is built so that the calibrated magnitude is untouched: on
# the first day 4 x 361 + 8 x 580 = 6,084 aboard readings out of 10,800, which
# is the 56.3 per cent of labelled rows aboard that measure.py found -- and the
# same 6,084 the generator produced before this table existed. Every archive
# label is on the first day, so the second day's aboard share was never
# calibrated; here it is chosen so that the two-day comparison reverses by
# composition: the aboard share rises on each shuttle (Bus 1: 361 -> 405 of
# 900; Bus 2: 580 -> 630 of 900) and falls when the two are pooled, because the
# second day has fewer volunteers and most of them rode the shuttle whose
# riders spend the smaller share of their time aboard. Same rows, opposite
# conclusion, and the sign flip is what Lab 1 grades.
RIDERS = {
    "2020-01-22": {"Bus 1": {"phones": 4, "ride": 361},
                   "Bus 2": {"phones": 8, "ride": 580}},
    "2020-01-23": {"Bus 1": {"phones": 6, "ride": 405},
                   "Bus 2": {"phones": 2, "ride": 630}},
}


def _riders_for(day: str) -> list[tuple[str, int]]:
    """(shuttle, ride length) for every phone of the day, calibrated count kept."""
    plan = RIDERS[day]
    riders = [(bus, spec["ride"]) for bus, spec in plan.items()
              for _ in range(spec["phones"])]
    calibrated = CALIBRATION["phones_per_day"][day]
    assert len(riders) == calibrated, (
        f"RIDERS lists {len(riders)} phones on {day}; calibration.json measured "
        f"{calibrated} -- the mix may change, the count may not")
    return riders


def generate(day: str = "2020-01-22", seed: int = SEED,
             with_truth: bool = False) -> pd.DataFrame:
    """One day of phone traces. Set with_truth to keep the hidden columns.

    Hidden columns, kept only when with_truth is True: `<beacon>_true`, the
    signal strength every reading would have had, heard or not, in
    decibel-milliwatts; `<beacon>_distance_true`, the distance to that beacon in
    metres; `aboard_truth`, the aboard state as a boolean; and `bus`, the
    shuttle this volunteer rode that day. The student's view carries none of
    them, because in the archive nobody has them either.
    """
    rng = np.random.default_rng(seed)
    interval = CALIBRATION["phone_interval_s"]
    riders = _riders_for(day)
    start = pd.Timestamp(f"{day} 09:00:00", tz="UTC")

    frames = []
    for phone, (bus, ride) in enumerate(riders):
        clock = start + pd.to_timedelta(np.arange(PER_PHONE) * interval, unit="s")
        # Nanosecond resolution, matching what pd.to_datetime gives on the
        # vehicle file. Mismatched resolutions make merge_asof refuse, which is
        # a real nuisance but not the lesson of any lab here.
        clock = clock.astype("datetime64[ns, UTC]")

        # A journey: waiting at a stop, riding, then waiting again. The rider is
        # aboard for `ride` consecutive readings, placed at random in the day.
        boarding = int(rng.integers(0, max(1, PER_PHONE - ride)))
        position = np.arange(PER_PHONE)
        is_aboard = (position >= boarding) & (position < boarding + ride)

        # Distance to each beacon. Note what this does NOT do: tie hearing a
        # vehicle beacon to being aboard. The archive says that link is not
        # there -- "this beacon was heard" agrees with "aboard" on 42 to 48 per
        # cent of labelled rows, against a base rate of 56.3. The whole trial
        # fits in a box about 67 by 86 metres (Module 1), and beacon range is
        # tens of metres, so being near a stop beacon and being on the vehicle
        # are not separable by proximity here.
        #
        # So closeness is its own process, and each beacon is heard at the rate
        # the archive measured for each state. What survives, because it is real,
        # is that the readings which ARE absent are the far and weak ones.
        distances = {}
        for name in BEACONS:
            wander = rng.normal(0, 1.0, PER_PHONE).cumsum()
            wander = (wander - wander.min()) / (np.ptp(wander) or 1.0)
            distances[name] = 1.0 + wander * 95.0

        frame = pd.DataFrame({
            "phone_id": f"p{phone:02d}",
            "timestamp_utc": clock,
            # The same trap as the archive: a column named `timestamp` that is
            # local time, sitting beside the one that is not.
            "timestamp": clock.tz_convert("Europe/Copenhagen").tz_localize(None),
            "speed": np.where(is_aboard, rng.uniform(0, 3.5, PER_PHONE),
                              rng.uniform(0, 1.4, PER_PHONE)).round(3),
            "stationary": (~is_aboard).astype(int),
        })

        for name in BEACONS:
            distance = np.clip(distances[name], 0.5, None)
            true_strength = (STRENGTH_AT_ONE_METRE
                             - 10 * PATH_LOSS * np.log10(distance)
                             + rng.normal(0, 2.0, PER_PHONE))

            # Heard at the archive's measured rate, separately for aboard and
            # not-aboard rows, and within each state the strongest are heard.
            # Both the marginal absence rate and the (weak) association with the
            # label then match the real file rather than a story about it.
            heard = np.zeros(PER_PHONE, dtype=bool)
            for state, share_table in ((True, CALIBRATION["beacon_heard_when_aboard"]),
                                       (False, CALIBRATION["beacon_heard_when_not_aboard"])):
                rows = is_aboard if state else ~is_aboard
                count = int(rows.sum())
                if count == 0:
                    continue
                keep = int(round(share_table[name] / 100 * count))
                if keep == 0:
                    continue
                cutoff = np.sort(true_strength[rows])[::-1][keep - 1]
                heard |= rows & (true_strength >= cutoff)

            frame[f"{name}_true"] = true_strength.round(1)
            frame[f"{name}_distance_true"] = distance.round(1)
            frame[name] = np.where(heard, true_strength.round(1), np.nan)
            # The same absence, encoded a second way: -1 rather than empty.
            bands = np.select(
                [true_strength > -60, true_strength > -75], [1, 2], default=3)
            frame[PROXIMITY[name]] = np.where(heard, bands, -1)

        # The archive's own label vocabulary: the shuttle's name while aboard,
        # the stop while waiting.
        frame["label"] = np.where(is_aboard, bus, "Stop C")
        frame["label2"] = np.where(is_aboard, "IN", "OUT")
        # The leak. Present exactly when aboard, as in the archive.
        frame["bus_id"] = np.where(is_aboard, "VJRD1A10224000055", None)
        frame["aboard_truth"] = is_aboard
        frame["bus"] = bus
        frames.append(frame)

    everything = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp_utc", "phone_id"]).reset_index(drop=True)

    if with_truth:
        return everything
    hidden = ([c for c in everything.columns if c.endswith("_true")]
              + ["aboard_truth", "bus"])
    return everything.drop(columns=hidden)


def simpson_table(seed: int = SEED) -> pd.DataFrame:
    """The 2 x 2 x 2 table the reversal lives in: aboard share by day and shuttle.

    One row per (day, shuttle) plus one pooled row per day; columns `readings`,
    `aboard`, `share`. Measured from the generated frames, never typed, so the
    slide and the check read the same numbers.
    """
    rows = []
    for day in RIDERS:
        frame = generate(day=day, seed=seed, with_truth=True)
        for bus, part in frame.groupby("bus"):
            rows.append({"day": day, "group": bus, "readings": int(len(part)),
                         "aboard": int(part["aboard_truth"].sum())})
        rows.append({"day": day, "group": "pooled", "readings": int(len(frame)),
                     "aboard": int(frame["aboard_truth"].sum())})
    table = pd.DataFrame(rows)
    table["share"] = (table["aboard"] / table["readings"] * 100).round(1)
    return table


if __name__ == "__main__":
    phones = generate()
    print(f"{len(phones):,} rows, {phones.shape[1]} columns, "
          f"{phones['phone_id'].nunique()} phones")
    print("\nabsent share, generated against the archive's measured share:")
    for name in BEACONS:
        measured = CALIBRATION["beacon_absent_share"][name]
        print(f"  {name:7} generated {phones[name].isna().mean() * 100:5.1f} %"
              f"   archive {measured:5.1f} %")
    same = all((phones[r].isna() == (phones[p] == -1)).all() for r, p in PROXIMITY.items())
    print(f"\nabsence encoded twice, and the two agree everywhere: {same}")
    aboard = (phones["label2"] == "IN").mean() * 100
    print(f"\naboard share, first day, generated {aboard:.1f} %   "
          f"archive {CALIBRATION['aboard_share_of_labelled']:.1f} %")
    print("\naboard share by day and shuttle -- the planted reversal:")
    print(simpson_table().to_string(index=False))

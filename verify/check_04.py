#!/usr/bin/env python3
"""Check 4 — the window is the window, the split is by time, the join is measured."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready, explain                   # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from lab_support import load_bus, load_phones                     # noqa: E402
    import numpy as np                                                # noqa: E402
    import pandas as pd                                               # noqa: E402
except ImportError as unready:
    not_ready(unready)

WIDTH = 60
TOLERANCES = (1, 2, 5, 10, 30)
SEED = 20200122
GRAIN = "5s"
TARGET_COLUMN = "aboard"
HANDOFF_SCHEMA = "1.0"
HANDOFF_KEYS = ("schema_version", "rows", "columns", "key", "feature_columns",
                "mask_columns", "target", "split_point", "train_rows", "test_rows",
                "transform", "ledger")

# A planted window whose six features can be worked out on paper: mean 3,
# population standard deviation sqrt(2), minimum 1, maximum 5, slope 1 per
# position, and r_1 = (2 + 0 + 0 + 2) / 10 = 0.4. A lookup table copied off a
# slide, or a constant, meets none of them.
PLANTED = [1.0, 2.0, 3.0, 4.0, 5.0]
PLANTED_FEATURES = {"mean": 3.0, "std": np.sqrt(2.0), "minimum": 1.0, "maximum": 5.0,
                    "slope": 1.0, "autocorrelation_1": 0.4}


def sample_autocorrelation(values, lag: int = 1) -> float:
    """r_k = sum_(t=k+1)^n (x_t - xbar)(x_(t-k) - xbar) / sum_(t=1)^n (x_t - xbar)^2.

    The Box-Jenkins sample autocorrelation (Box, Jenkins, Reinsel & Ljung, 2015,
    section 2.1.4), which is the estimator this course defined in Module 1 and
    teaches here. It is deliberately NOT pandas.Series.autocorr: that function
    returns the Pearson correlation between the series and its shifted copy, with
    two means and two variances, and the two disagree on every finite window. The
    check grades the course's definition so that Module 1's number and this one
    are the same quantity.
    """
    x = np.asarray(values, dtype=float)
    centred = x - np.nanmean(x)
    later, earlier = centred[lag:], centred[:-lag]
    both = ~np.isnan(later) & ~np.isnan(earlier)
    denominator = float(np.nansum(centred ** 2))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(later[both] * earlier[both]) / denominator)


def reference_growth(phones, bus) -> dict:
    """The check's own nearest-match-within-tolerance, to compare against."""
    stamp = "datetime64[ns, UTC]"
    left = pd.DataFrame(
        {"_t": pd.to_datetime(phones["timestamp_utc"], utc=True).astype(stamp)}
    ).dropna().sort_values("_t")
    right = pd.DataFrame(
        {"_t": pd.to_datetime(bus["utc_time"], utc=True).astype(stamp),
         "bus_speed": bus["speed"]}).sort_values("_t")

    growth = {}
    for seconds in TOLERANCES:
        matched = pd.merge_asof(left, right, on="_t", direction="nearest",
                                tolerance=pd.Timedelta(seconds=seconds))
        growth[seconds] = float(matched["bus_speed"].notna().mean()) * 100
    return growth


def handoff_inputs():
    """What Labs 1 and 3 hand to Lab 4, built here so that Lab 4 stands alone.

    The check computes its own aligned frame and its own fitted transform rather
    than importing the earlier labs: a student who has not finished Lab 1 should
    fail Lab 1, not Lab 4. The frame is deliberately small and its gaps are
    deliberately uneven -- `rssi1` is absent on most rows, `phone_speed` on none
    -- because the mask rule is a measurement and not a convention, and a frame
    where every column looks the same cannot tell the two apart.
    """
    phones = load_phones().copy()
    when = pd.to_datetime(phones["timestamp_utc"], utc=True)
    phones["window"] = when.dt.floor(GRAIN)
    aligned = phones.groupby(["phone_id", "window"], as_index=False).agg(
        phone_speed=("speed", "mean"),
        rssi1=("rssi1", "mean"),
        rssi2=("rssi2", "mean"),
        aboard=("label2", lambda values: float((values == "IN").mean())))

    # Shuffled, because a hand-off built by a function that assumes its input
    # already arrives in time order is a hand-off that breaks the first time
    # somebody hands it a table straight out of a database.
    aligned = aligned.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    columns = ["phone_speed", "rssi1", "rssi2"]
    cut = int(len(aligned) * 0.7)
    train = aligned.sort_values("window").iloc[:cut]
    fitted = {
        "medians": {c: float(train[c].median()) for c in columns},
        "means": {c: float(train[c].mean()) for c in columns},
        "stds": {c: float(train[c].std(ddof=0)) or 1.0 for c in columns},
        "columns": columns,
    }
    ledger = {"grain_seconds": 5, "bus_rows_in": 48290, "phone_rows_in": len(phones),
              "windows_out": len(aligned), "phone_rows_used": len(phones),
              "phone_rows_dropped": 0, "drop_reasons": {}}
    return aligned, fitted, ledger


def grade_handoff(lab):
    """The object the next three modules open, graded against the slide's promise."""
    aligned, fitted, ledger = handoff_inputs()
    table, manifest = lab.assemble(aligned, fitted, ledger, TARGET_COLUMN, 0.7)

    for key in HANDOFF_KEYS:
        assert key in manifest, (
            f"the hand-off manifest has no {key!r}. Module 3 opens this file and reads "
            f"{', '.join(HANDOFF_KEYS)}; a manifest missing one of them is a table "
            "nobody downstream can use without asking you what you did.")
    assert manifest["schema_version"] == HANDOFF_SCHEMA, (
        f"the manifest says schema version {manifest['schema_version']!r} and Module 3 "
        f"is written for {HANDOFF_SCHEMA!r}.")

    # 1. One row per phone per window. Not most rows -- one.
    assert list(manifest["key"]) == ["phone_id", "window"], (
        f"the manifest says the key is {manifest['key']}; the grain of this table is "
        "one row per phone per window.")
    assert not table.duplicated(subset=["phone_id", "window"]).any(), (
        "the hand-off table holds more than one row for the same phone in the same "
        "window. Every count Modules 3, 4 and 5 take off this table is wrong by "
        "however many duplicates there are, and none of them will notice.")
    assert len(table) == len(aligned), (
        f"the hand-off table has {len(table):,} rows and it was given {len(aligned):,}. "
        "Rows are not dropped here — every drop was decided, counted and explained in "
        "Lab 1, and this step has nothing left to decide.")
    assert manifest["rows"] == len(table), (
        f"the manifest says {manifest['rows']:,} rows and the table has {len(table):,}. "
        "A manifest that disagrees with its own table is worse than none.")

    # 2. The mask beside every filled value -- and beside nothing else.
    for column in fitted["columns"]:
        assert column in table.columns, (
            f"the hand-off table has no {column!r}, which the stored transform names.")
        assert not table[column].isna().any(), (
            f"{column!r} still has gaps in the hand-off table. The point of carrying the "
            "fitted transform this far is that the fill happens here, once, with the "
            "stored constant.")
        absent = aligned.set_index(["phone_id", "window"])[column].isna()
        name = f"{column}_missing"
        if bool(absent.any()):
            assert name in table.columns, explain(
                f"handoff:mask:{column}",
                f"{column!r} was filled on {int(absent.sum()):,} of {len(aligned):,} rows "
                f"and the table carries no {name!r} beside it",
                "A filled value and a measured value are the same number in a table. The "
                "mask is the only record of which is which, and once it is left behind "
                "the difference cannot be recovered by anybody downstream — not by "
                "Module 3 serving it, not by Module 4 looking for drift in it.")
            assert name in manifest["mask_columns"], (
                f"{name!r} is in the table and not in the manifest's mask_columns. "
                "Module 3 reads that list to know which columns are masks.")
            got = table.set_index(["phone_id", "window"])[name]
            expected = absent.reindex(got.index).astype(int)
            assert (got.astype(int) == expected).all(), (
                f"{name!r} does not mark the rows that were actually absent. It is 1 "
                "where the value was missing before the fill and 0 where it was "
                "measured, and nothing else.")
        else:
            assert name not in table.columns, explain(
                f"handoff:mask:none:{column}",
                f"nothing was ever filled in {column!r} and the table carries a "
                f"{name!r} column anyway",
                "It would hold nought on every row. A constant column is noise that "
                "costs memory in every model that reads it and tells nobody anything, "
                "and writing one is how a schema fills up with columns no one dares "
                "remove. Look at the column before you write its mask.")

    # 2b. The values are the stored constants applied, and only the stored ones.
    #     Until this existed, a table that carried the raw column through
    #     untouched passed every assertion above: the gaps were checked, the
    #     masks were checked, and nothing looked at a number. It is also where
    #     "fitted on the training rows only" stops being a claim about the input
    #     and becomes a property of the object handed over.
    indexed = aligned.set_index(["phone_id", "window"])
    got_frame = table.set_index(["phone_id", "window"])
    for column in fitted["columns"]:
        source = indexed[column]
        stored = ((source.fillna(fitted["medians"][column]) - fitted["means"][column])
                  / fitted["stds"][column])
        naive = ((source.fillna(source.median()) - source.mean())
                 / (source.std(ddof=0) or 1.0))
        # The fixture has to be able to tell the two apart, or the assertion
        # below is decoration. The training rows are the first 70 per cent in
        # time order, so the two sets of constants differ; say so if they ever
        # stop differing rather than passing quietly.
        assert float((stored - naive).abs().max()) > 1e-3, (
            f"the check cannot distinguish the stored constants from a fit over the "
            f"whole frame on {column!r} — the fixture has lost its teeth; tell the "
            "instructor.")
        got = got_frame[column].reindex(stored.index)
        worst = float((got - stored).abs().max())
        assert worst <= 1e-9, explain(
            f"handoff:applied:{column}",
            f"{column!r} in the hand-off table is not the stored transform applied to "
            f"it — the largest disagreement is {worst:.6g}",
            "Fill with the stored median, subtract the stored mean, divide by the "
            "stored standard deviation. Those constants were estimated on the "
            "training rows and on nothing else, which is the only reason they are "
            "worth carrying this far. Recomputing them from the table in front of "
            "you uses the test rows to prepare the model's input, and a service that "
            "later recomputes them from one live request gets a third answer again — "
            "that is training-serving skew, and Module 3 spends a block causing it "
            "on purpose.")

    # 3. The split point, recorded, and recorded as an instant.
    assert "split" in table.columns, (
        "the hand-off table has no 'split' column. The split has to travel with the "
        "table: a boundary that lives only in the script that made it is a boundary "
        "the next comparison will silently choose differently.")
    assert set(table["split"]) <= {"train", "test"}, (
        f"the split column holds {sorted(set(table['split']))}; it holds 'train' and "
        "'test'.")
    split_point = pd.Timestamp(manifest["split_point"])
    windows = pd.to_datetime(table["window"], utc=True)
    trains = table["split"] == "train"
    assert trains.any() and (~trains).any(), "the split put every row on one side"
    assert windows[trains].max() < split_point, explain(
        "handoff:split_straddle",
        f"the last training window is {windows[trains].max()} and the recorded split "
        f"point is {split_point} — the training half reaches the split point itself",
        "The split point is an instant, and every row at that instant belongs to the "
        "same side of it. A window with some of its phones training and the rest "
        "testing is a random split hiding inside a time split, on data whose "
        "consecutive readings correlate at about 0.997.")
    assert windows[~trains].min() == split_point, explain(
        "handoff:split_point",
        f"the test half starts at {windows[~trains].min()} and the manifest records "
        f"the split point as {split_point}",
        "The recorded instant is the one the test period actually begins at, or the "
        "record is a description of a split somebody else made.")
    assert manifest["train_rows"] == int(trains.sum()), (
        f"the manifest says {manifest['train_rows']:,} training rows and the table has "
        f"{int(trains.sum()):,}.")
    assert manifest["train_rows"] + manifest["test_rows"] == manifest["rows"], (
        "train_rows + test_rows does not equal rows. The same arithmetic as Lab 1's "
        "ledger, on the way out instead of on the way in.")

    # The fraction has to be honoured, to within one window's worth of phones.
    phones_per_window = int(table.groupby("window").size().max())
    assert abs(manifest["train_rows"] - int(len(table) * 0.7)) <= phones_per_window, (
        f"{manifest['train_rows']:,} rows train out of {len(table):,}, which is not the "
        "0.7 you were asked for. Moving the cut to the start of a window may move it by "
        f"at most one window's worth of rows ({phones_per_window} here); this moved it "
        "further.")

    # 4. The transform and the ledger, stored unchanged.
    assert manifest["transform"] == fitted, explain(
        "handoff:transform",
        "the transform in the manifest is not the one that was handed over",
        "The fitted constants and the table are one artefact. Recomputed, rounded or "
        "rebuilt on the way into the manifest, they are constants Module 3's service "
        "will not reproduce, and every answer it gives will be a little wrong in a way "
        "nothing detects. Store what you were given.")
    assert manifest["ledger"] == ledger, (
        "the ledger in the manifest is not the one that was handed over. It is the "
        "answer to 'what did you do to the data', and it is copied, not summarised.")

    # 5. Asked again at a different fraction: a manifest whose split point is a
    #    constant passes everything above and fails here.
    other, other_manifest = lab.assemble(aligned, fitted, ledger, TARGET_COLUMN, 0.5)
    assert other_manifest["train_rows"] < manifest["train_rows"], explain(
        "handoff:fraction",
        f"at a training fraction of 0.5 the manifest still records "
        f"{other_manifest['train_rows']:,} training rows, the same as at 0.7",
        "The fraction is an argument, so the split point is computed from the table "
        "in front of the function rather than remembered from the last time it ran.")
    assert pd.Timestamp(other_manifest["split_point"]) < split_point, (
        "a smaller training fraction has to move the split point earlier, and it did "
        "not move at all.")


def body(lab):
    phones = load_phones()
    speed = phones["speed"].astype(float)

    features = lab.window_features(speed, WIDTH)
    for key in ("mean", "std", "minimum", "maximum", "slope", "autocorrelation_1"):
        assert key in features, f"window_features() returned no '{key}'"

    tail = speed.tail(WIDTH).reset_index(drop=True)
    close(features["mean"], float(tail.mean()), 1e-6, "window_features → mean")
    close(features["std"], float(tail.std(ddof=0)), 1e-6, "window_features → std")
    close(features["minimum"], float(tail.min()), 1e-9, "window_features → minimum")
    close(features["maximum"], float(tail.max()), 1e-9, "window_features → maximum")
    expected_slope = float(np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0])
    close(features["slope"], expected_slope, 1e-6, "window_features → slope")
    close(features["autocorrelation_1"], sample_autocorrelation(tail, 1), 1e-6,
          "window_features → autocorrelation_1, by the Box-Jenkins formula r_k = "
          "sum (x_t - xbar)(x_(t-k) - xbar) / sum (x_t - xbar)^2 — one mean over the "
          "whole window and the whole window's sum of squares underneath. "
          f"pandas.Series.autocorr(1) is a different estimator and reads "
          f"{float(tail.autocorr(1)):.6g} on this window")

    # Exactly the last `width` values. A window that reaches further back is a leak.
    narrow = lab.window_features(speed, 10)
    close(narrow["mean"], float(speed.tail(10).mean()), 1e-6,
          "window_features(series, 10) → mean: use exactly the last 10 values, no more")

    # And the six on a window whose answers are arithmetic, so that a table of
    # numbers copied off the slide has nothing to copy.
    planted = lab.window_features(pd.Series(PLANTED), len(PLANTED))
    for key, expected in PLANTED_FEATURES.items():
        close(planted[key], expected, 1e-9,
              f"window_features([1, 2, 3, 4, 5], 5) → {key}: work it out on paper. "
              "The standard deviation is the population one (ddof = 0) and the "
              "autocorrelation is the Box-Jenkins r_1")

    # The split. Shuffled first, with a fixed seed so the failure is the same
    # failure every time: the shipped table already arrives in time order, so a
    # split that cuts it where it stands and never sorts passes on it, and the
    # one line this exercise is about goes ungraded.
    shuffled = phones.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train, test = lab.split_by_time(shuffled, 0.7)
    assert len(train) + len(test) == len(phones), (
        f"{len(train)} + {len(test)} != {len(phones)}: the split lost or duplicated rows")
    train_time = pd.to_datetime(train["timestamp_utc"], utc=True)
    test_time = pd.to_datetime(test["timestamp_utc"], utc=True)
    assert train_time.max() <= test_time.min(), (
        f"the training set reaches {train_time.max()} and the test set starts "
        f"{test_time.min()} — they overlap in time. That is a random split wearing a "
        "time split's name, and on data whose consecutive readings correlate at 0.997 "
        "it tests the model on paraphrases of its own training data. The rows handed "
        "to you here are out of order, as the archive's own rows are; sort before you "
        "cut.")
    close(len(train), int(len(phones) * 0.7), 1, "training rows at fraction 0.7")

    # The join.
    bus = load_bus()
    growth = lab.join_growth(phones, bus, TOLERANCES)
    assert set(growth) == set(TOLERANCES), f"join_growth returned keys {sorted(growth)}"
    shares = [growth[t] for t in TOLERANCES]
    assert all(b >= a - 1e-9 for a, b in zip(shares, shares[1:])), (
        f"the matched share falls as the tolerance widens: {shares}. A wider window can "
        "only ever match more rows, so something is being recomputed rather than relaxed.")
    assert shares[0] > 0, "nothing matched even at one second — check the time column"
    assert shares[-1] <= 100.0, "more than 100 per cent of rows matched"

    # And the trade the lab exists to measure. On the pair above the vehicle
    # reports twice a second, so every phone row already matches at one second
    # and every tolerance reads 100.0 — which a function returning one constant
    # reproduces exactly. Ask again with the vehicle reporting once every thirty
    # seconds, where the share has to climb from a few per cent to all of them.
    in_time_order = bus.assign(_when=pd.to_datetime(bus["utc_time"], utc=True))
    sparse_bus = in_time_order.sort_values("_when").iloc[::60].drop(columns="_when")

    sparse = lab.join_growth(phones, sparse_bus, TOLERANCES)
    expected_growth = reference_growth(phones, sparse_bus)
    for seconds in TOLERANCES:
        close(sparse[seconds], expected_growth[seconds], 0.2,
              f"join_growth at {seconds} s against a vehicle reporting once every "
              "thirty seconds")
    grade_handoff(lab)

    assert sparse[30] > sparse[1] + 50, (
        f"widening the tolerance from one second to thirty moved the matched share "
        f"from {sparse[1]:.1f} to {sparse[30]:.1f} per cent. Against a vehicle "
        "reporting once every thirty seconds it has to move from a few per cent to "
        "nearly all of them; a share that hardly moves was not measured.")


run(4, "04_windows_and_cost", "window_features", body)

"""Lab 4, solved — with the reasoning, not only the code."""
from __future__ import annotations

import sys
import pathlib

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_bus, load_phones  # noqa: E402,F401
from _narrate import narrator, show_table, save_figure  # noqa: E402

LAB = 4
TOLERANCES = (1, 2, 5, 10, 30)
DDOF = 0                   # population standard deviation, the same choice as Lab 3
TRAIN_FRACTION = 0.7       # the share of rows that train, stated once
TARGET_COLUMN = "aboard"   # what the next three modules are asked to predict
HANDOFF_SCHEMA = "1.0"     # the version Module 3 refuses to read anything but


def sample_autocorrelation(values, lag: int = 1) -> float:
    """r_k = Σ_{t=k+1}^{n} (x_t − x̄)(x_{t−k} − x̄) / Σ_{t=1}^{n} (x_t − x̄)².

    The Box–Jenkins sample autocorrelation (Box, Jenkins, Reinsel & Ljung, 2015,
    section 2.1.4): one mean over the whole window, and the whole window's sum of
    squares in the denominator. This is the estimator Module 1 defined and it is
    not what `pandas.Series.autocorr` computes -- that one is the Pearson
    correlation between the series and its shifted copy, with two means and two
    variances, so the two disagree on every finite window. The course uses one
    definition so that Module 1's 0.997 and this lab's number are the same
    quantity. Pairs where either member is absent are skipped where they stand;
    the mean and the denominator use the values that are present.
    """
    x = np.asarray(values, dtype=float)
    present = ~np.isnan(x)
    if present.sum() < lag + 2:
        return float("nan")
    centred = x - np.nanmean(x)
    later, earlier = centred[lag:], centred[:-lag]
    both = ~np.isnan(later) & ~np.isnan(earlier)
    denominator = float(np.nansum(centred ** 2))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(later[both] * earlier[both]) / denominator)


def window_features(series, width: int) -> dict:
    """Six numbers describing the last `width` readings.

    The slope is the one worth dwelling on. Mean and standard deviation say
    where a signal sits and how much it moves; only the gradient says which way
    it is going, and "decelerating" is usually the operationally interesting
    state. Fitting a straight line against position is the cheapest way to get
    it, and it costs one call.

    Note the strictness about `width`: exactly the last `width` values and no
    more. A window that quietly reaches further back on the first few rows is a
    small leak of the same family as Lab 3 -- information from outside the
    window entering a feature that claims to summarise it. The standard
    deviation is the population one (ddof = 0), stated so that the check and
    this file grade the same number.
    """
    values = pd.Series(series).astype(float).tail(width).reset_index(drop=True)
    positions = np.arange(len(values), dtype=float)

    present = values.notna()
    if present.sum() >= 2:
        slope = float(np.polyfit(positions[present], values[present], 1)[0])
    else:
        slope = float("nan")

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=DDOF)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "slope": slope,
        "autocorrelation_1": sample_autocorrelation(values, 1),
    }


def split_by_time(frame, train_fraction: float = 0.7):
    """Sort by time, then cut. Never shuffle.

    Module 1 measured the reason rather than asserting it: consecutive readings
    on this telemetry correlate at about 0.997. A random split therefore puts a
    reading in the training set and its own neighbour -- very nearly the same
    number -- in the test set, and the model is scored on a paraphrase of what it
    was taught (Roberts et al., 2017; Bergmeir & Benítez, 2012).

    A time split costs you something real: the test set is a different part of
    the day, possibly a different vehicle state, and scores are lower. Those
    lower scores are the honest ones. The cut is at ⌊fraction × rows⌋ after
    sorting by timestamp_utc, so the last training time is at or before the
    first test time and no row is in both.
    """
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def join_growth(phones, bus, tolerances=TOLERANCES) -> dict:
    """What each tolerance buys, measured rather than chosen.

    Widening the tolerance always buys matches and always spends accuracy: the
    vehicle reading you matched is further away in time, so the speed you
    attached to a phone row is less and less the speed at that moment.

    Nearest match within a tolerance: each phone row is paired with the vehicle
    reading closest to it in time, if that reading is within τ seconds, and the
    matched share is the fraction of phone rows that found one. On the archive
    this runs from 95.3 per cent matched at one second to 97.4 at thirty: 2.1
    percentage points for thirty times the tolerance, which is a bad trade, and
    the only way to know that is to measure it.
    """
    # merge_asof refuses to join two time columns of different resolution, and
    # pandas infers the resolution from the text it parsed: the vehicle file has
    # milliseconds and lands on microseconds, the phones on nanoseconds. Force
    # both to the same unit before joining -- a small nuisance, and a real one.
    stamp = "datetime64[ns, UTC]"

    left = phones.copy()
    left["_t"] = pd.to_datetime(left["timestamp_utc"], utc=True, errors="coerce").astype(stamp)
    left = left.dropna(subset=["_t"]).sort_values("_t")

    right = bus.copy()
    right["_t"] = pd.to_datetime(right["utc_time"], utc=True).astype(stamp)
    right = right[["_t", "speed"]].rename(columns={"speed": "bus_speed"}).sort_values("_t")

    growth = {}
    for seconds in tolerances:
        merged = pd.merge_asof(
            left[["_t"]], right, on="_t", direction="nearest",
            tolerance=pd.Timedelta(seconds=seconds))
        growth[seconds] = round(float(merged["bus_speed"].notna().mean()) * 100, 1)
    return growth


def assemble(aligned, fitted: dict, ledger: dict, target: str = TARGET_COLUMN,
                  train_fraction: float = TRAIN_FRACTION):
    """The table this module hands to Module 3, and the manifest that describes it.

    Three things go wrong here, every year, and all three are invisible.

    **The mask is dropped on the way out.** A filled value and a measured value
    look identical in a table; the mask is the only thing that tells them apart,
    and the moment it is left behind the difference is unrecoverable. So the mask
    is written immediately beside the value it belongs to -- not in a separate
    frame, not in a comment -- and only for the columns something was actually
    filled in. A mask of all noughts on a complete column is a column of noise
    that costs memory and buys nothing, so it is not written, and the manifest
    says which masks exist.

    **The split point is remembered as a row number.** Rows are reordered,
    filtered and appended; instants are not. The cut here is the *window* the
    ⌊f·n⌋-th row sits in, so that every row sharing that instant lands on the
    same side of the split. Recorded as text, because a manifest has to survive
    the trip through a file, and a timestamp object does not.

    **The transform travels separately from the table, or not at all.** Then
    Module 3's service prepares a live request differently from the way the
    training rows were prepared, and every number it produces is a little wrong
    in a way nothing detects. It is stored here, unchanged, in the same object.

    The ledger goes in for the same reason all three do: the first question
    anybody asks about a result months later is what was done to the data, and
    the honest answer has to be in the file rather than in somebody's memory.
    """
    table = aligned.sort_values(["window", "phone_id"]).reset_index(drop=True)

    out = pd.DataFrame({"phone_id": table["phone_id"], "window": table["window"]})
    mask_columns = []
    for column in fitted["columns"]:
        values = pd.Series(table[column] if column in table.columns else np.nan,
                           index=table.index, dtype="float64")
        absent = values.isna()
        filled = values.fillna(fitted["medians"][column])
        out[column] = (filled - fitted["means"][column]) / fitted["stds"][column]
        # The mask beside the value, and only where the fill actually did
        # something. `any()` is a measurement, not a convention: it is what makes
        # this a decision about this table rather than a habit.
        if bool(absent.any()):
            name = f"{column}_missing"
            out[name] = absent.astype(int)
            mask_columns.append(name)
    out[target] = table[target].to_numpy()

    # The split point is an instant. Take the window the cut row sits in, so that
    # no window has some of its rows training and the rest testing.
    cut = int(len(out) * train_fraction)
    windows = pd.Series(out["window"])
    split_point = windows.iloc[min(cut, len(out) - 1)]
    trains = windows < split_point
    out["split"] = np.where(trains, "train", "test")

    manifest = {
        "schema_version": HANDOFF_SCHEMA,
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "key": ["phone_id", "window"],
        "feature_columns": list(fitted["columns"]),
        "mask_columns": mask_columns,
        "target": target,
        "split_point": str(split_point),
        "train_fraction": float(train_fraction),
        "train_rows": int(trains.sum()),
        "test_rows": int((~trains).sum()),
        "transform": fitted,
        "ledger": ledger,
    }
    return out, manifest


if __name__ == "__main__":
    say = narrator(LAB)
    say.info("Lab 4 — features from a window, a split that keeps time, and what a join costs")

    phones, bus = load_phones(), load_bus()
    say.info("phones: %s rows, generated (seed 20200122); vehicle: %s rows, archive slice",
             f"{len(phones):,}", f"{len(bus):,}")
    speed = phones["speed"].astype(float)
    features = window_features(speed, 60)
    say.info("window of the last 60 phone speed readings, metres per second: mean %.3f, "
             "sd (ddof = 0) %.3f, min %.3f, max %.3f, slope %.5f per position, r_1 %.4f",
             features["mean"], features["std"], features["minimum"], features["maximum"],
             features["slope"], features["autocorrelation_1"])
    say.info("the same window through pandas.Series.autocorr(1) reads %.4f — a different "
             "estimator (two means, two variances); the course grades the Box–Jenkins one",
             float(speed.tail(60).autocorr(1)))

    shuffled = phones.sample(frac=1.0, random_state=20200122).reset_index(drop=True)
    train, test = split_by_time(shuffled, 0.7)
    say.info("split by time from a shuffled table: %s train rows ending %s, %s test rows "
             "starting %s — no overlap: %s", f"{len(train):,}",
             train["timestamp_utc"].max().strftime("%H:%M:%S"), f"{len(test):,}",
             test["timestamp_utc"].min().strftime("%H:%M:%S"),
             bool(train["timestamp_utc"].max() <= test["timestamp_utc"].min()))

    dense = join_growth(phones, bus, TOLERANCES)
    in_order = bus.assign(_when=pd.to_datetime(bus["utc_time"], utc=True)).sort_values("_when")
    sparse_bus = in_order.iloc[::60].drop(columns="_when")
    sparse = join_growth(phones, sparse_bus, TOLERANCES)
    table = pd.DataFrame({"vehicle twice a second (per cent matched)": dense,
                          "vehicle once per 30 s (per cent matched)": sparse}).rename_axis(
                              "tolerance (s)")
    show_table(table, "nearest match within a tolerance", logger=say)
    say.info("against the shipped vehicle every phone row matches at one second, so the "
             "curve is flat; thin the vehicle to one reading per thirty seconds and the "
             "share climbs from %.1f to %.1f per cent — the trade the archive shows as "
             "95.3 to 97.4", sparse[TOLERANCES[0]], sparse[TOLERANCES[-1]])

    fig = go.Figure()
    fig.add_scatter(x=list(TOLERANCES), y=[dense[t] for t in TOLERANCES], mode="lines+markers",
                    name="vehicle reporting twice a second (shipped slice)",
                    line=dict(color="#2A78D6"), marker=dict(size=9))
    fig.add_scatter(x=list(TOLERANCES), y=[sparse[t] for t in TOLERANCES],
                    mode="lines+markers+text",
                    name="vehicle thinned to one reading per 30 s",
                    line=dict(color="#E07B39"), marker=dict(size=9),
                    text=[f"{sparse[t]:.1f}" for t in TOLERANCES], textposition="top center")
    fig.update_xaxes(type="log", tickvals=list(TOLERANCES),
                     title_text="how far in time a match may reach (seconds)")
    fig.update_yaxes(title_text="phone rows matched (per cent)", range=[0, 110])
    fig.update_layout(title="What widening the tolerance buys, measured", legend=dict(y=0.35))
    save_figure(fig, "join_cost", LAB, logger=say)

    # The hand-off. Built from this module's own labs: Lab 1's alignment and
    # ledger, Lab 3's fitted transform, and the split from this lab. Written to
    # out/handoff/ so that Module 3 has something to open rather than a promise.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from lab_01 import align                                   # noqa: E402
    from lab_03 import fit_preprocessing                        # noqa: E402

    aligned, ledger = align(bus, phones, 5)
    stored_columns = ["phone_speed", "rssi1", "rssi2", "bus_speed"]
    in_time_order = aligned.sort_values("window")
    training_rows = in_time_order.iloc[:int(len(in_time_order) * TRAIN_FRACTION)]
    fitted = fit_preprocessing(training_rows[stored_columns])
    table, manifest = assemble(aligned, fitted, ledger, TARGET_COLUMN, TRAIN_FRACTION)

    destination = pathlib.Path(__file__).resolve().parent.parent / "out" / "handoff"
    destination.mkdir(parents=True, exist_ok=True)
    table.to_parquet(destination / "table.parquet", index=False)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    say.info("hand-off written to out/handoff/: %s rows x %d columns, key %s, masks %s, "
             "split point %s, %s train and %s test rows — the transform and the ledger "
             "are inside manifest.json", f"{manifest['rows']:,}", manifest["columns"],
             manifest["key"], manifest["mask_columns"], manifest["split_point"],
             f"{manifest['train_rows']:,}", f"{manifest['test_rows']:,}")
    show_table(table.head(6), "the first rows of the table Module 3 opens", logger=say)

    say.info("what the check grades: the six features against its own arithmetic over exactly "
             "the last width values (r_1 by the Box–Jenkins formula, sd with ddof = 0); a "
             "split whose last training time is at or before the first test time from a "
             "shuffled table; and matched shares that never fall as the tolerance widens and "
             "match the reference within 0.2 points on the thinned vehicle; and the "
             "hand-off table — one row per phone per window, a mask beside what was "
             "filled and beside nothing else, a split point recorded as an instant with "
             "no window straddling it, and the transform and the ledger stored unchanged")

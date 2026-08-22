#!/usr/bin/env python3
"""Check 2 — the mask survives, the recursion is the stated one, and the bias is measured."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, not_ready, explain                          # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from lab_support import load_phones                               # noqa: E402
    import numpy as np                                                # noqa: E402
    import pandas as pd                                               # noqa: E402
except ImportError as unready:
    not_ready(unready)

BEACONS = ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]
PROXIMITY = {"rssiA": "proxA", "rssiB": "proxB", "rssiC": "proxC",
             "rssi1": "prox1", "rssi2": "prox2"}
ALPHA = 0.3     # the smoothing weight the slide and the stub both state
SENTINEL = -1

# One planted trace per phone, with the gaps written where they can be checked
# by hand. Phone P2 opens with a gap on purpose: with one series per phone there
# is nothing yet to carry forward there, and a single recursion run down the
# whole column would quietly fill those two rows with P1's last state.
PLANTED = {
    "P1": [-60.0, -62.0, None, None, -70.0, -68.0, None, -75.0],
    "P2": [None, None, -80.0, -82.0, None, -79.0, -77.0, None],
}


def planted_frame():
    """A frame with the same columns the labs read, and a gap we know the answer to."""
    rows = []
    for phone, readings in PLANTED.items():
        for reading in readings:
            row = {"phone_id": phone}
            for beacon in BEACONS:
                row[beacon] = reading
                row[PROXIMITY[beacon]] = SENTINEL if reading is None else 2
            rows.append(row)
    return pd.DataFrame(rows)


def reference_ema(values, series, alpha=ALPHA):
    """s_t = alpha*x_t + (1-alpha)*s_{t-1} over the heard readings, one series each.

    Written as the recursion rather than called from a library, so that the check
    grades the formula on the slide instead of grading agreement with whichever
    library the solution happens to use. At a gap the state is carried: the
    smoothed value does not move, because nothing was heard to move it.
    """
    state, carried = {}, []
    for key, value in zip(series, values):
        if value is not None and not pd.isna(value):
            previous = state.get(key)
            state[key] = value if previous is None else alpha * value + (1 - alpha) * previous
        carried.append(state.get(key, float("nan")))
    return carried


def bias_against_truth(filled, truth, missing):
    """How far the filled values sit from what was really there."""
    if missing.sum() == 0:
        return 0.0
    guessed = filled[missing]
    actual = truth[missing]
    both = guessed.notna() & actual.notna()
    if both.sum() == 0:
        return 0.0
    return float((guessed[both] - actual[both]).mean())


def body(lab):
    # The check holds the truth; the student's loader does not hand it over.
    honest = load_phones(with_truth=True)
    student_view = honest.drop(columns=[c for c in honest.columns
                                        if c.endswith("_true") or c == "aboard_truth"])

    biases, filled_by_method = {}, {}
    for method in ("drop", "mean", "ema_masked"):
        result = lab.impute_with_mask(student_view, method)
        filled_by_method[method] = result

        for beacon in BEACONS:
            mask_name, filled_name = f"{beacon}_missing", f"{beacon}_filled"
            assert mask_name in result.columns, f"[{method}] no '{mask_name}' column"
            assert filled_name in result.columns, f"[{method}] no '{filled_name}' column"

            # The mask must mark BOTH encodings of absence.
            expected = student_view[beacon].isna() | (student_view[PROXIMITY[beacon]] == -1)
            got = result[mask_name].astype(bool)
            wrong = int((got != expected).sum())
            assert wrong == 0, (
                f"[{method}] the mask for {beacon} is wrong on {wrong:,} rows. The same "
                f"absence is written down twice here: {beacon} is empty and "
                f"{PROXIMITY[beacon]} is -1, on exactly the same rows. A mask that "
                "recognises only one of the two will let -1 be averaged in as though "
                "it were a real proximity band.")

        # The mask must not be destroyed by the filling.
        assert result["rssi1_missing"].sum() > 0, (
            f"[{method}] the mask is empty. Build the mask before you fill, not after.")

        biases[method] = bias_against_truth(
            result["rssi1_filled"], honest["rssi1_true"], result["rssi1_missing"].astype(bool))

    for method in ("mean", "ema_masked"):
        assert biases[method] > 5.0, (
            f"filling with {method} sits only {biases[method]:.1f} decibels from the truth. "
            "On this mechanism it should be badly wrong — the absent readings are the far "
            "ones, so anything built from the present readings stands in for a far value "
            "using near evidence. Check that you are filling the masked rows.")

    # The bias assertions above are satisfied by any strong constant: a fill of
    # -40 decibel-milliwatts sits a long way from readings that were far and
    # weak, whatever produced it. So hold each of the three methods to the thing
    # it was actually asked for.
    absent = student_view["rssi1"].isna() | (student_view[PROXIMITY["rssi1"]] == -1)
    heard = student_view["rssi1"].where(~absent)

    assert filled_by_method["drop"]["rssi1_filled"][absent].isna().all(), (
        '["drop"] filled the absent readings in. "drop" means leave the gaps as '
        "gaps: the mask still has to be right, and nothing may be invented. It is "
        "on the figure at nought bias for exactly that reason.")

    mean_fill = filled_by_method["mean"]["rssi1_filled"][absent].to_numpy(dtype=float)
    assert np.allclose(mean_fill, float(heard.mean())), (
        f'["mean"] filled the absent rssi1 readings with something other than '
        f"{heard.mean():.2f}, the mean of the readings that were actually heard. A "
        "number chosen by hand is not a mean, however wrong it manages to be.")

    ema_fill = filled_by_method["ema_masked"]["rssi1_filled"][absent]
    assert ema_fill.nunique() > 1, (
        '["ema_masked"] gave every absent reading the same value, so it is a flat '
        "fill wearing a moving average's name. An average that moves reports a "
        "different value at different points in the series — that is what carrying "
        "it forward across a gap means.")

    # The recursion itself, on a planted gap. "An average that moves" is
    # satisfied by anything that varies; this is the formula the slide states.
    planted = planted_frame()
    smoothed = lab.impute_with_mask(planted, "ema_masked")
    expected = reference_ema(planted["rssi1"], planted["phone_id"])
    for beacon in ("rssi1", "rssiC"):
        got = smoothed[f"{beacon}_filled"].to_numpy(dtype=float)
        for index, (mine, yours) in enumerate(zip(expected, got)):
            heard = planted[beacon].iloc[index]
            target = float(heard) if not pd.isna(heard) else mine
            if pd.isna(target):
                assert pd.isna(yours), (
                    f'["ema_masked"] {beacon} row {index} is the opening of a phone\'s '
                    "trace and nothing has been heard from that phone yet, so there is "
                    f"no state to carry — you filled it with {yours}.")
                continue
            assert abs(yours - target) <= 1e-9, (
                f'["ema_masked"] {beacon} row {index} of the planted trace reads '
                f"{yours:.4f}; the recursion s_t = {ALPHA}*x_t + {1 - ALPHA:.1f}*s_(t-1) "
                f"over the heard readings gives {target:.4f}. Check the weight, check "
                "that the gaps are skipped rather than averaged in, and check that each "
                "phone carries its own state — the frame holds two.")

    # The bias, as arithmetic rather than as a direction. Three planted triples
    # whose answers can be worked out on paper.
    for filled, truth, missing, expected_bias, why in (
            ([1.0, 2.0, 100.0, 100.0], [0.0, 0.0, 0.0, 0.0], [True, True, False, False],
             1.5, "the mean runs over the filled rows only, not over every row"),
            ([float("nan"), float("nan"), 3.0, 4.0], [0.0, 0.0, 0.0, 0.0],
             [True, True, False, False], 0.0,
             '"drop" fills nothing, so it invents nothing and its bias is nought'),
            ([-70.0, -60.0], [-65.0, -75.0], [True, True], 5.0,
             "the difference is fill minus truth, in that order")):
        measured = lab.imputation_bias(pd.Series(filled), pd.Series(truth),
                                       pd.Series(missing))
        assert abs(float(measured) - expected_bias) <= 1e-9, (
            f"imputation_bias({filled}, {truth}, {missing}) returned {measured}; "
            f"it is {expected_bias}. Here {why}.")

    # And it has to agree with the check's own arithmetic on the real frame.
    for method in ("drop", "mean", "ema_masked"):
        result = filled_by_method[method]
        yours = float(lab.imputation_bias(result["rssi1_filled"], honest["rssi1_true"],
                                          result["rssi1_missing"]))
        assert abs(yours - biases[method]) <= 1e-6, (
            f'["{method}"] imputation_bias() reports {yours:+.3f} decibels on the whole '
            f"day and the check measures {biases[method]:+.3f}.")

    answer = lab.fills_are_biased_which_way().strip().lower()
    assert answer in {"too strong", "too weak"}, (
        f'fills_are_biased_which_way() must return "too strong" or "too weak"; '
        f'you returned "{answer}"')
    # Two strings, so on its own this is a coin flip -- and until this pass the
    # failure message flipped it for the student, in as many words. It now says
    # which property broke and withholds the reasoning until the third attempt.
    assert answer == "too strong", explain(
        "fills_are_biased_which_way",
        "that is not the direction the fills are biased in",
        "You measured this two functions ago and did not read it. imputation_bias() "
        "is the mean of (fill - truth) over the rows each fill touched, so its SIGN is "
        "the answer -- there is nothing else to decide. Print the two numbers you "
        "already computed for the mean and for the masked moving average, look at the "
        "sign, and then say why it comes out that way: which readings are the ones "
        "that exist, and which are the ones that are missing?")


run(2, "02_the_mechanism", "impute_with_mask", body)

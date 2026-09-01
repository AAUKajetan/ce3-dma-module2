"""Lab 2, solved — with the reasoning, not only the code."""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_phones  # noqa: E402,F401
from _narrate import narrator, show_table, save_figure  # noqa: E402

LAB = 2
BEACONS = ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]
PROXIMITY = {"rssiA": "proxA", "rssiB": "proxB", "rssiC": "proxC",
             "rssi1": "prox1", "rssi2": "prox2"}
SENTINEL = -1
SMOOTHING = 0.3   # alpha, the weight of the newest heard reading -- a stated choice


def _missing(frame, beacon: str) -> pd.Series:
    """Absent, however it was written down: m_t = 1 when rssi is null or prox is −1.

    The archive encodes the same absence twice: the signal strength is empty and
    the proximity column holds -1, on exactly the same rows, for every beacon.
    Recognise only the empty one and every mean you take over the proximity
    column averages in -1 as though a phone had been at proximity band minus
    one, which is not a place.
    """
    absent = frame[beacon].isna()
    proximity = PROXIMITY[beacon]
    if proximity in frame.columns:
        absent = absent | (frame[proximity] == SENTINEL)
    return absent


def _ema_masked(values: pd.Series, series_id) -> pd.Series:
    """s_t = α·x_t + (1−α)·s_{t−1} over the heard readings, carried across gaps.

    One series per phone, in the frame's row order (time order): a gap in one
    volunteer's trace is filled from that volunteer's last heard readings, not
    from whoever happened to report in the row above. `adjust=False` is the
    plain recursion the slide states; `ignore_na=True` skips the gaps rather
    than letting them dilute the weights, and at a gap the mean is simply the
    last state, which is what "carried forward" means (Servizi et al., 2023).
    """
    def recursion(x: pd.Series) -> pd.Series:
        return x.ewm(alpha=SMOOTHING, adjust=False, ignore_na=True).mean()

    if series_id is None:
        return recursion(values)
    return values.groupby(series_id, sort=False).transform(recursion)


def impute_with_mask(frame, method: str = "ema_masked"):
    """Fill the absent readings, and keep the record that you did.

    The mask is the part people leave out, and it is the part that matters. A
    filled value and a measured value look identical in a table; only the mask
    distinguishes them, and every model downstream needs to know which is which.
    Without it you cannot even tell afterwards how much of your data you made up.
    """
    filled_frame = frame.copy()
    series_id = frame["phone_id"] if "phone_id" in frame.columns else None

    for beacon in BEACONS:
        absent = _missing(frame, beacon)
        values = frame[beacon].where(~absent)

        if method == "drop":
            filled = values                       # leave the gaps as gaps
        elif method == "mean":
            filled = values.fillna(values.mean())  # x̄ over the heard readings
        elif method == "ema_masked":
            # It does not pretend the phone was near the beacon; it holds the
            # last thing it knew about that phone. Heard rows keep their value.
            filled = values.fillna(_ema_masked(values, series_id))
        else:
            raise ValueError(f"unknown method {method!r}")

        filled_frame[f"{beacon}_filled"] = filled
        filled_frame[f"{beacon}_missing"] = absent

    return filled_frame


def imputation_bias(filled, truth, missing) -> float:
    """Mean of (fill − truth) over the rows that were filled, in decibels.

    Positive means the fills sit above what was really there -- "too strong".
    The mean runs over the rows the mask marks absent *and* the method actually
    filled: "drop" fills nothing, so its bias is nought by construction, which
    is exactly what the figure shows for it. A method is judged only on what it
    invented.
    """
    filled = pd.Series(filled, dtype="float64").reset_index(drop=True)
    truth = pd.Series(truth, dtype="float64").reset_index(drop=True)
    missing = pd.Series(missing).astype(bool).reset_index(drop=True)
    touched = missing & filled.notna() & truth.notna()
    if not touched.any():
        return 0.0
    return float((filled[touched] - truth[touched]).mean())


def fills_are_biased_which_way() -> str:
    """Too strong — and the mechanism settles it before any code runs.

    Rubin's (1976) three cases, in one line each. Missing completely at random:
    the chance of a gap does not depend on any value, seen or unseen. Missing at
    random: it depends only on values you can see. Missing not at random: it
    depends on the very value that is missing. Signal strength falls with
    distance and a beacon is heard only when it is near, so a reading is absent
    *because* it was weak -- the third case, and the one where every fill made
    from what you can see is made from the near readings.

    Any fill built from what you have — a mean, a moving average, the last value
    carried forward — is therefore made of near evidence and used to stand in for
    a far measurement. It comes out too strong. Measured here, both methods put
    the absent readings well over ten decibels stronger than they really were.

    The more useful surprise is how close the two methods are. The masked moving
    average is barely better than the flat mean, because it too is carrying
    forward a near reading. No imputation recovers information that was never
    recorded; the choice of method changes the damage by a little, and the mask
    is what lets anyone downstream know the damage exists at all.

    That is why "drop" sits at zero on the figure. It invents nothing — and it
    throws away every row where the absence was itself the measurement.
    """
    return "too strong"


if __name__ == "__main__":
    say = narrator(LAB)
    say.info("Lab 2 — three fills against a known mechanism, and what each one invents")

    student_view = load_phones()
    truth = load_phones(with_truth=True)
    say.info("phones: %s rows, generated (seed 20200122); the truth frame keeps the "
             "signal every reading would have had and the distance to each beacon",
             f"{len(student_view):,}")
    absent = _missing(student_view, "rssi1")
    say.info("rssi1 absent on %.1f per cent of rows, both encodings agreeing on every "
             "row: %s", absent.mean() * 100,
             bool((student_view["rssi1"].isna() == (student_view["prox1"] == SENTINEL)).all()))

    biases, filled_by_method = {}, {}
    for method in ("drop", "mean", "ema_masked"):
        result = impute_with_mask(student_view, method)
        filled_by_method[method] = result
        biases[method] = imputation_bias(result["rssi1_filled"], truth["rssi1_true"],
                                         result["rssi1_missing"])
        say.info("%-10s masked %s rows; bias = mean(fill − truth) on the filled rows: "
                 "%+.1f dB", method, f"{int(result['rssi1_missing'].sum()):,}", biases[method])
    show_table(pd.DataFrame({"bias_db": biases}).rename_axis("method"),
               "imputation bias on rssi1, decibels above the hidden truth", logger=say)
    say.info("the fills come out %s, because the readings you can see are the near ones",
             fills_are_biased_which_way())

    # One volunteer, one beacon: where each fill puts the phone against where it was.
    one = truth["phone_id"] == truth["phone_id"].iloc[0]
    distance = truth.loc[one, "rssi1_distance_true"]
    heard = ~absent[one]
    fig = make_subplots(cols=2, rows=1, column_widths=[0.68, 0.32],
                        subplot_titles=("one volunteer, beacon 1, first day",
                                        "bias on the filled rows, whole day"))
    fig.add_scatter(x=distance[heard], y=truth.loc[one & heard, "rssi1"], mode="markers",
                    name="heard, and recorded", marker=dict(color="#2A78D6", size=6), row=1, col=1)
    fig.add_scatter(x=distance[~heard], y=truth.loc[one & ~heard, "rssi1_true"], mode="markers",
                    name="not heard: the hidden truth",
                    marker=dict(color="#52514E", size=4, opacity=0.5), row=1, col=1)
    fig.add_scatter(x=distance[~heard],
                    y=filled_by_method["ema_masked"].loc[one & ~heard, "rssi1_filled"],
                    mode="markers", name="masked moving average, carried forward",
                    marker=dict(color="#E07B39", size=6, symbol="diamond"), row=1, col=1)
    mean_of_heard = float(student_view["rssi1"].where(~absent).mean())
    fig.add_scatter(x=[float(distance.min()), float(distance.max())],
                    y=[mean_of_heard, mean_of_heard], mode="lines",
                    name="mean of the heard readings", line=dict(color="#C0392B", width=2),
                    row=1, col=1)
    fig.add_bar(x=["drop", "mean", "masked EMA"],
                y=[biases["drop"], biases["mean"], biases["ema_masked"]],
                marker_color=["#52514E", "#C0392B", "#E07B39"],
                text=[f"{b:+.1f}" for b in (biases["drop"], biases["mean"], biases["ema_masked"])],
                textposition="outside", showlegend=False, row=1, col=2)
    fig.update_xaxes(title_text="true distance to the beacon (metres)", row=1, col=1)
    fig.update_yaxes(title_text="signal strength (decibel-milliwatts)", row=1, col=1)
    fig.update_yaxes(title_text="fill − truth (decibels)", row=1, col=2)
    fig.update_layout(title="Every fill is made of near readings and stands in for far ones",
                      legend=dict(orientation="h", y=-0.2))
    save_figure(fig, "imputation_by_distance", LAB, logger=say, width=1100, height=600)

    say.info("what the check grades: the mask marks exactly the rows where rssi is null or "
             "prox is −1; drop leaves gaps, mean fills with the mean of the heard readings, "
             "the masked average reproduces s_t = α·x_t + (1−α)·s_{t−1} with α = 0.3 on a "
             "planted gap; imputation_bias() equals the mean of fill − truth on the filled "
             "rows; and the direction is 'too strong'")

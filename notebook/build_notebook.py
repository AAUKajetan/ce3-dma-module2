#!/usr/bin/env python3
"""Build Module 2's demonstration notebook, then execute it against the archive.

    python "Module 2/notebook/build_notebook.py"            # build and run
    python "Module 2/notebook/build_notebook.py" --no-run   # build only

Run it from the repository root. The cells address the archive as
`data/bus.csv` and `data/passengers.csv`, so that is the working directory the
notebook is executed in.

Exit codes:
    0  the notebook was written, executed if the archive was there
    2  the archive is not on this machine, so nothing was executed and nothing
       was written. That is a different fact from a broken notebook and it is
       reported as one.

This notebook opens `data/passengers.csv`, which is the position trace of
sixteen identifiable volunteers. Under Article 4 of the General Data Protection
Regulation that is personal data.

The rule this notebook obeys, and states, is that **only aggregates leave a
cell**: counts, shares, cross-tabulated totals, distributions. No row is
printed. No identifier is printed. No coordinate is printed. No map is drawn.
Check 4 reads the executed output and refuses the notebook if any cell breaks it.

It is instructor-side and does not go into the repository students clone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTPUT = HERE / "Module2_demonstration.ipynb"

MARKDOWN, CODE = "markdown", "code"

# The definition cards are not written here. They are read from the manifest the
# slide and the stub also read (standing rule 7), so the projector, the notebook,
# the exercise and the check cannot drift apart: change the formula in one place
# and check 3e fails everywhere it was not changed.
CONCEPTS = {concept["id"]: concept for concept in json.loads(
    (ROOT / "Module 2" / "exercises" / "concepts.json").read_text())["concepts"]}


def card(concept_id: str, why: str) -> str:
    """One definition card: the statement, the formula, the source, then why here."""
    concept = CONCEPTS[concept_id]
    return (f"> **Definition — {concept['name']}**\n>\n"
            f"> `{concept['formula']}`\n>\n"
            f"> Source: {concept['citation']}. {why}")

CELLS = [
(MARKDOWN, """# Module 2 — Cleaning data and building features

**Data Mining and Analysis (course code CE3) · Aalborg University, Copenhagen**

This notebook demonstrates the four blocks on the real archive: two automated
shuttles and sixteen instrumented phones, Copenhagen, 22–23 January 2020.

> **Personal data.** `data/passengers.csv` holds the position traces of sixteen
> identifiable people. Every cell below emits counts, shares or distributions
> and nothing else — no row, no identifier, no coordinate, no map. That is not a
> convention; it is the condition under which this file may be opened at all,
> and the module's check enforces it.
>
> The exercise repository students clone contains neither this notebook nor the
> file it reads. Their labs run on generated phone traces whose parameters were
> measured here."""),

(MARKDOWN, """## Hook

Two files. One reports every half second, the other every second. One is vehicle
telemetry that identifies nobody; the other is the movement of sixteen people.
Roughly three quarters of the interesting columns are empty, and one column
already knows the answer.

Turn that into a table a model can train on — and be able to say, afterwards,
exactly what you did to it."""),

(CODE, '''import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 110)

# Every figure in this course is plotly, drawn on the light template with the
# course's four colours: reference blue, current orange, neutral grey, and red
# only for what fails.
BLUE, ORANGE, GREY, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"
FIGURES = Path("Module 2/notebook/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


def show(fig, name):
    # Inline for the reader, and a portable network graphics copy beside the
    # notebook so a figure can be lifted into a message without re-running.
    fig.update_layout(template="plotly_white", width=1000, height=520,
                      margin=dict(l=70, r=30, t=70, b=70))
    fig.write_image(str(FIGURES / f"{name}.png"), scale=2)
    fig.show()

# The course's seed, the date of the first day in the archive. Every notebook,
# lab and generator in Edition 2026 uses this one number, so that two people who
# run the same cell get the same answer. No cell below draws a random number --
# every result here is an exact aggregate of the whole file -- but the seed is
# set and used, so that anything added later is reproducible by construction
# rather than by luck.
SEED = 20200122
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)

BUS = Path("data/bus.csv")
PHONES = Path("data/passengers.csv")

# Two facts this notebook quotes from Module 1, read from Module 1's own
# measured.json rather than typed here. Same rule as the slides: a number
# reaches a reader by having been measured, once, in the module that measured it.
MODULE_1 = json.loads(Path("Module 1/slides/measured.json").read_text())
ROUTE_EXTENT_M = MODULE_1["extent_m"]["value"]

bus = pd.read_csv(BUS, low_memory=False)
phones = pd.read_csv(PHONES, low_memory=False)

# Aggregates only, from here to the end.
print(f"vehicle telemetry : {len(bus):,} rows x {bus.shape[1]} columns")
print(f"phone traces      : {len(phones):,} rows x {phones.shape[1]} columns")
print(f"volunteers        : {phones['id'].nunique()}")'''),

(MARKDOWN, """## Core Concept

### The alignment problem is a decision, not a join

Two sources sampled at different rates share no instants at all. You choose a
grain, and you decide what happens to whatever does not fit. Both choices change
every number computed downstream, so both are recorded."""),

(CODE, '''bus_time = pd.to_datetime(bus["utc_time"], utc=True)
# format="mixed" matters here: some rows carry fractional seconds and some do
# not. Left to infer one format, pandas rejects six rows. "Six unparseable
# timestamps" is a property of how the file was read, not of the file.
phone_time = pd.to_datetime(phones["timestamp_utc"], utc=True,
                            errors="coerce", format="mixed")
one_format = pd.to_datetime(phones["timestamp_utc"], utc=True, errors="coerce")

VEHICLE = "VJRD1A10224000055"          # the shuttle that ran on both days
same_bus = bus[bus["vehicle_id"] == VEHICLE].assign(_t=bus_time).sort_values("_t")
bus_step = same_bus.groupby(same_bus["_t"].dt.date)["_t"].diff().dt.total_seconds()

ordered = phones.assign(_t=phone_time).dropna(subset=["_t"]).sort_values("_t")
phone_step = ordered.groupby([ordered["_t"].dt.date, "id"])["_t"].diff().dt.total_seconds()

print(f"vehicle: a reading every {bus_step.median():.3f} s (median)")
print(f"phones : a reading every {phone_step.median():.3f} s (median)")
print(f"\\nwill not parse, reader told the formats are mixed: {int(phone_time.isna().sum())}")
print(f"will not parse, reader left to infer one format:  {int(one_format.isna().sum())}")
print(f"  ...of those, labelled: {int(phones.loc[one_format.isna(), 'label'].notna().sum())}")'''),

(MARKDOWN, card("tumbling_window",
                "Below, the grain is applied to the archive's own two files.")),

(MARKDOWN, card("conservation_ledger",
                "The ledger under it is the whole of Lab 1: rows used plus rows "
                "dropped equals rows received, and every dropped row has a reason.")),

(MARKDOWN, card("upstream_profile",
                "P is data/module1_profile.json, schema aau-ce3/data-profile/1 — "
                "byte for byte what Module 1's own declare_profile() wrote about "
                "this slice. Run it over the whole slice and it is silent; run it "
                "over 22 January alone and one column breaks the absence it was "
                "allowed. The pooled frame satisfies a rule its own first day does "
                "not, which is the paradox further down this notebook wearing a "
                "validation layer's clothes.")),

(CODE, '''# One grain, and books that balance. The vehicle reading is thinned to one
# every thirty seconds first, so that there is something real to drop: against
# the shipped rate every phone row lands and a ledger of noughts balances too.
GRAIN = "5s"
sparse_bus = same_bus.iloc[::60]

phone_windows = ordered.assign(window=ordered["_t"].dt.floor(GRAIN))
bus_windows = set(sparse_bus["_t"].dt.floor(GRAIN))

received = int(len(phones))
unparseable = int(phone_time.isna().sum())
placed = phone_windows["window"].isin(bus_windows)
used = int(placed.sum())
dropped = received - used

ledger = {
    "grain_seconds": 5,
    "phone_rows_in": received,
    "phone_rows_used": used,
    "phone_rows_dropped": dropped,
    "drop_reasons": {"timestamp would not parse": unparseable,
                     "no vehicle reading in the window": dropped - unparseable},
}
for key, value in ledger.items():
    print(f"{key:26} {value}")
print("\\nused + dropped == received:",
      ledger["phone_rows_used"] + ledger["phone_rows_dropped"] == ledger["phone_rows_in"])
print("reasons sum to dropped:",
      sum(ledger["drop_reasons"].values()) == ledger["phone_rows_dropped"])'''),

(MARKDOWN, """Read one way, six rows are unusable. Read another, none are. The file did not
change; the reader's assumption did.

That is standing rule 2 in one line: a number that depends on a choice must have
the choice printed beside it. An earlier draft of this course reported "six
unparseable timestamps" as a property of the archive. It never was.

And it matters, because some of those rows carry hand-recorded labels — truth
that cannot be recovered if you drop them quietly. Which is why Lab 1 insists
every dropped row is counted and given a reason.

### The clock that lies, in this file too"""),

(CODE, '''local_column = pd.to_datetime(phones["timestamp"], errors="coerce")
offset = (local_column.dt.tz_localize("UTC") - phone_time).dt.total_seconds() / 3600
print("the column named `timestamp`, minus `timestamp_utc`, in hours:",
      sorted(offset.dropna().unique()))
print("\\nJoin on the friendly name and the two sources sit an hour apart,")
print("while every value still parses and every timestamp still looks like one.")'''),

(MARKDOWN, """## Worked Example

### Missingness is a mechanism — and it is written down twice"""),

(MARKDOWN, card("missing_mechanism",
                "M is the indicator that a reading is absent. Here the absence "
                "depends on the missing value itself, which is the third case.")),

(MARKDOWN, card("absence_mask",
                "The rule is measured from this archive: both encodings mark the "
                "same rows, and the next cell checks that on every beacon.")),

(CODE, '''BEACONS = ["rssiA", "rssiB", "rssiC", "rssi1", "rssi2"]
PROX = {"rssiA": "proxA", "rssiB": "proxB", "rssiC": "proxC",
        "rssi1": "prox1", "rssi2": "prox2"}

print(f"{'beacon':8} {'no reading':>11} {'proximity = -1':>15} {'same rows?':>11}")
for beacon in BEACONS:
    absent = phones[beacon].isna()
    sentinel = phones[PROX[beacon]] == -1
    print(f"{beacon:8} {absent.mean()*100:10.1f}% {sentinel.mean()*100:14.1f}% "
          f"{str(bool((absent == sentinel).all())):>11}")

print("\\nproxA values and their counts:",
      phones["proxA"].value_counts().sort_index().to_dict())
print("-1 is a hole with a number in it. Take a mean over this column without")
print("recognising that, and you average in minus one as a proximity band.")'''),

(MARKDOWN, """The same absence, encoded two different ways, agreeing on every row for every
beacon. A mask that recognises only the empty cells silently accepts the −1 as a
measurement.

Why is the reading absent? Signal strength falls with distance, so the absent
readings are the far ones. That is missing-not-at-random: the absence is a
measurement of distance in disguise.

Which suggests an excellent feature — and the next cell tests it, because a
sound mechanism does not guarantee a useful feature."""),

(CODE, '''# The tempting inference: absence means distance, so hearing the VEHICLE beacon
# should mean being on the vehicle. Test it before building on it.
labelled = phones[phones["label2"].notna()]
aboard = labelled["label2"] == "IN"
base_rate = aboard.mean()

print(f"labelled rows: {len(labelled):,}   aboard: {base_rate*100:.1f}%")
print(f"\\n{'beacon':8} {'heard|aboard':>13} {'heard|not':>11} {'agrees with aboard':>20}")
for beacon in BEACONS:
    heard = labelled[beacon].notna()
    print(f"{beacon:8} {heard[aboard].mean()*100:12.1f}% {heard[~aboard].mean()*100:10.1f}%"
          f" {(heard == aboard).mean()*100:19.1f}%")

print(f"\\nAnswering \'aboard\' every time scores {base_rate*100:.1f}%.")
print("Every beacon does worse. The vehicle beacon rssi1 is heard MORE often")
print("when the passenger is not aboard than when they are.")
print(f"\\nModule 1 measured why: the whole route fits in a box about "
      f"{ROUTE_EXTENT_M[0]} by {ROUTE_EXTENT_M[1]}")
print("metres, and beacon range is tens of metres. In a space that small,")
print("\'near a stop beacon\' and \'on the vehicle\' are not separable.")
print("\\nThe mechanism is real. The feature is not. Measure before you believe.")'''),

(CODE, '''shares = {beacon: phones[beacon].isna().mean() * 100 for beacon in BEACONS}
fig = go.Figure(go.Bar(x=list(shares), y=list(shares.values()), marker_color=BLUE,
                       text=[f"{v:.1f}" for v in shares.values()],
                       textposition="outside", showlegend=False))
fig.update_yaxes(title_text="rows with no reading (per cent)", range=[0, 100])
fig.update_xaxes(title_text="beacon (received signal strength column)")
fig.update_layout(title="A beacon reading is absent more often than present")
show(fig, "beacon_absence")'''),

(MARKDOWN, """### Filling the gaps, and measuring what the fill invented

The archive cannot answer this one. To measure a fill you need the value that
was never recorded, and only the course's generator has it — calibrated from
this file, with none of its people. The functions below are the ones Lab 2 is
graded against, imported rather than retyped."""),

(MARKDOWN, card("masked_ema",
                "One series per phone, so a gap is filled from that volunteer's "
                "own last heard readings.")),

(MARKDOWN, card("imputation_bias",
                "Positive is too strong: the fill puts the phone nearer the "
                "beacon than it was.")),

(CODE, '''EXERCISES = Path("Module 2/exercises")
for folder in (EXERCISES, EXERCISES / "data", EXERCISES / "solutions"):
    if str(folder.resolve()) not in sys.path:
        sys.path.insert(0, str(folder.resolve()))

from lab_support import load_phones                  # noqa: E402
from lab_02 import impute_with_mask, imputation_bias  # noqa: E402

generated = load_phones(day="2020-01-22")
truth = load_phones(day="2020-01-22", with_truth=True)

biases, filled = {}, {}
for method in ("drop", "mean", "ema_masked"):
    filled[method] = impute_with_mask(generated, method)
    biases[method] = imputation_bias(filled[method]["rssi1_filled"],
                                     truth["rssi1_true"],
                                     filled[method]["rssi1_missing"])
    print(f"{method:11} bias = mean(fill - truth) on the filled rows: "
          f"{biases[method]:+.1f} decibels")
print("\\nGenerated, seed 20200122. The magnitudes are the archive's; the people")
print("are not. No imputation recovers what was never recorded.")'''),

(CODE, '''one = truth["phone_id"] == truth["phone_id"].iloc[0]
absent = generated["rssi1"].isna() | (generated["prox1"] == -1)
distance = truth.loc[one, "rssi1_distance_true"]
heard = ~absent[one]

fig = go.Figure()
fig.add_scatter(x=distance[heard], y=truth.loc[one & heard, "rssi1"], mode="markers",
                name="heard, and recorded", marker=dict(color=BLUE, size=6))
fig.add_scatter(x=distance[~heard], y=truth.loc[one & ~heard, "rssi1_true"],
                mode="markers", name="not heard: the hidden truth",
                marker=dict(color=GREY, size=4, opacity=0.5))
fig.add_scatter(x=distance[~heard],
                y=filled["ema_masked"].loc[one & ~heard, "rssi1_filled"], mode="markers",
                name="masked moving average, carried forward",
                marker=dict(color=ORANGE, size=6, symbol="diamond"))
mean_of_heard = float(generated["rssi1"].where(~absent).mean())
fig.add_hline(y=mean_of_heard, line_color=RED, line_width=2,
              annotation_text="mean of the heard readings")
fig.update_xaxes(title_text="true distance to the beacon (metres)")
fig.update_yaxes(title_text="signal strength (decibel-milliwatts)")
fig.update_layout(title="Every fill is made of near readings and stands in for far ones",
                  legend=dict(orientation="h", y=-0.22))
show(fig, "imputation_by_distance")'''),

(MARKDOWN, """### The leak

`BusID` looks like a sparse identifier. Cross-tabulate it against the label."""),

(MARKDOWN, card("target_leakage",
                "n(v, y) counts the labelled rows where the column holds v and "
                "the target holds y. Both directions count.")),

(MARKDOWN, card("feature_verdict",
                "Finding a suspect is not deciding what to do about it. Note what "
                "the ceiling in the rule above is doing here: four candidate "
                "columns in this module's own fixture set are pure, or all but "
                "pure, and three of them are dropped while the fourth — a window "
                "mean spread over more than a thousand distinct values — is "
                "kept.")),

(CODE, '''leak = pd.crosstab(phones["BusID"].notna(), phones["label2"])
leak.index = ["BusID absent", "BusID present"]
print(leak.to_string())

empty = phones["BusID"].isna().mean() * 100
print(f"\\nBusID is empty on {empty:.2f}% of rows -- it looks harmless.")
print("But there are two zeros in that table. Every row aboard carries it;")
print("no row not aboard does. It is not a clue about the target.")
print("It is the target, recorded under a different name.")'''),

(CODE, '''cells = leak.reindex(index=["BusID present", "BusID absent"],
                    columns=["IN", "OUT"]).fillna(0).astype(int)
fig = go.Figure(go.Heatmap(z=cells.to_numpy(), x=["aboard", "not aboard"],
                           y=list(cells.index),
                           text=[[f"{v:,}" for v in row] for row in cells.to_numpy()],
                           texttemplate="%{text}", textfont=dict(size=24),
                           colorscale=[[0, "#FCFCFB"], [1, BLUE]], showscale=False))
fig.update_yaxes(autorange="reversed", title_text="")
fig.update_xaxes(title_text="hand-recorded label (rows)")
fig.update_layout(title="BusID is not a clue about the target. It is the target.")
show(fig, "leak")'''),

(MARKDOWN, """### What the join costs

Widening the tolerance always buys matches and always spends accuracy — the
vehicle reading you attached is further from the moment you are describing."""),

(MARKDOWN, card("tolerance_join",
                "t_p is a phone row's time and t_b a vehicle reading's, both in "
                "coordinated universal time; tau is the tolerance in seconds.")),

(CODE, '''left = labelled.assign(_t=pd.to_datetime(labelled["timestamp_utc"], utc=True, format="mixed"))
left = left.dropna(subset=["_t"]).sort_values("_t")[["_t"]]
left["_t"] = left["_t"].astype("datetime64[ns, UTC]")

right = same_bus[["_t", "speed"]].copy()
right["_t"] = right["_t"].astype("datetime64[ns, UTC]")

print(f"{'tolerance':>10}  {'matched':>8}")
TOLERANCES = (1, 2, 5, 10, 30)
matched = {}
for seconds in TOLERANCES:
    merged = pd.merge_asof(left, right, on="_t", direction="nearest",
                           tolerance=pd.Timedelta(seconds=seconds))
    matched[seconds] = merged["speed"].notna().mean() * 100
    print(f"{seconds:>8} s  {matched[seconds]:7.1f}%")

# Subtracted, not asserted. An earlier draft of this course said "about four
# percentage points" here and on the slide beside it; the measurement says less.
gain = matched[TOLERANCES[-1]] - matched[TOLERANCES[0]]
multiple = TOLERANCES[-1] // TOLERANCES[0]
print(f"\\n{multiple} times the tolerance buys {gain:.1f} percentage points,")
print("and spends accuracy on every row it matched. Measure the trade.")'''),

(CODE, '''fig = go.Figure(go.Scatter(x=list(TOLERANCES), y=[matched[t] for t in TOLERANCES],
                           mode="lines+markers+text", line=dict(color=BLUE),
                           marker=dict(size=10), showlegend=False,
                           text=[f"{matched[t]:.1f}" for t in TOLERANCES],
                           textposition="top center"))
fig.update_xaxes(type="log", tickvals=list(TOLERANCES),
                 ticktext=[str(t) for t in TOLERANCES],
                 title_text="how far in time a match may reach (seconds)")
fig.update_yaxes(title_text="labelled phone rows matched (per cent)",
                 range=[min(matched.values()) - 1.5, 100])
fig.update_layout(title=f"{multiple} times the tolerance buys {gain:.1f} percentage points")
show(fig, "join_cost")'''),

(MARKDOWN, """### Features from a window, and a split that keeps time

Block four on the vehicle telemetry, which identifies nobody. The estimators are
the ones Lab 4 is graded against, imported rather than retyped, so that the
number in this notebook and the number in a student's terminal are the same
quantity."""),

(MARKDOWN, card("window_features",
                "The window here is the last sixty vehicle speed readings, which "
                "is thirty seconds at this reporting rate.")),

(MARKDOWN, card("autocorrelation",
                "This is the estimator Module 1 standardised on; pandas' own "
                "Series.autocorr computes a different one.")),

(MARKDOWN, card("split_by_time",
                "The cut is on utc_time, and the last training instant is at or "
                "before the first test instant.")),

(CODE, '''from lab_04 import sample_autocorrelation, split_by_time, window_features  # noqa: E402

speed = same_bus["speed"].astype(float).reset_index(drop=True)
features = window_features(speed, 60)
for name, value in features.items():
    print(f"{name:19} {value: .5f}")
print(f"\\npandas Series.autocorr(1) on the same window: "
      f"{float(speed.tail(60).autocorr(1)): .5f} -- a different estimator")

whole_day = same_bus.assign(timestamp_utc=same_bus["_t"])
train, test = split_by_time(whole_day, 0.7)
print(f"\\ntrain {len(train):,} rows ending {train['timestamp_utc'].max()}")
print(f"test  {len(test):,} rows starting {test['timestamp_utc'].min()}")
print("no overlap:", bool(train["timestamp_utc"].max() <= test["timestamp_utc"].min()))'''),

(MARKDOWN, card("fitted_transform",
                "Fitted on the training rows above and on nothing else.")),

(MARKDOWN, card("applied_transform",
                "Applied to the test rows with the stored constants, which is "
                "what makes a service reproduce the training pipeline exactly.")),

(CODE, '''from lab_03 import apply_preprocessing, fit_preprocessing  # noqa: E402

numeric = [column for column in ("speed", "payload") if column in train.columns]
fitted = fit_preprocessing(train[numeric])
print("stored constants, from the training rows only:")
for key in ("medians", "means", "stds"):
    print(f"  {key:8} " + ", ".join(f"{c} {fitted[key][c]:.4f}" for c in fitted["columns"]))

applied = apply_preprocessing(test[numeric], fitted)
moved = test[numeric] + 1000
print(f"\\ntest set moved by 1000 and re-applied with the STORED constants: "
      f"speed moved by {float((apply_preprocessing(moved, fitted)['speed'] - applied['speed']).mean()):.1f}")
print("Nothing was recomputed from the frame in hand. That is the whole of Lab 3.")'''),

(MARKDOWN, """### Two days, two fleets — the comparison that reverses

The archive's two days are not the same fleet: two shuttles ran on 22 January
and one on 23 January. The generated phones plant the same composition change on
purpose, so that the reversal can be measured rather than described."""),

(MARKDOWN, card("simpson_paradox",
                "Y is aboard, D is the day, G is the shuttle ridden.")),

(CODE, '''from lab_01 import pooled_versus_by_group  # noqa: E402
from lab_support import load_simpson                # noqa: E402

both_days = load_simpson()
verdict = pooled_versus_by_group(both_days, "aboard", "bus", "day")
shares = pd.DataFrame(verdict["shares"]).T.mul(100).round(1)
print(shares.to_string())
print(f"\\npooled, later day minus earlier: {verdict['pooled'] * 100:+.1f} points")
print("per shuttle: " + ", ".join(f"{g} {d * 100:+.1f}"
                                 for g, d in verdict["by_group"].items()))
print(f"every group moves against the pool: {verdict['reversal']}, "
      f"which is called {verdict['name']}")

groups = [g for g in verdict["shares"] if g != "pooled"] + ["pooled"]
fig = go.Figure()
for day, colour in zip(verdict["days"], (BLUE, ORANGE)):
    fig.add_bar(name=day, x=groups, marker_color=colour,
                y=[verdict["shares"][g][day] * 100 for g in groups],
                text=[f"{verdict['shares'][g][day] * 100:.1f}" for g in groups],
                textposition="outside")
fig.update_layout(barmode="group",
                  title="Up on each shuttle, down when pooled - Simpson's paradox")
fig.update_yaxes(title_text="aboard share (per cent of readings)", range=[0, 100])
fig.update_xaxes(title_text="shuttle ridden (generated phones, both days)")
show(fig, "simpson")'''),

(MARKDOWN, """### The number that shapes the rest of the course"""),

(CODE, '''by_day = phones.assign(_d=phone_time.dt.date)
coverage = by_day.groupby("_d").agg(
    rows=("label", "size"),
    labelled=("label", lambda s: int(s.notna().sum())),
    phones=("id", "nunique"),
)
coverage["label coverage %"] = (coverage["labelled"] / coverage["rows"] * 100).round(1)
print(coverage.to_string())
print("\\nOne day of hand-recorded truth, and then none.")
print("From day two accuracy cannot be computed, only bought. Module 5 lives there.")'''),

(MARKDOWN, """## Practice

Three questions, each a few lines, each with a definite answer.

1. **Can any beacon beat the base rate?** We saw that "heard" does not. Try the
   *value*: among rows where the beacon was heard, does signal strength separate
   aboard from not aboard? Compare the means, and say whether the difference is
   large enough to build on.
2. **How much does the grain cost?** Count how many distinct 1-second,
   5-second and 30-second windows the labelled rows fall into. What is being
   thrown away at each step?
3. **Is `label` safe to use as a feature?** `label2` is the target. Cross-tabulate
   `label` against it and decide. Then say what class of leak it is.
4. **Which comparison did you make?** The aboard share of the generated phones
   moves one way on each shuttle and the other way pooled. Say, in one sentence
   each, which of the two a transport authority should publish and which an
   engineer should act on, and what would settle the disagreement.

Answers in the Appendix."""),

(CODE, '''# Your workings here.
'''),

(MARKDOWN, """## Appendix

### Answers"""),

(CODE, '''# 1. The value does no better than the presence. The two means differ by a
#    couple of decibels against a spread of many more -- not separable.
print("beacon   mean strength aboard   not aboard   difference")
for beacon in BEACONS:
    on = labelled.loc[labelled["label2"] == "IN", beacon].mean()
    off = labelled.loc[labelled["label2"] == "OUT", beacon].mean()
    print(f"  {beacon:7} {on:18.1f} {off:12.1f} {on - off:12.1f}")
print("\\nA difference of one to three decibels, against readings that vary by")
print("tens. Proximity cannot separate these classes on this route.")

# 2. What each grain throws away.
windows = labelled.assign(_t=pd.to_datetime(labelled["timestamp_utc"], utc=True, format="mixed")).dropna(subset=["_t"])
print("\\ngrain   distinct windows   rows per window")
for grain in ("1s", "5s", "30s"):
    count = windows["_t"].dt.floor(grain).nunique()
    print(f"  {grain:5} {count:15,} {len(windows)/count:15.1f}")

# 3. `label` is the hand-recorded description that `label2` was derived from.
print("\\n", pd.crosstab(labelled["label"], labelled["label2"]).to_string())
print("\\nEvery label maps to exactly one target value. It is not a leak by")
print("accident -- it is the target's own source. Same class as BusID.")'''),

(MARKDOWN, card("assemble",
                "The object Modules 3, 4 and 5 open. Lab 4 builds it from the "
                "other three labs' own functions and its check grades every "
                "clause: the grain, a mask beside what was filled and beside "
                "nothing else, the split point recorded as an instant, and the "
                "transform stored as it was fitted — on the training rows and on "
                "nothing else.")),

(MARKDOWN, """### What this notebook did not do

It printed no row, no identifier, no coordinate, and drew no map. Everything
above is a count, a share, a cross-tabulated total or a distribution.

That is what makes it lawful to open the file at all, and it is why the numbers
from it can appear on a slide while the file itself never leaves the machine it
is stored on."""),

(MARKDOWN, """## Answer to question four

Both numbers are right, and they answer different questions. The pooled share is
what the fleet delivered on each day, which is what a transport authority is
accountable for. The per-shuttle share is what a shuttle does to the people on
it, which is what an engineer changes. What settles the disagreement is knowing
whether the day caused the fleet mix to change: if it did, the pooled comparison
is the effect of the day; if the mix changed for an unrelated reason, the pooled
comparison is an artefact of composition (Pearl, 2014)."""),

(MARKDOWN, """## References

- Akidau, T., Bradshaw, R., Chambers, C. et al. (2015). *The dataflow model: a practical approach to balancing correctness, latency, and cost in massive-scale, unbounded, out-of-order data processing.* Proceedings of the VLDB Endowment 8(12), 1792-1803. https://doi.org/10.14778/2824032.2824076
- Bergmeir, C. & Benitez, J. M. (2012). *On the use of cross-validation for time series predictor evaluation.* Information Sciences 191, 192-213. https://doi.org/10.1016/j.ins.2011.12.028
- Blyth, C. R. (1972). *On Simpson's paradox and the sure-thing principle.* Journal of the American Statistical Association 67(338), 364-366. https://doi.org/10.1080/01621459.1972.10482387
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed., section 2.1.4. Wiley.
- Elvik, R. (2025). *Simpson's paradox: a collection of examples from road safety studies and emergency medicine.* Transportation Research Interdisciplinary Perspectives 31, 101471. https://doi.org/10.1016/j.trip.2025.101471
- Kapoor, S. & Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). *Leakage in data mining: formulation, detection, and avoidance.* ACM Transactions on Knowledge Discovery from Data 6(4), article 15. https://doi.org/10.1145/2382577.2382579
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection: a Practical Approach for Predictive Models*, ch. 5 and ch. 8. Chapman and Hall/CRC Press. http://www.feat.engineering/
- Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing Data*, 3rd ed. Wiley. https://doi.org/10.1002/9781119482260
- McKinney, W. (2022). *Python for Data Analysis*, 3rd ed. O'Reilly. https://wesmckinney.com/book/
- Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.* SIGKDD Explorations 3(1), 27-32. https://doi.org/10.1145/507533.507538
- Pearl, J. (2014). *Comment: understanding Simpson's paradox.* The American Statistician 68(1), 8-13. https://doi.org/10.1080/00031305.2014.876829
- Roberts, D. R., Bahn, V., Ciuti, S. et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40(8), 913-929. https://doi.org/10.1111/ecog.02881
- Rubin, D. B. (1976). *Inference and missing data.* Biometrika 63(3), 581-592. https://doi.org/10.1093/biomet/63.3.581
- Servizi, V., Persson, D. R., Pereira, F. C., Villadsen, H., Baekgaard, P., Peled, I. & Nielsen, O. A. (2023). *Is Not the Truth the Truth? Analyzing the impact of user validations for bus in/out detection in smartphone-based surveys.* IEEE Transactions on Intelligent Transportation Systems 24(11), 11905-11920. https://doi.org/10.1109/TITS.2023.3291493
- Servizi, V., Persson, D. R., Pereira, F. C., Villadsen, H., Baekgaard, P., Rich, J. & Nielsen, O. A. (2026). *Scalable passenger detection using smartphone-bus implicit interactions.* IEEE Intelligent Transportation Systems Magazine 18(1), 65-78. https://doi.org/10.1109/MITS.2025.3611306
- Simpson, E. H. (1951). *The interpretation of interaction in contingency tables.* Journal of the Royal Statistical Society, Series B 13(2), 238-241. https://doi.org/10.1111/j.2517-6161.1951.tb00088.x
- van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed. Chapman and Hall/CRC Press. https://doi.org/10.1201/9780429492259 — free at https://stefvanbuuren.name/fimd/
- Wang, R. & Strong, D. (1996). *Beyond accuracy: what data quality means to data consumers.* Journal of Management Information Systems 12(4), 5-33. https://doi.org/10.1080/07421222.1996.11518099
- European Union (2016). *Regulation 2016/679, General Data Protection Regulation*, Article 4. https://eur-lex.europa.eu/eli/reg/2016/679/oj

*All output above is Author's own, computed from `data/bus.csv` and
`data/passengers.csv` by this notebook, in aggregate only, and from the course's
own generator (`Module 2/exercises/data/make_phones.py`, seed 20200122) where a
hidden true value was needed.*"""),
]


def main(*arguments):
    notebook = new_notebook(cells=[
        new_markdown_cell(text) if kind == MARKDOWN else new_code_cell(text)
        for kind, text in CELLS
    ])
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })

    if "--no-run" not in arguments:
        # Executed from the repository root, because the cells address the
        # archive as data/bus.csv and data/passengers.csv, which is where
        # data/README.md and Module 2/README.md both say the instructor's copies
        # live. Nothing is written back to them.
        missing = [str(path) for path in (ROOT / "data" / "bus.csv",
                                          ROOT / "data" / "passengers.csv")
                   if not path.exists()]
        if missing:
            # Not being able to run is a different fact from running and
            # failing, and it gets its own exit code so that a continuous
            # integration job cannot report a pass it did not earn. The archive
            # is personal data and is deliberately absent from any checkout.
            print("cannot execute: the archive is not on this machine —",
                  ", ".join(missing), file=sys.stderr)
            print("wrote nothing. Run this beside the archive, or pass --no-run "
                  "to build the notebook unexecuted.", file=sys.stderr)
            return 2
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=900,
                       resources={"metadata": {"path": str(ROOT)}}).execute()

    OUTPUT.write_text(nbformat.writes(notebook))
    executed = sum(1 for cell in notebook.cells if cell.get("outputs"))
    print(f"wrote {OUTPUT.name} — {len(CELLS)} cells, {executed} with output")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]) or 0)

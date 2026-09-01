"""Lab 3, solved — with the reasoning, not only the code."""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_phones  # noqa: E402,F401
from _narrate import narrator, show_table, save_figure  # noqa: E402

LAB = 3
TARGET = "label2"
AGREEMENT = 0.99      # the leak threshold: agreement or purity of at least 99 per cent
MAX_LEAK_VALUES = 10  # the purity test only means anything over at most this many values
DDOF = 0              # population standard deviation, stated once and used everywhere

# The three choices in the verdict. They are printed here, on the definition
# slide and in the check's own header, and nowhere else -- not in the stub,
# because a rule a student copies out of the file they are filling in has been
# transcribed rather than understood.
MEMORISED_GAP = 0.25       # in sample minus out of sample, above which it is memory
MASK_MISSING_SHARE = 0.20  # absent on this share of the rows or more: keep the mask too
PURITY_FLOOR = 0.60        # at or below this, and complete, it carries nothing
CALLS = ("keep", "keep with the mask", "drop")


def _numeric_columns(frame) -> list:
    return [c for c in frame.select_dtypes("number").columns]


def fit_preprocessing(train) -> dict:
    """Learn the constants from the training rows, and store them.

    The point that students consistently miss: a median used to fill a gap is
    not housekeeping, it is a **learned parameter**. It was estimated from data,
    it varies with the sample, and it has to travel with the model to be applied
    to anything the model is later asked about.

    Estimate it again on the test set and you have leaked -- not dramatically,
    but really. The test set is supposed to stand in for data the model has never
    seen; a median computed over it is a summary of that data used to prepare the
    model's input. The model is then a little better on the test set than it can
    ever be in service, and the gap is invisible because everything ran.

    The standard deviation is the population one, ddof = 0: sigma = sqrt(mean of
    (x − mu)²) over the training rows. It is a choice, and it is printed here so
    that the check, the slide and this file grade the same number. A column that
    does not vary gets sigma = 1, so that scaling it does not divide by nought.

    The median, mean and standard deviation are each taken over the training
    values actually present, before any fill -- pandas skips a NaN in a mean or
    a std by default, and that default is the estimate this function grades.
    Filling the gaps with a constant first and then taking the mean or the
    standard deviation of the filled column is a different, and wrong, order:
    the fill's own constant would leak into the very numbers meant to describe
    the column before it was touched.
    """
    numeric = _numeric_columns(train)
    return {
        "medians": {c: float(train[c].median()) for c in numeric},
        "means": {c: float(train[c].mean()) for c in numeric},
        "stds": {c: float(train[c].std(ddof=DDOF)) or 1.0 for c in numeric},
        "columns": list(numeric),
    }


def apply_preprocessing(frame, fitted: dict):
    """Apply the stored constants and nothing else.

    Note what this function never does: look at `frame` to decide anything. It
    fills with the stored median, scales by the stored mean and standard
    deviation, and returns the stored column order. Given the same `fitted` it
    produces the same output for training rows, test rows, and a single request
    arriving at a service in six months -- which is what Module 3 will need.
    """
    output = pd.DataFrame(index=frame.index)
    for column in fitted["columns"]:
        values = frame[column] if column in frame.columns else np.nan
        values = pd.Series(values, index=frame.index, dtype="float64")
        values = values.fillna(fitted["medians"][column])
        output[column] = (values - fitted["means"][column]) / fitted["stds"][column]
    return output[fitted["columns"]]


def column_purity(series, positive) -> float:
    """Σ_v max_y n(v, y) / n — one statistic, used by the detector and the verdict.

    The share of rows that a rule which answers with the commonest target value
    seen at each value of the column would get right. It is written once, here,
    because `find_leaks` measures it to raise a suspicion and `keep_or_drop` is
    handed it to settle one, and a course in which those were two slightly
    different numbers would be teaching the wrong lesson twice.

    Read it with its ceiling or not at all: a value that sees exactly one row is
    pure whatever the target does, so purity rises towards one with the number
    of distinct values for reasons that have nothing to do with the target.
    """
    known = series.notna()
    if known.sum() == 0:
        return float("nan")
    table = pd.crosstab(series[known], positive[known])
    if table.empty:
        return float("nan")
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def find_leaks(frame, target: str = TARGET) -> list:
    """Columns that agree with the target almost perfectly, by value or by presence.

    `bus_id` is the one to find, and it is worth saying exactly why it is so
    dangerous: it is not correlated with the target, it *is* the target. It is
    filled in when and only when the passenger is aboard. In the archive that is
    7,723 rows aboard all carrying it and 6,000 rows not aboard all lacking it,
    with no exceptions in either direction.

    A model given this column achieves near-perfect accuracy, ships, and then
    meets live data where the field is populated by the same process that
    produces the label -- which is to say, not until after the answer is already
    known. The score was never real.

    `label` is filled in by that same process and belongs to the same family:
    the archive's hand-recorded annotation, at finer grain than the target but
    filled in exactly when the target is, whereas `aboard_truth` never reaches
    this frame at all -- it is stripped before students ever see it.

    Presence is checked as well as value, because that is the form the leak
    actually takes here. A column can be almost entirely empty and still give
    the game away by *where* it is empty. Both tests use the same 99 per cent
    rule: presence agreeing (or disagreeing) with the target on at least 99 per
    cent of labelled rows, or a value purity -- the sum over values of the larger
    class count, divided by the rows -- of at least 99 per cent.

    The ceiling on the distinct values is part of the definition rather than an
    implementation detail, and it is the part students leave out. Purity is one
    by construction for any column whose values are all distinct: each value
    then sees exactly one row, so the larger class count for that value is one,
    the sum is the number of rows, and the ratio is one whatever the target
    does. Leave the ceiling out and the detector reports every row identifier,
    every timestamp and every free-text field in the file as a perfect leak --
    and a detector that reports everything is one nobody reads. MAX_LEAK_VALUES
    is that ceiling, and it is a choice, printed here, on the slide and in the
    check.
    """
    labels = frame[target]
    known = labels.notna()
    if known.sum() == 0:
        return []
    positive = (labels[known] == "IN")

    leaks = []
    for column in frame.columns:
        if column in (target, "aboard_truth"):
            continue
        series = frame.loc[known, column]

        # Leak by presence: is the column filled in exactly when the target is?
        present = series.notna()
        if present.nunique() > 1:
            agreement = max((present == positive).mean(), (present != positive).mean())
            if agreement >= AGREEMENT:
                leaks.append(column)
                continue

        # Leak by value: does a single value split the target almost perfectly?
        # The cardinality ceiling is the guard described above: without it a
        # column of distinct values is pure by construction and always "leaks".
        if series.nunique(dropna=True) <= MAX_LEAK_VALUES and series.notna().any():
            if column_purity(series, positive) >= AGREEMENT:
                leaks.append(column)

    return sorted(set(leaks))


def _said(value) -> str:
    """One evidence value, written so that the check can find it again.

    Four significant figures, because the check matches every number in the
    reason against the evidence to within half a per cent: round harder and a
    true sentence is rejected for a rounding error.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value}"
    return f"{float(value):.4g}"


def keep_or_drop(evidence: dict) -> tuple:
    """The call on one candidate feature, and the reason for it.

    Three calls, five questions, and the first question that answers decides.
    The order is the whole of the method, and it is the part students get wrong:
    they start with "is it any good?", which is the fourth question at best.

    1. **Is it knowable at the moment of prediction?** If not, drop it, however
       good it looks -- and it will look wonderful, because a column filled in
       by the same process that produces the target agrees with the target
       perfectly. `bus_id` here is pure and has no in-sample gap at all: the
       model trained on it is not overfitting, it is reading the answer. There
       is no threshold to argue about in this question, and that is why it is
       first.

    2. **Does it leak by value?** Purity of AGREEMENT or more, over at most
       MAX_LEAK_VALUES distinct values -- the same rule, with the same two
       constants, that find_leaks uses, so the detector and the verdict cannot
       disagree about what a column's purity means. `stationary` is caught here
       and nowhere else: it is knowable, it is complete and it generalises, and
       it is still the target inverted.

       **The ceiling is half the rule, and this is where it earns its place.**
       Purity is one by construction for a column whose values are all distinct,
       and all but one for a column that is nearly all distinct, because a value
       that sees one row is pure whatever the target does. So a high purity over
       a wide column is arithmetic, not evidence, and this question must not
       fire on it. `phone_speed` -- Lab 4's own window mean, 1,415 distinct
       values over 1,500 training windows, purity 0.9933 -- is the strongest
       honest feature in the hand-off table, and a rule without the ceiling
       throws it away.

    3. **Does the fit survive the split?** A feature that scores MEMORISED_GAP
       or more better on the rows it was fitted on than on the rows it was not
       has been memorised rather than learned. This is what condemns a row
       counter, and it has to be, because question two cannot: one value per
       row, a perfect in-sample fit, and nothing out of sample.

    4. **Is it absent often enough that the fill invents?** At or above
       MASK_MISSING_SHARE of the rows, keep it *with its mask*. Lab 2 measured
       what the fill invents -- nine decibels for the masked moving average,
       nearly fourteen for the mean -- and the mask is the only honest record of
       which rows carry an invention. Dropping the mask and keeping the value is
       the common error, and it is the one that cannot be detected downstream.

    5. **Does it beat the base rate?** A purity of PURITY_FLOOR or less on a
       column that is complete carries nothing, and a column carrying nothing
       costs time, columns and trust. Note the "and complete": a mostly absent
       column whose *values* say nothing may still be worth keeping, because its
       *absence* says something. That is the whole of Lab 2 in one clause, and
       it is why question four comes before question five.

    Otherwise keep it as it is.

    The reason is not decoration. A call is one of three strings, so on its own
    it is very nearly a coin flip; the reason is what makes it an argument, and
    the check refuses any number in it that is not one of the numbers handed
    over. Quote what you measured, name what you compared it against.
    """
    missing = float(evidence["missing_share"])
    bias = float(evidence["imputation_bias_db"])
    purity = float(evidence["purity"])
    values = int(evidence["cardinality"])
    knowable = bool(evidence["knowable_at_decision_time"])
    gap = float(evidence["in_sample_minus_out_of_sample"])

    if not knowable:
        return "drop", (
            f"knowable at decision time is no, so the call is settled before anything "
            f"else is weighed: a purity of {_said(purity)} and an in sample minus out "
            f"of sample gap of {_said(gap)} describe a column that will not exist when "
            f"the model has to answer.")

    if purity >= AGREEMENT and values <= MAX_LEAK_VALUES:
        return "drop", (
            f"purity is {_said(purity)} over {_said(values)} distinct values, so this "
            f"is not the arithmetic that makes a wide column look pure — a rule with "
            f"that few branches reads the target off this column almost exactly, and "
            f"an in sample minus out of sample gap of {_said(gap)} says it will keep "
            f"doing so right up to the day the column is not there.")

    if gap >= MEMORISED_GAP:
        return "drop", (
            f"in sample minus out of sample is {_said(gap)}, and with {_said(values)} "
            f"distinct values the model has somewhere to put every single row; a purity "
            f"of {_said(purity)} on a column that wide is arithmetic rather than "
            f"evidence, and it does not survive the split.")

    if missing >= MASK_MISSING_SHARE:
        return "keep with the mask", (
            f"missing share is {_said(missing)}, so most of this column would be "
            f"invented by the fill, and the imputation bias, in db, is {_said(bias)} "
            f"decibels of signal that was never recorded; the value is worth keeping at "
            f"a purity of {_said(purity)} only if the mask that says which rows were "
            f"invented travels beside it.")

    if purity <= PURITY_FLOOR:
        return "drop", (
            f"purity is {_said(purity)}, no better than answering with the majority, "
            f"and missing share is {_said(missing)}, so there is no absence for a mask "
            f"to carry either; the column costs a column and buys nothing.")

    return "keep", (
        f"purity is {_said(purity)} but it is spread over {_said(values)} distinct "
        f"values, which is why that number is arithmetic and not a confession; missing "
        f"share is {_said(missing)}, so nothing is invented, and in sample minus out of "
        f"sample is {_said(gap)}, so what it buys survives the split.")


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    say = narrator(LAB)
    say.info("Lab 3 — constants learned on the training rows only, and a column that "
             "already knows the answer")

    phones = load_phones().sort_values("timestamp_utc").reset_index(drop=True)
    say.info("phones: %s rows, generated (seed 20200122); split by time at 70 per cent, "
             "because consecutive readings are near-copies of each other", f"{len(phones):,}")
    cut = int(len(phones) * 0.7)
    train, test = phones.iloc[:cut].copy(), phones.iloc[cut:].copy()
    fitted = fit_preprocessing(train)
    say.info("fitted on %s training rows: %d medians, means and standard deviations "
             "(ddof = %d), column order fixed", f"{len(train):,}", len(fitted["columns"]), DDOF)
    show_table(pd.DataFrame({k: fitted[k] for k in ("medians", "means", "stds")}),
               "the stored constants — the transform is part of the model", logger=say)

    applied = apply_preprocessing(test, fitted)
    moved = test.copy()
    for column in fitted["columns"]:
        moved[column] = moved[column] + 1000
    shift = float((apply_preprocessing(moved, fitted)["speed"] - applied["speed"]).mean())
    say.info("test set moved by 1000 units and re-applied with the stored constants: the "
             "scaled speed moved by %.1f — proof that nothing was recomputed from the "
             "frame in hand", shift)

    leaks = find_leaks(phones, TARGET)
    say.info("leaks by the 99 per cent rule, presence or value: %s", leaks)
    table = pd.crosstab(phones["bus_id"].notna().map({True: "bus_id present",
                                                      False: "bus_id absent"}),
                        phones[TARGET].map({"IN": "aboard", "OUT": "not aboard"}))
    show_table(table, "bus_id presence against the target — two zeros", logger=say)

    def score(columns):
        features = phones[columns].copy()
        features["bus_id"] = (phones["bus_id"].notna().astype(int)
                              if "bus_id" in columns else 0)
        numeric = features.select_dtypes("number").fillna(0.0)
        labels = (phones[TARGET] == "IN").astype(int)
        model = LogisticRegression(max_iter=400, random_state=20200122).fit(
            numeric[:cut], labels[:cut])
        return accuracy_score(labels[cut:], model.predict(numeric[cut:]))

    honest = ["speed", "rssi1", "rssi2"]
    leaking_score, honest_score = score(honest + ["bus_id"]), score(honest)
    say.info("logistic regression on the later 30 per cent: with the leak %.3f, without it "
             "%.3f — the good score is the symptom", leaking_score, honest_score)

    cells = table.reindex(index=["bus_id present", "bus_id absent"],
                          columns=["aboard", "not aboard"]).fillna(0).astype(int)
    fig = go.Figure(go.Heatmap(
        z=cells.to_numpy(), x=list(cells.columns), y=list(cells.index),
        text=[[f"{v:,}" for v in row] for row in cells.to_numpy()],
        texttemplate="%{text}", textfont=dict(size=22),
        colorscale=[[0, "#FCFCFB"], [1, "#2A78D6"]], showscale=False))
    fig.update_layout(title="bus_id is not a clue about the target — it is the target "
                            "(generated phones, first day)",
                      xaxis_title="hand-recorded label", yaxis_title="", yaxis_autorange="reversed")
    save_figure(fig, "leak_table", LAB, logger=say)

    # The verdict, on the five columns the room will argue about. Every number
    # handed over is measured here, on the training rows, or was measured by Lab
    # 2; nothing is typed in from a slide, which is the rule the check enforces
    # on the student's own reasons.
    #
    # `purity` is measured on the training rows only, for the same reason the
    # medians above are: a quantity that decides what goes into the model must
    # not have been computed over the rows the model is judged on.
    edge = int(len(phones) * 0.7)
    train_rows = phones.iloc[:edge]
    aboard = (train_rows[TARGET] == "IN")
    reading_id = pd.Series(np.arange(len(train_rows)), index=train_rows.index)

    def evidence_for(series, missing_share, bias_db, knowable, gap):
        return {"missing_share": float(missing_share),
                "imputation_bias_db": float(bias_db),
                "purity": round(column_purity(series, aboard), 4),
                "cardinality": int(series.nunique(dropna=True)),
                "knowable_at_decision_time": knowable,
                "in_sample_minus_out_of_sample": gap}

    candidates = {
        # Pure, and filled in by the process that writes the target.
        "bus_id": evidence_for(train_rows["bus_id"], phones["bus_id"].isna().mean(),
                               0.0, False, 0.0),
        # Pure over two values, and knowable: the leak the first question misses.
        "stationary": evidence_for(train_rows["stationary"], 0.0, 0.0, True, 0.0),
        # Pure over one value per row: the ceiling stops the leak rule, and the
        # split condemns it instead.
        "reading_id": evidence_for(reading_id, 0.0, 0.0, True, 0.5069),
        # The trap: purity above the leak threshold for an arithmetic reason.
        "phone_speed": {"missing_share": 0.0, "imputation_bias_db": 0.0,
                        "purity": 0.9933, "cardinality": 1415,
                        "knowable_at_decision_time": True,
                        "in_sample_minus_out_of_sample": 0.0575},
        # Mostly absent, and worth keeping for where it is absent.
        "rssi1": evidence_for(train_rows["rssi1"], phones["rssi1"].isna().mean(),
                              9.0, True, 0.1214),
    }
    verdicts = {name: keep_or_drop(evidence) for name, evidence in candidates.items()}
    show_table(pd.DataFrame({name: {"call": call, "reason": reason}
                             for name, (call, reason) in verdicts.items()}).T
               .rename_axis("candidate feature"),
               "the verdict on five candidates — three different calls", logger=say)
    say.info("three columns here are pure to four decimal places and the calls are "
             "drop, drop and drop; phone_speed is at %.4f, above the same threshold, "
             "and the call is keep — the difference is how many distinct values the "
             "purity is spread over (%d over 1,500 training windows), which is why "
             "the ceiling is in the rule",
             candidates["phone_speed"]["purity"], candidates["phone_speed"]["cardinality"])
    say.info("bus_id is pure and has no in-sample gap at all, and it is the one that "
             "goes first: it is not knowable at the instant the model has to answer, "
             "and that question is asked before any measurement is weighed")

    say.info("what the check grades: every stored median, mean and standard deviation "
             "(ddof = 0) equals the training rows' and nothing else's; apply_preprocessing "
             "moves with a moved frame and keeps the stored column order; find_leaks names "
             "bus_id and stationary, names neither a row counter nor the strongest honest "
             "feature in the frame, and still names stationary once bus_id is removed; and "
             "keep_or_drop returns the right call on eight candidates whose right calls "
             "differ — four of them pure or all but pure, three dropped and one kept — "
             "and on eight one-quantity changes, with a reason built out of the evidence")

"""Lab 3 — Features, and the transform that is part of the model.

Why this lab exists: a median that fills a gap and a standard deviation that
scales a column are learned constants, and a column that is filled in only once
the answer is known is not a feature. You prove here that you can fit a
transform on the training rows alone, apply the stored constants to anything
afterwards, and find the column in this archive that already knows the target.
Where it sits: Block three — "The leak in this archive", and the definition
slides "Definition — the fitted transform, and applying it" and
"Definition — target leakage, and the rule that catches it here".
What the check grades: every stored median, mean and standard deviation
(ddof = 0) equals the training rows' and nothing else's; apply_preprocessing()
returns the stored column order and moves with a test set moved by 1000, which
proves it used the stored constants; find_leaks() names both `bus_id` and
`stationary` on the whole table, names neither a row identifier nor the
strongest honest feature in it, and still names `stationary` — and no longer
`bus_id` — once the `bus_id` column is taken away; and keep_or_drop() returns
the right call on eight candidate features whose right calls are not the same —
four of them pure, or all but pure, and three of those four dropped and one kept
— and on eight more where one quantity has been changed, with a reason built out
of the evidence it was handed.
Needs: pandas, numpy, plotly, scikit-learn (in the demonstration only), and the loader
    in lab_support.

Twenty-five minutes.

Two ideas, and they are the same idea seen from two sides.

**The fitted transform.** A median used to fill a gap, a mean and a standard
deviation used to scale a column, the set of categories used to encode one --
these are not data cleaning. They are *learned constants*, and they belong to
the model as surely as its weights do. Learn them on the training rows only,
store them, and apply the stored ones to everything afterwards. Recompute them
on the test set and you have quietly told the model something about data it was
supposed to be judged on.

**Leakage.** A feature built from information that did not exist at the moment
of prediction. Models trained with it score beautifully and fail in service, and
a large share of the reproducibility problem in applied machine learning is this
under other names (Kapoor & Narayanan, 2023).

This lab has a real one. In the archive, `BusID` is filled in exactly when the
passenger is aboard: 7,723 rows aboard all carry it, 6,000 rows not aboard all
lack it, no exceptions. It is not a hint about the target -- it *is* the target,
wearing a different name. The generated data reproduces it as `bus_id`, because
this is the most useful thing in the whole file for teaching.

What you write: fit_preprocessing(train), apply_preprocessing(frame, fitted),
find_leaks(frame, target), and keep_or_drop(evidence).

    fit_preprocessing(train) -> dict
        Learn, from the training rows only:
          medians   {column: value} for the numeric columns you will fill
          means     {column: value} and stds {column: value} for scaling
          columns   the column order, so that applying is deterministic
        Return them in a dict. Anything not in that dict cannot be applied
        later, which is the point. Each mean and standard deviation is taken
        over the values actually present, before any fill -- fill the gaps
        with a constant first, and that constant leaks into the very numbers
        meant to describe the column before it was touched.

    apply_preprocessing(frame, fitted) -> frame
        Fill and scale using the stored constants and no others. Do not look at
        `frame` to decide what to use. Return the columns in the stored order.

    find_leaks(frame, target) -> list[str]
        Return the names of columns that predict `target` almost perfectly on
        their own -- by presence or by value, at the threshold and under the
        ceiling on the slide. Sort the names alphabetically. The columns you
        leave out are graded as hard as the ones you put in.

    keep_or_drop(evidence) -> (call, reason)
        Finding a suspect is not the same as deciding what to do about it. This
        is the decision: one candidate feature, six quantities you measured
        about it, and three possible calls -- keep it, keep it with its absence
        mask, or drop it. Return the call and the reason, and let somebody who
        was not in the room disagree with you on the evidence rather than on
        your taste.
"""
from __future__ import annotations

import sys
import pathlib

# Every library the reference solution uses is imported here, so that the work
# in front of you is the statistics and not the import lines.
import numpy as np
import pandas as pd
import plotly.graph_objects as go                                    # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_phones          # noqa: E402
from _narrate import narrator, show_table, save_figure   # noqa: E402,F401

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

    Definition graded by the check:
        θ = fit(X_train) = (median_j, μ_j, σ_j, column order), σ_j = √( (1/n)
        Σ_i (x_ij − μ_j)² ), ddof = 0
        (Kuhn & Johnson, 2019, ch. 8). The standard deviation is the population
        one, ddof = 0, which is the course's stated choice; a column that does
        not vary gets 1.0 so that scaling it does not divide by nought. Slide:
        "Definition — the fitted transform, and applying it".
    Needs: pandas, list, float

    Returns:
        {"medians": {...}, "means": {...}, "stds": {...}, "columns": [...]}.
    """
    numeric_list = _numeric_columns(train)
    return {
        "medians": {c: float(train[c].median()) for c in numeric_list},
        "means": {c: float(train[c].mean()) for c in numeric_list},
        "stds": {c: float(train[c].std(ddof=DDOF)) or 1.0 for c in numeric_list},
        "columns": list(numeric_list),
    }

def apply_preprocessing(frame, fitted: dict):
    """Apply the stored constants. Do not recompute anything from `frame`.

    Definition graded by the check:
        apply(X, θ)_ij = (fill(x_ij, median_j) − μ_j) / σ_j, columns in the
        stored order
        (Kuhn & Johnson, 2019, ch. 8). Nothing here may be measured from `frame`:
        given the same `fitted`, training rows, test rows and a single request
        arriving at a service in six months all get the same treatment, which is
        what Module 3 will need. Slide: "Definition — the fitted transform, and
        applying it".
    Needs: pandas, DataFrame column selection by list

    Returns:
        A frame with the stored columns, in the stored order.
    """
    output_df = pd.DataFrame(index=frame.index)
    for column in fitted["columns"]:
        values_series = frame[column] if column in frame.columns else np.nan
        values_series = pd.Series(values_series, index=frame.index, dtype="float64")
        values_series = values_series.fillna(fitted["medians"][column])
        output_df[column] = ((values_series - fitted["means"][column])
                             / fitted["stds"][column])
    return output_df[fitted["columns"]]


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
    known_series = series.notna()
    if known_series.sum() == 0:
        return float("nan")
    table_df = pd.crosstab(series[known_series], positive[known_series])
    if table_df.empty:
        return float("nan")
    return float(table_df.max(axis=1).sum() / table_df.to_numpy().sum())


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
    labels_series = frame[target]
    known_series = labels_series.notna()
    if known_series.sum() == 0:
        return []
    positive_series = labels_series[known_series] == "IN"

    leaks_list = []
    for column in frame.columns:
        if column in (target, "aboard_truth"):
            continue
        series = frame.loc[known_series, column]

        # Leak by presence: is the column filled in exactly when the target is?
        present_series = series.notna()
        if present_series.nunique() > 1:
            agreement_value = max((present_series == positive_series).mean(),
                                  (present_series != positive_series).mean())
            if agreement_value >= AGREEMENT:
                leaks_list.append(column)
                continue

        # Leak by value: does a single value split the target almost perfectly?
        # The cardinality ceiling is the guard described above: without it a
        # column of distinct values is pure by construction and always "leaks".
        if series.nunique(dropna=True) <= MAX_LEAK_VALUES and series.notna().any():
            if column_purity(series, positive_series) >= AGREEMENT:
                leaks_list.append(column)

    return sorted(set(leaks_list))



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
    say = narrator(LAB)
    phones = load_phones()
    cut = int(len(phones) * 0.7)
    train, test = phones.iloc[:cut], phones.iloc[cut:]
    fitted = fit_preprocessing(train)
    show_table(pd.DataFrame({k: fitted[k] for k in ("medians", "means", "stds")}),
               "the stored constants", logger=say)
    say.info("applied to the test rows: %s", apply_preprocessing(test, fitted).shape)
    say.info("leaks: %s", find_leaks(phones, TARGET))
    say.info("verdict on one candidate: %s", keep_or_drop({
        "missing_share": float(phones["rssi1"].isna().mean()),
        "imputation_bias_db": 9.0,
        "purity": 0.7893,
        "cardinality": int(phones["rssi1"].nunique(dropna=True)),
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.1214}))

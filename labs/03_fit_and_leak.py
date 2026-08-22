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
        later, which is the point.

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
    # TODO: compute medians, means, stds and the column order. Train only.
    raise NotSolved("fit_preprocessing(train) still raises instead of returning constants")


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
    # TODO: fill and scale using `fitted`, and return the stored column order.
    raise NotSolved("apply_preprocessing(frame, fitted) still raises instead of returning a frame")


def find_leaks(frame, target: str = TARGET) -> list:
    """Columns that agree with the target almost perfectly, by value or by presence.

    Definition graded by the check:
        column c leaks when agreement(c) ≥ 0.99, or purity(c) ≥ 0.99 with
        |values(c)| ≤ 10, where agreement(c) = max( mean(1[c present] = 1[Y =
        IN]), mean(1[c present] ≠ 1[Y = IN]) ) and purity(c) = Σ_v max_y n(v, y)
        / n
        (Kaufman et al., 2012; Kapoor & Narayanan, 2023). n(v, y) is the number
        of labelled rows where the column holds v and the target holds y; the
        threshold 0.99 is AGREEMENT above and the ceiling on the distinct values
        is MAX_LEAK_VALUES. The ceiling is part of the rule, not a tidying-up:
        purity is one by construction for a column whose values are all
        distinct, so without it every identifier in the file is a perfect leak
        for a reason that has nothing to do with the target. Presence is tested
        as well as value, because that is the form the leak takes here, and both
        directions count: a column that disagrees perfectly gives the game away
        as completely as one that agrees perfectly. Slide: "Definition — target
        leakage, and the rule that catches it here".
    Needs: pandas, sorted, set

    Returns:
        The leaking column names, sorted.
    """
    # TODO: check each column, by value and by presence. Return sorted names.
    raise NotSolved("find_leaks(frame, target) still raises instead of returning a list")


def keep_or_drop(evidence: dict) -> tuple:
    """The call on one candidate feature, and the reason for it.

    Definition graded by the check:
        verdict: E → {keep, keep with the mask, drop}, a total function of the
        six measured quantities alone, with reason ⊆ E: every number in the
        reason is a value in E
        (Kaufman et al., 2012; Kapoor & Narayanan, 2023). E is `evidence`, whose
        six keys are listed below. The order in which they are asked, and the
        choice printed beside each, are on the slide "Definition — the verdict
        on a candidate feature" and nowhere in this file: a verdict you can copy
        off the page you are writing on is not a verdict.
    Needs: dict, float, str

    `evidence` holds one candidate feature's numbers, every one of them measured
    by you earlier in this module:

        missing_share                  rows with no value, as a share of the rows
        imputation_bias_db             the mean of (fill − truth) over the rows
                                       the fill touched, in decibels, as Lab 2
                                       measured it; nought when nothing is filled
        purity                         Σ_v max_y n(v, y) / n on the training
                                       rows: the share of rows a rule answering
                                       with the commonest target value seen at
                                       each value of this column would get
                                       right. The same statistic find_leaks
                                       measures, and it has the same ceiling on
                                       it for the same reason
        cardinality                    how many distinct values the column holds
        knowable_at_decision_time      True when the value already exists at the
                                       instant the model has to answer
        in_sample_minus_out_of_sample  the score on the rows it was fitted on,
                                       minus the score on the rows it was not

    Returns:
        (call, reason).
        `call` is exactly one of "keep", "keep with the mask", "drop".
        `reason` is a sentence somebody who was not in the room can disagree
        with. It names at least two of the six quantities and quotes their
        values, and every number in it has to be one of those values — a number
        that came from a slide, a paper or a memory is rejected, which is the
        habit this course exists to break.
    """
    # TODO: ask the six in the order the slide gives, and build the reason from
    # `evidence` -- not from anything you remember.
    raise NotSolved("keep_or_drop(evidence) still raises instead of returning a call and a reason")


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

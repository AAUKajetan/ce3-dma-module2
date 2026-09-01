#!/usr/bin/env python3
"""Check 3 — the constants come from train, and the leak is found."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready, explain, grade_reason   # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    from lab_support import load_phones                               # noqa: E402
    import numpy as np                                                # noqa: E402
    import pandas as pd                                               # noqa: E402
    from sklearn.linear_model import LogisticRegression               # noqa: E402
    from sklearn.metrics import accuracy_score                        # noqa: E402
except ImportError as unready:
    not_ready(unready)

TARGET = "label2"
SEED = 20200122

# The ceiling on the distinct values in the purity test, from the definition
# slide and from the stub. It is graded here as well, by the `reading_id`
# control below: purity is one by construction for a column of distinct values,
# so a rule without the ceiling reports every identifier in the file.
MAX_LEAK_VALUES = 10


def with_controls(phones):
    """The frame, plus three columns that must NOT be reported as leaks.

    Until this existed, `find_leaks` could return every column in the table and
    score full marks: nothing anywhere asserted that a column was *not* flagged,
    so the laziest possible answer taught the opposite of the lesson. These three
    are the answer to that, and each is a different way of being innocent.

    reading_id      distinct on every row. Purity is one for it by construction
                    -- each value sees exactly one row, so the larger class
                    count for that value is one and the sum is the number of
                    rows -- whatever the target does. A purity test without the
                    cardinality ceiling reports it, and every other identifier,
                    as a perfect leak. This column is why the ceiling is in the
                    definition.
    beacon_band     strongly but legitimately predictive: a proximity band that
                    agrees with the target on about ninety-two per cent of rows
                    and is knowable at the moment of prediction. It is the
                    column a blanket detector destroys, and the reason a suspect
                    is not a verdict.
    battery_level   an ordinary measurement with ordinary gaps, absent on about
                    thirty per cent of rows at random. Absent often, and absent
                    for reasons that have nothing to do with the target.
    """
    planted = phones.copy()
    rng = np.random.default_rng(SEED)
    planted["reading_id"] = np.arange(len(planted))
    aboard = (planted[TARGET] == "IN").to_numpy()
    disagreeing = rng.random(len(planted)) < 0.08
    planted["beacon_band"] = np.where(aboard ^ disagreeing, 4, 1)
    battery = rng.integers(20, 100, len(planted)).astype(float)
    battery[rng.random(len(planted)) < 0.30] = np.nan
    planted["battery_level"] = battery
    return planted


# --------------------------------------------------------------------------
# The verdict: eight candidate features whose right calls are not the same
# --------------------------------------------------------------------------
# Every number in these dictionaries was measured on this module's own data and
# is written down here so that the fixture cannot drift with a regenerated file.
#
#   missing_share, cardinality      data/phones_2020-01-22.parquet. Both are
#                                   counted on the TRAINING rows -- the first 70
#                                   per cent in time order -- for the same reason
#                                   fit_preprocessing is: a quantity measured
#                                   over the test rows has already used them.
#   purity                          Σ_v max_y n(v, y) / n over the labelled
#                                   training rows: the share of rows a rule that
#                                   answers with the commonest target value seen
#                                   at each value of the column would get right.
#                                   Exactly the statistic find_leaks measures,
#                                   so the verdict and the detector cannot
#                                   disagree about what a column's purity is.
#   in_sample_minus_out_of_sample   the same column through an unrestricted
#                                   decision tree, seed 20200122: training
#                                   accuracy minus test accuracy across that
#                                   same split
#   imputation_bias_db              the two fills Lab 2 measures, from
#                                   slides/measured.json -> imputation_bias_db:
#                                   nine decibels for the masked moving average,
#                                   13.6 for the mean
#   knowable_at_decision_time       not a measurement but a fact about the
#                                   process that writes the column, which is why
#                                   it takes a conversation and not a query
#
# The point of the set is that no single quantity orders it. Four columns have a
# purity of one or all but one and their calls are drop, drop, drop and KEEP.
# `bus_id` has the highest purity of any of them and no gap across the split at
# all, and must go; `phone_speed` is at 0.9933, above the leak threshold, and
# must stay. A rule that reads one number, and a lookup table on any one of them,
# both fail.
CALLS = ("keep", "keep with the mask", "drop")

FIXTURES = {
    # Perfectly pure and unknowable: the leak. It is not overfitting -- the
    # in-sample and out-of-sample scores are the same, and both are perfect.
    "bus_id": {
        "missing_share": 0.4367, "imputation_bias_db": 0.0,
        "purity": 1.0, "cardinality": 1,
        "knowable_at_decision_time": False,
        "in_sample_minus_out_of_sample": 0.0},
    # A leak by value that survives the first question. Two values, purity one:
    # `stationary` is 1 on exactly the rows where the target is OUT. Asking only
    # "is it knowable?" would keep it.
    "stationary": {
        "missing_share": 0.0, "imputation_bias_db": 0.0,
        "purity": 1.0, "cardinality": 2,
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.0},
    # Purity one, and by construction: a row counter, distinct on every row.
    # The ceiling stops the leak rule firing, so something else has to condemn
    # it -- and something else does, across the split.
    "reading_id": {
        "missing_share": 0.0, "imputation_bias_db": 0.0,
        "purity": 1.0, "cardinality": 7559,
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.5069},
    # THE TRAP. Lab 4's own window mean of the phone's speed: 1,415 distinct
    # values over 1,500 training windows, so its purity is 0.9933 for the same
    # arithmetic reason the row counter's is one -- nearly every value sees one
    # row. It is above the leak threshold, it is complete, it is knowable, it
    # survives the split, and it is the strongest honest feature in the whole
    # hand-off table. The call is KEEP. A student who reads the purity and stops
    # throws away the best column they have.
    "phone_speed": {
        "missing_share": 0.0, "imputation_bias_db": 0.0,
        "purity": 0.9933, "cardinality": 1415,
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.0575},
    # Heavy missingness, a real fill bias, and still worth keeping -- with the mask.
    "rssi1": {
        "missing_share": 0.8874, "imputation_bias_db": 9.0,
        "purity": 0.7893, "cardinality": 279,
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.1214},
    # Heavier missingness, a larger bias, and values that say little on their
    # own: kept for the mask rather than for the value.
    "rssiC": {
        "missing_share": 0.6926, "imputation_bias_db": 13.6,
        "purity": 0.7257, "cardinality": 328,
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.0779},
    # Complete, knowable, harmless -- and barely above the base rate. Beacon B is
    # heard on about as many not-aboard rows as aboard ones.
    "proxB": {
        "missing_share": 0.0, "imputation_bias_db": 0.0,
        "purity": 0.5969, "cardinality": 4,
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.0971},
    # Entirely fine, and nothing near either threshold, so that "drop" is not a
    # safe default and "keep" is not reached only by the trap.
    "speed": {
        "missing_share": 0.0, "imputation_bias_db": 0.0,
        "purity": 0.8209, "cardinality": 2873,
        "knowable_at_decision_time": True,
        "in_sample_minus_out_of_sample": 0.0776},
}

EXPECTED = {"bus_id": "drop", "stationary": "drop", "reading_id": "drop",
            "phone_speed": "keep", "rssi1": "keep with the mask",
            "rssiC": "keep with the mask", "proxB": "drop", "speed": "keep"}

# Why each call is what it is, shown only from the third failure of that same
# fixture. The brief message says which fixture broke; this says which of the
# six settles it, and never which number it is compared against.
BECAUSE = {
    "bus_id": "This one is filled in by the same process that writes the target, so "
              "it does not exist at the instant the model has to answer. Ask that "
              "question first and nothing else about this column matters -- note that "
              "it does not even overfit: it scores the same in sample and out of it, "
              "and both are perfect. A leak's symptom is a good result, not a gap.",
    "stationary": "This one is knowable, complete and generalises, so the first two "
                  "things you would ask about a suspect column both clear it. It is "
                  "still the target under another name. Look at how pure it is, and "
                  "then at how many distinct values that purity is spread over.",
    "reading_id": "Its purity is one and always will be: a value that sees one row is "
                  "pure whatever the target does, so on a column this wide that number "
                  "carries no information at all and the leak rule is right not to "
                  "fire. Something else condemns this column. Look at what it scores "
                  "on the rows it was fitted on against the rows it was not.",
    "phone_speed": "Everything about this one is respectable, including the purity -- "
                   "which is high for the same arithmetic reason the row counter's is, "
                   "and not because the column knows the answer. Count the distinct "
                   "values before you read the purity as a confession. This is the "
                   "strongest honest feature in the hand-off table, and a verdict that "
                   "throws it away has cost the model more than any leak would.",
    "rssi1": "Most of this column is not there, and Lab 2 measured what the fill puts "
             "in its place. That does not make the column worthless -- it makes the "
             "record of which rows were invented part of the feature.",
    "rssiC": "The values of this one say little on their own. Where it is absent, "
             "however, is not random, which Lab 2 measured: the silence is the far "
             "readings. Ask what carries the information here before you ask whether "
             "the numbers do.",
    "proxB": "This one is complete, so nothing is invented and there is no absence for "
             "a mask to carry; and a rule that answers with the commonest target value "
             "at each of its four bands does barely better than answering with the "
             "majority every time.",
    "speed": "Nothing is wrong with this one on any of the six. A verdict that never "
             "says keep is not a verdict, it is a habit.",
}

# One quantity changed, and the call has to change with it. This is what stops a
# lookup table: the same feature name, the same five other numbers, a different
# answer. Two of these turn the cardinality ceiling, one each way.
PERTURBED = (
    # A control, and the only one here whose call does NOT move: a rule built on
    # the first question alone passes every other fixture in this set and fails
    # this one.
    ("bus_id", "knowable_at_decision_time", True, "drop",
     "Take away the thing that was deciding this column and it still goes. Its "
     "purity is one over a single distinct value, which is the other half of the "
     "leak rule. The first question is the first question, not the only one."),
    ("phone_speed", "cardinality", 4, "drop",
     "The same purity, now spread over four distinct values instead of a "
     "thousand-odd. At four values a purity that high is not arithmetic, it is a "
     "column that knows the answer."),
    ("stationary", "purity", 0.8209, "keep",
     "The same two-valued, knowable, complete column, and now its purity is well "
     "short of the threshold and well clear of the base rate."),
    ("reading_id", "in_sample_minus_out_of_sample", 0.0575, "keep",
     "Its purity is still one and its width is still what makes that meaningless. "
     "What condemned it has gone. If you still say drop, say which measurement "
     "says so."),
    ("rssi1", "in_sample_minus_out_of_sample", 0.5069, "drop",
     "The same column, and now it scores far better on the rows it was fitted on "
     "than on the rows it was not. That question is asked before the mask."),
    ("rssiC", "knowable_at_decision_time", False, "drop",
     "The first question is first for a reason: it overrules everything under it."),
    ("proxB", "purity", 0.7257, "keep",
     "The same complete, knowable column, and now a rule built on its four bands "
     "is well clear of the base rate."),
    ("speed", "missing_share", 0.6926, "keep with the mask",
     "The same column, now absent on more than half the rows: something has to "
     "record which rows were filled."),
)


def one_verdict(lab, label, evidence, expected, because, key):
    """Grade one call and one reason, and refuse a verdict that edits its evidence."""
    handed = dict(evidence)
    result = lab.keep_or_drop(handed)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"keep_or_drop() returned {result!r} for {label}; it returns a pair, "
        "(call, reason).")
    call, reason = result
    assert call in CALLS, (
        f"keep_or_drop() called {call!r} on {label}. The three calls are "
        f"{', '.join(repr(c) for c in CALLS)} — spelled exactly like that, because "
        "Module 3 reads them.")
    assert handed == evidence, (
        f"keep_or_drop() changed the evidence dictionary it was handed while judging "
        f"{label}. A verdict reads its evidence; it does not edit it.")
    assert call == expected, explain(
        key, f"on {label} you called {call!r}, and that is not the call", because)
    grade_reason(reason, evidence, key=key + ":reason", minimum_keys=2)
    return reason


def grade_keep_or_drop(lab):
    """Eight features whose right calls differ, then eight one-quantity changes."""
    for label, evidence in FIXTURES.items():
        reason = one_verdict(lab, label, evidence, EXPECTED[label], BECAUSE[label],
                             f"keep_or_drop:{label}")
        if label == "bus_id":
            # The one reason that has to name the quantity that settles it. A
            # correct call on this fixture with a reason about how well it scores
            # is the misunderstanding this whole lab exists to correct.
            assert "knowab" in reason.lower(), explain(
                "keep_or_drop:bus_id:knowable",
                "the call on bus_id is right and the reason does not mention whether "
                "the column is knowable at decision time",
                "This column's association is the best of the six and its in-sample "
                "gap is nought. If the reason rests on either of those, the right "
                "call was reached for a reason that would send you the wrong way on "
                "the next column. Say what actually settles it.")
        if label == "phone_speed":
            # The other half of the same lesson, and the trap this fixture set
            # exists for. `phone_speed` is above the leak threshold. What saves
            # it is the count of distinct values that purity is spread over, and
            # a reason that keeps it without naming that count has kept the best
            # feature in the table by luck.
            assert any(word in reason.lower()
                       for word in ("cardinal", "distinct", "values")), explain(
                "keep_or_drop:phone_speed:cardinality",
                "the call on phone_speed is right and the reason never mentions how "
                "many distinct values its purity is spread over",
                "This column's purity is 0.9933, which is above the threshold the "
                "leak rule uses. The only thing standing between it and a drop is "
                "that 1,415 distinct values over 1,500 rows make a high purity "
                "arithmetic rather than evidence. Say that, or the next wide column "
                "you meet goes in the bin.")

    for label, field, replacement, expected, because in PERTURBED:
        changed = dict(FIXTURES[label])
        changed[field] = replacement
        one_verdict(lab, f"{label} with {field} changed to {replacement!r}", changed,
                    expected, because, f"keep_or_drop:{label}:{field}")


def body(lab):
    phones = load_phones().sort_values("timestamp_utc").reset_index(drop=True)
    cut = int(len(phones) * 0.7)
    train, test = phones.iloc[:cut].copy(), phones.iloc[cut:].copy()

    fitted = lab.fit_preprocessing(train)
    for key in ("medians", "means", "stds", "columns"):
        assert key in fitted, f"fit_preprocessing() stored no '{key}'"
    assert fitted["columns"], "fit_preprocessing() stored an empty column list"

    # The test set, moved a long way, so apply_preprocessing below has something
    # to give itself away with. Nothing is refitted on it: fit_preprocessing(train)
    # returns the same answer however far the test set moves, so comparing one
    # call against another holds for every implementation ever written and can
    # never fail. Compare the stored constants against the training rows instead.
    moved = test.copy()
    for column in fitted["columns"]:
        if column in moved.columns:
            moved[column] = moved[column] + 1000

    # Every stored constant, not only the first, and not only the means: a fit
    # that reaches past `train` for one column and not for another would
    # otherwise go unnoticed.
    for column in fitted["columns"]:
        close(fitted["means"][column], float(train[column].mean()), 1e-6,
              f"means['{column}'] is not mean(train[column]) with the gaps still "
              "gaps. Two ways to get here: fitting on rows outside `train` -- the "
              "test set is supposed to stand in for data the model has never seen, "
              "and a mean computed over it has already been used to prepare the "
              "model's input -- or filling the gaps with a constant before taking "
              "the mean, which folds that constant into the very number meant to "
              "describe the column before it was touched.")
        close(fitted["medians"][column], float(train[column].median()), 1e-6,
              f"medians['{column}'] is not the training median. It is the constant "
              "that fills every gap in this column for the rest of the model's life, "
              "so it is estimated once, on the training rows, and stored.")
        population = float(train[column].std(ddof=0))
        close(fitted["stds"][column], population or 1.0, 1e-6,
              f"stds['{column}'] is not the training standard deviation with ddof = 0. "
              "The choice is stated on the slide and in the stub: the population "
              f"standard deviation, sigma = sqrt(mean of (x - mu)^2). The sample one "
              f"(ddof = 1) reads {float(train[column].std(ddof=1)):.6g} here and the "
              f"population one {population:.6g}; the difference is small, and a "
              "constant that differs between the model and the service is the whole "
              "of Module 3's skew.")

    reference_column = fitted["columns"][0]

    # Applying must use the stored constants, not recompute from what it is given.
    applied_test = lab.apply_preprocessing(test, fitted)
    applied_moved = lab.apply_preprocessing(moved, fitted)
    assert list(applied_test.columns) == list(fitted["columns"]), (
        "apply_preprocessing() did not return the stored column order. Column order is "
        "one of the constants; a service that reindexes differently is Module 3's skew.")
    shifted = (applied_moved[reference_column] - applied_test[reference_column]).mean()
    assert shifted > 1.0, (
        "moving the test set by 1000 barely moved the scaled output, which means "
        "apply_preprocessing() rescaled using the frame it was given rather than the "
        "stored constants.")

    # The leak.
    leaks = lab.find_leaks(phones, TARGET)
    assert isinstance(leaks, list), "find_leaks() must return a list"
    assert "bus_id" in leaks, (
        f"find_leaks() returned {leaks} and missed 'bus_id'. It is filled in exactly "
        "when the passenger is aboard — in the archive, 7,723 rows aboard all carry it "
        "and 6,000 rows not aboard all lack it, with no exceptions. It is not correlated "
        "with the target; it is the target under another name.")
    assert "stationary" in leaks, (
        f"find_leaks() returned {leaks} and missed 'stationary'. It is 1 on exactly the "
        "rows where the target is OUT and 0 on every row where it is IN — the target "
        "inverted. A column that disagrees perfectly gives the game away as completely "
        "as one that agrees perfectly, so check agreement both ways round.")
    assert "label" in leaks, (
        f"find_leaks() returned {leaks} and missed 'label'. It is the archive's own "
        "hand-recorded annotation -- the shuttle's name while aboard, a fixed value "
        "while not -- and it is filled in by exactly the same process that writes the "
        "target, the way bus_id is. It is not hidden from the frame the way "
        "aboard_truth is; a student who applies the 99 per cent rule to every column "
        "finds it the same way they find bus_id.")
    assert leaks == sorted(leaks), "find_leaks() must return the names sorted"

    # Negative controls on the shipped frame. Until these existed, `return
    # list(frame.columns)` scored full marks: every assertion above asks whether
    # a name is present and none asks whether a name is absent, so a detector
    # that flags everything passed while teaching the opposite of the lesson.
    for innocent, why in (
            ("speed", "an ordinary continuous measurement, and the honest "
                      "feature this lab's own score comparison is built on"),
            ("phone_id", "the identifier of the phone, which is present on every "
                         "row and so agrees with nothing"),
            ("prox1", "the proximity band of the vehicle beacon: four values, "
                      "informative, and nowhere near perfect")):
        assert innocent not in leaks, explain(
            f"find_leaks:innocent:{innocent}",
            f"find_leaks() reported {innocent!r} as a leak, and it is not one — "
            f"{why}",
            "A detector that reports everything reports nothing. Measure the rule "
            "on each column and return only the columns that meet it; the point of "
            "the exercise is the columns you leave out.")

    # And the two ways of being innocent that this frame does not contain, on a
    # frame that does. `reading_id` is the one the definition's cardinality
    # ceiling exists for.
    controls = with_controls(phones)
    flagged = lab.find_leaks(controls, TARGET)
    assert "bus_id" in flagged and "stationary" in flagged, (
        f"find_leaks() returned {flagged} on the frame with the controls added and "
        "lost a leak it had already found. Adding innocent columns cannot remove a "
        "guilty one.")
    assert "reading_id" not in flagged, explain(
        "find_leaks:reading_id",
        "find_leaks() reported 'reading_id' as a leak. It is a row counter: it is "
        "distinct on every row and it knows nothing about the target.",
        "Purity is one for it by construction. Each value appears on exactly one "
        "row, so the larger class count for that value is one, the sum over values "
        "is the number of rows, and the ratio is one — whatever the target does. "
        f"That is why the definition puts a ceiling of {MAX_LEAK_VALUES} distinct "
        "values on the purity test. Without the ceiling the rule reports every "
        "identifier, every timestamp and every free-text column in the file.")
    assert "beacon_band" not in flagged, explain(
        "find_leaks:beacon_band",
        "find_leaks() reported 'beacon_band' as a leak. It is the strongest honest "
        "feature in the frame, and it is knowable at the moment of prediction.",
        "It agrees with the target on about ninety-two per cent of rows, which is "
        "well under the ninety-nine per cent the rule asks for. A rule that flags "
        "the best legitimate feature you have is a rule that costs you the model. "
        "Compare the number you computed for it against the threshold before you "
        "add the name.")
    assert "battery_level" not in flagged, explain(
        "find_leaks:battery_level",
        "find_leaks() reported 'battery_level' as a leak. It is absent on about "
        "thirty per cent of rows, at random, and it agrees with nothing.",
        "Mostly empty is not the same as leaking. The presence test asks whether "
        "the column is filled in exactly when the target says one thing — not "
        "whether it is often empty. Measure the agreement, both ways round, and "
        "compare it against the threshold.")
    assert flagged == sorted(flagged), "find_leaks() must return the names sorted"

    # Asked again, on a frame whose answer cannot be memorised. `return
    # ["bus_id"]` finds the leak this lab is famous for without looking at
    # anything, and passes every assertion above. Here the column is gone, so a
    # remembered name is a name that is not in the frame, while `stationary` is
    # still there and still perfectly inverted.
    without_bus_id = phones.drop(columns=["bus_id"])
    remaining = lab.find_leaks(without_bus_id, TARGET)
    assert "bus_id" not in remaining, (
        f"find_leaks() returned {remaining} for a frame that has no 'bus_id' column "
        "at all. The answer has to be measured from the frame in front of it, not "
        "recalled from the one before.")
    assert "stationary" in remaining, (
        f"find_leaks() returned {remaining} once 'bus_id' was taken away, and missed "
        "'stationary'. Removing one leak does not remove the other.")
    assert remaining == sorted(remaining), "find_leaks() must return the names sorted"

    # And the reason it matters: with the leak the score is not real.
    def score(columns):
        features = phones[columns].copy()
        features["bus_id"] = phones["bus_id"].notna().astype(int) if "bus_id" in columns else 0
        numeric = features.select_dtypes("number").fillna(0.0)
        labels = (phones[TARGET] == "IN").astype(int)
        edge = int(len(numeric) * 0.7)
        # The seed is stated rather than left to the default. The solver used
        # here happens not to draw at random, so this changes nothing today --
        # which is exactly why it is worth pinning before somebody changes the
        # solver and the check starts reporting a different score each run.
        model = LogisticRegression(max_iter=400, random_state=SEED).fit(
            numeric[:edge], labels[:edge])
        return accuracy_score(labels[edge:], model.predict(numeric[edge:]))

    honest_columns = ["speed", "rssi1", "rssi2"]
    leaking_score = score(honest_columns + ["bus_id"])
    honest_score = score(honest_columns)
    grade_keep_or_drop(lab)
    assert leaking_score > honest_score, (
        f"the leaking model scored {leaking_score:.3f} and the honest one "
        f"{honest_score:.3f}. If the leak did not help, the check cannot demonstrate "
        "what it is for — tell the instructor.")


run(3, "03_fit_and_leak", "fit_preprocessing", body)

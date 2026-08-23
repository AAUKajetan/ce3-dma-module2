# Why these solutions look like this

The reference solutions are not the shortest code that turns a check green. They
are what the module is teaching, written so that reading one is worth as much as
writing one. Four decisions run through all four files.

**Every solution narrates.** `python3 solutions/lab_01.py` prints what it
loaded, whether that came from the archive or from the generator, every
intermediate quantity with its unit, and the one sentence the check grades. A
student who cannot get a lab to run can still read what the answer looks like,
and a teacher marking work runs `make demo` and reads one page instead of four
files. This is also how a number reaches a slide: the same functions that a
student's terminal calls are the ones `slides/make_figs.py` imports.

**Every choice is printed beside its number.** The grain is five seconds, the
smoothing weight is three tenths, the standard deviation is the population one
with the delta degrees of freedom at nought, the leak threshold is ninety-nine
per cent of the labelled rows. None of those is a fact about the world; each is a
decision, and a decision that is not written down is one nobody can review. The
same strings appear on the definition slides, in the stub docstrings and in
`concepts.json`, and `tools/check_definitions.py` fails the module if they drift.

**Nothing is imported that hides the lesson.** Lab 4's autocorrelation is
written out as the Box–Jenkins sum rather than delegated to
`pandas.Series.autocorr`, because the two are different estimators and the course
grades one of them. Lab 2's moving average is the plain recursion, so that a
student can step through a gap by hand and see that the smoothed value does not
move when nothing was heard.

**Each check defends against a specific way of being right by accident.** That is
worth naming lab by lab.

## Lab 1 — the ledger balances on a day where rows are actually lost

On the shipped pair the vehicle reports right across the phone window, so
nothing has to be dropped and a ledger of hard-coded noughts balances exactly as
well as a measured one. So the check asks again on a day where the vehicle stops
half way through, where roughly half the phone rows have nowhere to go and the
books have to be kept for the sum to come out right.

The clock trap is graded on both sides. Flooring the column named `timestamp`
gives windows that still land on a five-second boundary — they are simply an
hour away from the data they claim to describe — so the check bounds the windows
inside the span the phones cover as well as checking the boundary.

`pooled_versus_by_group` is graded three times: once on the planted frame, where
the honest answer is that every shuttle moves against the pool; and twice where
the honest answer is that nothing reverses, on a single-shuttle frame and on a
frame that rises everywhere. A verdict that is always `True` says nothing about
any frame, and the two controls are what turn the function from a slogan into a
measurement.

### Lab 1, the other half — the profile that arrives with the frame

`align` runs Module 1's declaration over the vehicle frame **before it joins
anything**, and records what came back in the ledger. The obvious cheat is to
record an empty list, because on the frame the lab actually ships the declaration
is silent — so the check asks on three frames and computes each expected answer by
calling `check_against` itself. On 22 January alone the declaration is not silent:
`emergency_stop` is absent on more of that day's rows than Module 1 tolerated,
because the tolerance was measured over both days pooled.

That is worth stopping on. **The pooled frame satisfies a rule its own first day
violates** — which is the paradox in the other half of this lab, arriving in a
validation layer instead of in a comparison, and it is the concrete reason the bias
examination the Artificial Intelligence Act requires has to be made per group.

## Lab 2 — the recursion, not merely "a value that varies"

The first version of this check asked only that the filled values were not all
the same. A random number generator passes that. It now compares the fills
against the recursion itself, computed in the check in plain Python, on a
planted two-phone trace where the answers can be worked out on paper — including
at the opening of the second phone's trace, where nothing has been heard yet and
there is no state to carry.

`imputation_bias` is graded as arithmetic on three planted triples, one of which
has large values on the rows the method did *not* fill. Averaging over every row
instead of over the filled ones gives fifty point seven five instead of one point
five, and the difference is the whole definition.

The direction question is last, because it is the one a student should be able
to answer before writing any code: the readings you can see are the near ones, so
anything built from them and used to stand in for a far one comes out too strong.

## Lab 3 — the constants, all of them, and a leak that cannot be recalled

Refitting on a moved test set and comparing one call against another can never
fail: `fit_preprocessing(train)` returns the same answer however far the test set
moves. So the check compares every stored median, mean and standard deviation
against the training rows directly, and separately proves that
`apply_preprocessing` used the stored constants by moving a frame by a thousand
and watching the output move with it.

The standard deviation is graded with the delta degrees of freedom at nought,
which is the course's stated choice, and the failure message quotes the sample
value beside the population one so that a student who chose the other one sees
immediately what happened rather than hunting for a bug.

`return ["bus_id"]` finds the famous leak without looking at anything, so the
check asks again on a frame with that column removed, where `stationary` — the
target inverted — is still there and still perfectly informative. Agreement is
tested in both directions for the same reason: a column that disagrees perfectly
gives the game away as completely as one that agrees perfectly.

### Lab 3, the verdict — where the ceiling earns its place

`find_leaks` used to be gradeable by returning every column: every assertion asked
whether a name was *present* and none asked whether a name was *absent*. Six
negative controls now say otherwise — an ordinary measurement, an identifier, a
proximity band, a row counter, a mostly empty battery reading, and the strongest
honest feature in the frame — and the last two are the ones that matter. **A
detector with no false-positive test is not a detector.**

`keep_or_drop` is graded on eight candidate features, and four of them have a
purity of one or all but one. Three go and one stays:

- `bus_id` — not knowable at the moment of prediction. Nothing else about it is
  weighed, and nothing else needs to be.
- `stationary` — knowable, complete, and it generalises. It is still the target
  inverted, and only the purity test at low cardinality catches it.
- a row counter — purity one, and the cardinality ceiling correctly stops the leak
  rule firing on it, so something else has to condemn it. Something else does: it
  scores far better on the rows it was fitted on than on the rows it was not.
- `phone_speed`, Lab 4's own window mean — purity 0.9933 over 1,415 distinct
  values, which is above the leak threshold. **It is kept.** A high purity over a
  wide column is arithmetic and not a confession, and a rule without the ceiling
  throws away the best feature in the hand-off table.

The reason is graded as hard as the call. Every number written in it has to be one
of the numbers handed over, and at least two of the quantities weighed have to be
named — so a sentence recited off the definition slide fails on the first number it
quotes, however right the call.

## Lab 4 — a window with arithmetic answers, and a join with something to measure

Against the shipped vehicle every phone row matches at one second, so every
tolerance reads a hundred per cent and a function returning one constant
reproduces the curve exactly. The check therefore asks again against a vehicle
thinned to one reading every thirty seconds, where the share has to climb from
under seven per cent to all of them, and compares against a reference join.

The six window features are graded twice: against the check's own arithmetic on
the shipped speeds, and against the planted window one, two, three, four, five,
whose mean is three, whose population standard deviation is the square root of
two, whose slope is one and whose lag-one autocorrelation is exactly two fifths.
A table of numbers copied off a slide has nothing to copy.

The split is graded on a table handed over *shuffled*, with a fixed seed. The
shipped table already arrives in time order, so a split that cuts it where it
stands and never sorts passes on it — and the one line the exercise is about goes
ungraded.

### Lab 4, the hand-off — the clause that had no assertion

`assemble` builds the object Modules 3, 4 and 5 open. Its shape, its mask columns,
its recorded split point and its stored transform were all graded — and until this
pass **no assertion looked at a value**, so a table that carried the raw columns
through untouched passed every one of them. The values are now compared against the
stored constants applied, to within 1e-9; and because an assertion like that is
worthless if the two candidate answers happen to coincide, the check first proves
that a fit over the whole frame would give a measurably different result before it
uses the difference to grade anybody.

That is also where "fitted on the training rows only" stops being a claim about the
argument passed in and becomes a property of the object handed over. Recomputing the
constants from the table in front of you uses the test rows to prepare the model's
input; a service that recomputes them again from one live request gets a third
answer. That failure has a name — training-serving skew — and Module 3 spends a
block causing it deliberately.

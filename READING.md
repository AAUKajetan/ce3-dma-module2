# Module 2 — Reading, and the exam question

## Required, before Module 3

**Kapoor, S. & Narayanan, A. (2023). *Leakage and the reproducibility crisis in
machine-learning-based science*. Patterns 4(9), 100804.**
<https://doi.org/10.1016/j.patter.2023.100804> — open access.

They surveyed published machine-learning results across seventeen fields and
found leakage in hundreds of papers, under many different names. The taxonomy is
in the section headed *"Toward a solution: a taxonomy of data leakage"* — read
that part with `bus_id` from Lab 3 in mind and decide which of their categories
it belongs to. It is the most useful thing you will read this term.

**Rubin, D. B. (1976). *Inference and missing data*. Biometrika 63(3), 581–592.**
<https://doi.org/10.1093/biomet/63.3.581> — through the AAU library.

Six pages, and the origin of the three mechanisms Lab 2 turns on. Read it for
the conditional statements themselves: missing completely at random is
P(M | Y_obs, Y_mis) = P(M), missing at random is P(M | Y_obs, Y_mis) =
P(M | Y_obs), and missing not at random is neither. Everything the course says
about imputation follows from which of those three you are in.

## Recommended

**van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed. Chapman
and Hall/CRC Press.** <https://doi.org/10.1201/9780429492259> — and free to read
at <https://stefvanbuuren.name/fimd/>. The practical companion to Rubin: what
each imputation method assumes, what it costs, and why a single filled value
without a record that it was filled is the worst of the options.

**Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing
Data*, 3rd ed. Wiley.** <https://doi.org/10.1002/9781119482260> — through the
library. The three mechanisms stated properly rather than sloganised, and the
bias of a fill written down as an expectation rather than asserted.

**Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). *Leakage in data
mining: formulation, detection, and avoidance*. ACM Transactions on Knowledge
Discovery from Data 6(4), article 15.**
<https://doi.org/10.1145/2382577.2382579> — through the AAU library. The
original formulation, and still the clearest: leakage as information about the
target that was not legitimately available at prediction time.

**Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality
categorical attributes in classification and prediction problems*. SIGKDD
Explorations 3(1), 27–32.** <https://doi.org/10.1145/507533.507538> — through
the library. Where target encoding comes from. Read it for the smoothing, and
then read the encoding slide again: the danger is entirely in *when* the average
is computed.

**Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection: a Practical
Approach for Predictive Models*. Chapman and Hall/CRC Press.**
<http://www.feat.engineering/> — free online edition. Chapter 8 is imputation
and chapter 5 is encoding. (Earlier versions of this list sent you to chapter 3
for both. They were wrong.)

**Roberts, D. R., Bahn, V., Ciuti, S. et al. (2017). *Cross-validation
strategies for data with temporal, spatial, hierarchical, or phylogenetic
structure*. Ecography 40(8), 913–929.**
<https://doi.org/10.1111/ecog.02881> — open access. Why a random split flatters
a model whenever the rows are not independent, and what to do instead. Read the
block cross-validation section beside Lab 4.

**Bergmeir, C. & Benítez, J. M. (2012). *On the use of cross-validation for time
series predictor evaluation*. Information Sciences 191, 192–213.**
<https://doi.org/10.1016/j.ins.2011.12.028> — through the library. The same
argument for time series specifically, with the experiments.

**Simpson, E. H. (1951). *The interpretation of interaction in contingency
tables*. Journal of the Royal Statistical Society, Series B 13(2), 238–241.**
<https://doi.org/10.1111/j.2517-6161.1951.tb00088.x> — through the library.
Four pages. The reversal Lab 1 grades, in the original. The name came later,
from **Blyth, C. R. (1972)**, *On Simpson's paradox and the sure-thing
principle*, Journal of the American Statistical Association 67(338), 364–366,
<https://doi.org/10.1080/01621459.1972.10482387>.

**Pearl, J. (2014). *Comment: understanding Simpson's paradox*. The American
Statistician 68(1), 8–13.** <https://doi.org/10.1080/00031305.2014.876829> —
through the library. The answer to the question the reversal raises: which of
the two comparisons should you act on? Pearl's answer is that the data alone
cannot say, and what settles it is what caused the group mix to differ.

**Elvik, R. (2025). *Simpson's paradox: a collection of examples from road
safety studies and emergency medicine*. Transportation Research
Interdisciplinary Perspectives 31, 101471.**
<https://doi.org/10.1016/j.trip.2025.101471> — open access. Read it if you
suspect the paradox is a textbook curiosity. These are real road-safety
evaluations in which the pooled answer and the per-group answer disagree about
whether a measure saved lives.

**Servizi, V., Persson, D. R., Pereira, F. C., Villadsen, H., Bækgaard, P.,
Peled, I. & Nielsen, O. A. (2023). *"Is Not the Truth the Truth?": Analyzing the
Impact of User Validations for Bus In/Out Detection in Smartphone-Based
Surveys*. IEEE Transactions on Intelligent Transportation Systems 24(11),
11905–11920.** <https://doi.org/10.1109/TITS.2023.3291493> — through the AAU
library. This is the study the archive comes from, and the source of the masked
exponential moving average Lab 2 implements. Read it for what the volunteers'
own validations did and did not settle.

**Servizi, V., Persson, D. R., Pereira, F. C., Villadsen, H., Bækgaard, P.,
Rich, J. & Nielsen, O. A. (2026). *Scalable Passenger Detection Using
Smartphone–Bus Implicit Interactions*. IEEE Intelligent Transportation Systems
Magazine 18(1), 65–78.** <https://doi.org/10.1109/MITS.2025.3611306> — through
the library. What the same team built afterwards, at scale. Read it to see the
pipeline of this module in service.

**Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). *Time
Series Analysis: Forecasting and Control*, 5th ed. Wiley.** Through the library.
Section 2.1.4 is the sample autocorrelation this course grades, in Lab 4 and in
Module 1: one mean over the window, and the window's own sum of squares
underneath.

**Akidau, T., Bradshaw, R., Chambers, C. et al. (2015). *The dataflow model: a
practical approach to balancing correctness, latency, and cost in massive-scale,
unbounded, out-of-order data processing*. Proceedings of the VLDB Endowment
8(12), 1792–1803.** <https://doi.org/10.14778/2824032.2824076> — open access.
Where the window vocabulary of Lab 1 comes from: fixed (tumbling), sliding and
session windows, defined precisely and for a reason.

**McKinney, W. (2022). *Python for Data Analysis*, 3rd ed.**
<https://wesmckinney.com/book/> — free online edition. The joining and reshaping
chapter is the practical companion to Lab 1, and the time-series chapter is
where `merge_asof` lives.

**Nielsen, A. (2019). *Practical Time Series Analysis*. O'Reilly.** Through the
library. The window-feature vocabulary, with code.

**Gowda, K., Ping, D., Mani, M. & Kuehn, S. (2022). *Genomic structure predicts
metabolite dynamics in microbial communities*. Cell 185(3), 530–546.**
<https://doi.org/10.1016/j.cell.2021.12.036> — through the AAU library. Read the
model-evaluation section only, and ignore the biology. It reports the in-sample
and out-of-sample performance of the *same* model, side by side, honestly: the
number roughly halves. Lab 3 taught leakage as a defect you remove. This is the
ordinary case with no defect at all, where the gap is simply what generalisation
costs, and where a paper that reported only the first number would not have been
lying — only useless.

> Nothing licensed is redistributed in this repository. Where a reading is a
> commercial book you get a library link and a chapter number.

## The exam question for Module 2

> **You are given a table someone else cleaned. It has no missing values. What
> do you ask them, and why? Then describe one defect that would survive every
> question you just asked, and how you would find it.**

A strong answer notices that "no missing values" is a claim about the *output*
and says nothing about what was done to get there: were rows dropped, were
values filled, and with what? It asks for the decision ledger and the masks. It
distinguishes a value that is absent from a record that never arrived. And for
the surviving defect it reaches for leakage — a table can be complete, typed,
consistent and internally perfect while containing a column that already knows
the answer, and no amount of quality measurement will reveal it. Finding it
takes the structural question: when does each field become known?

A very strong answer adds a second surviving defect: a table can be complete and
leak-free and still support two opposite conclusions, depending on whether you
compare groups or the pool they came from. Ask what each row is, what each group
is, and what changed between the periods being compared.

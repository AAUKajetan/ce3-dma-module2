"""The check harness.

Four outcomes, and the exit code says which:

    0   the lab is written and correct
    2   the lab has not been written yet
    1   the lab has been written and is wrong
    3   the environment is not ready, so nothing could be graded

Code 2 is deliberate and separate. "Not started" and "broken" are different
states and a student deserves to be told which one they are in. Every message
names the file, the function and the lab number, so that a red terminal is a
next action rather than a mood.

Code 3 is separate for the same reason. A missing library or a missing trained
artefact is the machine's fault, and reporting it as code 1 sends a student
hunting for a bug in code that is correct.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))

GREEN, RED, AMBER, PLAIN = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def not_ready(problem) -> None:
    """Exit 3. Called when an import failed before any grading could begin.

    A check imports pandas, numpy or scikit-learn at the top of the file, before
    run() can wrap anything in a try. Guard those imports with this and a student
    on an unprepared machine is told to run setup.sh, rather than being told the
    lab they have not opened yet is wrong.
    """
    print(f"{AMBER}[environment] your environment is not ready — "
          f"re-run `bash setup.sh`{PLAIN}")
    print(f"  problem:  {problem}")
    sys.exit(3)


# One definition, imported — not a second class with the same name. Two classes
# named NotSolved means `except NotSolved` never matches what the labs raise, and
# "not written yet" is reported as a crash. This exact bug shipped in 2025.
#
# Guarded because lab_support imports pandas, so this line is the first place an
# unprepared machine breaks — earlier than anything the checks import themselves.
try:
    from lab_support import NotSolved, EnvironmentNotReady  # noqa: E402
except ImportError as unready:
    not_ready(unready)


def load(lab_number: int, module_name: str):
    """Import one lab by file name, without needing it to be a package."""
    import importlib.util

    path = REPOSITORY / "labs" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crashed_in(module_name: str, function_name: str) -> str:
    """Where the crash really happened, which is not always the graded function.

    A check calls several of a lab's functions, but run() is told the name of
    only one of them. Printing that one above every traceback sends a student to
    a function that never ran. So walk the traceback for the deepest frame inside
    the lab file and name that instead.
    """
    deepest = function_name
    for frame in traceback.extract_tb(sys.exc_info()[2]):
        if Path(frame.filename).name == f"{module_name}.py":
            deepest = frame.name
    return f"labs/{module_name}.py → {deepest}()"


def run(lab_number: int, module_name: str, function_name: str, body):
    """Run one check. `body` receives the imported lab module and asserts."""
    label = f"Lab {lab_number}"
    where = f"labs/{module_name}.py → {function_name}()"
    try:
        module = load(lab_number, module_name)
        body(module)
    except NotSolved as unwritten:
        # crashed_in() already walks the traceback for the deepest frame inside the
        # lab file; a NotSolved deserves the same treatment. Printing the one name
        # run() was handed sent a student in a five-function lab to a function that
        # had nothing to do with the message underneath it.
        print(f"{AMBER}[{label}] not written yet{PLAIN}")
        print(f"  file:     {crashed_in(module_name, function_name)}")
        print(f"  what:     {unwritten}")
        print(f"  fix:      open labs/{module_name}.py, replace the TODO, "
              f"then run  make check{lab_number}")
        sys.exit(2)
    except (EnvironmentNotReady, ImportError) as unready:
        not_ready(unready)
    except AssertionError as wrong:
        print(f"{RED}[{label}] written, but not right yet{PLAIN}")
        print(f"  file:     {where}")
        print(f"  problem:  {wrong}")
        print(f"  fix:      re-read the docstring in labs/{module_name}.py, "
              f"then run  make check{lab_number}")
        sys.exit(1)
    except Exception:  # a crash is not a pass; say so, and show the traceback
        print(f"{RED}[{label}] crashed{PLAIN}")
        print(f"  file:     {crashed_in(module_name, function_name)}")
        traceback.print_exc()
        print(f"  fix:      make it run at all, then run  make check{lab_number}")
        sys.exit(1)
    print(f"{GREEN}[{label}] green{PLAIN}  {where}")
    sys.exit(0)


def close(measured, expected, tolerance, what):
    """Assert two numbers agree, and say by how much they did not."""
    difference = abs(measured - expected)
    assert difference <= tolerance, (
        f"{what}: you gave {measured:.6g}, the check measured {expected:.6g}, "
        f"a difference of {difference:.6g} (allowed {tolerance:g})")


# --------------------------------------------------------------------------
# Grading a judgement, not an arithmetic
# --------------------------------------------------------------------------
# Every module now asks for one verdict: a call, and the reason for it. The call
# is a small closed set, so on its own it is a coin flip -- and a coin flip that
# the failure message used to explain. Two rules make it an examination again:
#
#   1. the reason must cite the student's OWN measurements. Every number written
#      in it has to match a value in the evidence to the digit, so a reason
#      cannot be recited from a slide;
#   2. it must name at least two of the quantities it weighed, so "it is large"
#      does not pass for an argument.
#
# And `explain()` withholds the reasoning until the third attempt. A check that
# gives the answer away the first time it is wrong is a hill to climb, not a
# question to answer.

import json as _json
import re as _re

_ATTEMPTS = REPOSITORY / "labs" / ".attempts.json"
_NUMBER = _re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def attempts(key: str, record: bool = True) -> int:
    """How many times this student has failed this particular assertion."""
    try:
        counts = _json.loads(_ATTEMPTS.read_text())
    except Exception:
        counts = {}
    seen = int(counts.get(key, 0))
    if record:
        counts[key] = seen + 1
        try:
            _ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
            _ATTEMPTS.write_text(_json.dumps(counts, indent=1, sort_keys=True))
        except OSError:
            pass
    return seen


def explain(key: str, brief: str, full: str, after: int = 2) -> str:
    """The short message first; the reasoning only once the student is stuck.

    `brief` says which property broke. `full` says why, and appears from the
    third failure of this same assertion onwards -- or immediately if the
    attempt file cannot be written, because a check that cannot count must not
    also refuse to help.
    """
    if attempts(key) >= after:
        return f"{brief}\n            {full}"
    return f"{brief}\n            (the reasoning appears here after two more attempts, " \
           f"or read solutions/WHY.md once you are done)"


def numbers_in(text: str):
    """Every number a student wrote in their reason, as floats."""
    found = []
    for piece in _NUMBER.findall(str(text)):
        try:
            found.append(float(piece.replace(",", ".")))
        except ValueError:
            pass
    return found


def grade_reason(reason, evidence: dict, key: str, minimum_keys: int = 2,
                 tolerance: float = 5e-3) -> None:
    """The reason has to be built out of the student's own measurements.

    Raises AssertionError, which run() reports as exit 1.
    """
    text = str(reason).strip()
    assert len(text) >= 40, explain(
        key + ":length",
        f"the reason is {len(text)} character(s) long; a verdict needs an argument",
        "Name the quantities you weighed and give their values. Somebody who was "
        "not in the room has to be able to disagree with you on the evidence.")

    values = []
    for value in evidence.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (list, tuple)):
            values += [float(v) for v in value
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]
    written = numbers_in(text)
    assert written, explain(
        key + ":numbers",
        "the reason contains no numbers at all",
        "A verdict without a magnitude is an opinion. Quote the measurements you "
        "made -- the ones in the evidence you were handed.")
    for number in written:
        assert any(abs(number - value) <= tolerance * max(1.0, abs(value))
                   for value in values), explain(
            key + ":unmatched",
            f"the reason quotes {number:g}, which is not any value in the evidence",
            "Every number in the reason has to be one you measured. A number that "
            "came from a slide, a paper or a memory is exactly the habit this "
            "course exists to break.")

    # "noise_floor" should match "noise floor" and "target_index" should match
    # "the target's index", so the underscore stands for any short glue.
    def _mentioned(name: str) -> bool:
        parts = [_re.escape(word) for word in name.split("_") if word]
        # "target index", "target_index" and "the target's index" all count.
        # "target index", "target_index" and "the target's index" all count.
        # The separator class must include the underscore, which \W does not.
        glue = r"[^A-Za-z0-9]*(?:[a-z]{1,2}[^A-Za-z0-9]+)?"
        pattern = r"\b" + glue.join(parts) + r"\b"
        return bool(_re.search(pattern, text, _re.I))

    named = sum(1 for name in evidence if _mentioned(name))
    assert named >= minimum_keys, explain(
        key + ":keys",
        f"the reason names {named} of the quantities it weighed; at least "
        f"{minimum_keys} are needed",
        "A verdict rests on a comparison. Name what you compared against what -- "
        "the measurement and the thing that makes it large or small.")

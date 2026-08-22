#!/usr/bin/env bash
# One command, from nothing, in an empty container. If this fails, nothing else matters.
set -euo pipefail

# Two things go wrong on a Debian-packaged Python, and both stop the script dead
# under `set -e` before anything useful happens. Neither is the student's fault
# and neither is worth failing over.
#
#   upgrading pip  refused with "RECORD file not found" when pip came from apt
#   installing     refused with "externally-managed-environment" (PEP 668)
#
# So the upgrade is advisory, and the install falls back to the flag Debian
# wants. Inside the devcontainer neither fallback is reached.
#
# What pip actually said is now printed. It used to be sent to /dev/null and
# replaced by the sentence below, so a container with no network told the whole
# room that their pip was system-managed, which it was not.
if ! python3 -m pip install --quiet --upgrade pip; then
  echo "note: pip was not upgraded, and the reason is printed above. If it is"
  echo "      'externally-managed-environment' or 'RECORD file not found', pip"
  echo "      belongs to the system, this is expected, and setup continues."
fi

if ! python3 -m pip install --quiet -r requirements.txt; then
  echo "note: retrying with --break-system-packages, which a Debian-packaged"
  echo "      Python asks for. If the install failed for any other reason, it"
  echo "      is printed above and the retry will fail too."
  python3 -m pip install --quiet --break-system-packages -r requirements.txt
fi

# What this script promised, checked rather than assumed. A setup that prints
# "Ready" over a failed install sends a student into a lab that cannot run, and
# the check then reports a problem that is not theirs.
python3 -c "import pandas, numpy, pyarrow, sklearn, plotly" || {
  echo "setup failed: the requirements did not install —"
  echo "importing pandas, numpy, pyarrow, sklearn, plotly failed. Read pip's message above."
  exit 1; }
ls labs/*.py >/dev/null 2>&1 || {
  echo "setup failed: labs/ contains no lab files."
  exit 1; }

# Every dataset the labs read is generated (or, for the shipped slice, verified)
# here, once, and recorded in data/MANIFEST.json. `make check` starts by
# re-verifying that manifest, so "your data is not there" is reported before
# "your code is wrong" -- exit 3, not exit 1.
python3 data/prepare.py || {
  echo "setup failed: data/prepare.py could not prepare the datasets. Read its message above."
  exit 1; }

echo
echo "Ready. Run 'make check' to see where you are."
echo "Four labs are in labs/. Each has one function to write, marked TODO."

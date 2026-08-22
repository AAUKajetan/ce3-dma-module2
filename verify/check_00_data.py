#!/usr/bin/env python3
"""Check 0 -- are the datasets and artefacts the labs need in place?

    python3 verify/check_00_data.py        exit 0: everything setup.sh promised is here
                                           exit 3: something is missing -- run  bash setup.sh

`setup.sh` runs `python3 data/prepare.py`, which generates every dataset the labs
read (or verifies the ones that ship), trains any artefacts, and writes
data/MANIFEST.json: one entry per file with its row count, column count and a
content hash. This check reads that manifest and re-verifies each entry, so a
student is told "your data is not there" before being told "your code is
wrong" -- the two are different states and the exit codes keep them apart
(0 green, 1 wrong, 2 not written, 3 environment not ready).
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = HERE / "data" / "MANIFEST.json"

RED, GREEN, END = "\033[31m", "\033[32m", "\033[0m"


def not_ready(reason: str) -> int:
    print(f"{RED}[data] not ready{END}  {reason}")
    print("  fix:      run  bash setup.sh  and then  make check")
    return 3


def content_hash(frame) -> str:
    """A hash of the values, not of the bytes: parquet metadata carries a writer
    string and a timestamp, so two identical tables can differ byte for byte."""
    import pandas as pd
    return f"{int(pd.util.hash_pandas_object(frame, index=False).sum()):x}"


def main() -> int:
    if not MANIFEST.exists():
        return not_ready("data/MANIFEST.json is missing -- setup.sh has not run in this checkout")
    try:
        import pandas as pd
    except ImportError:
        return not_ready("pandas is not importable -- the requirements did not install")

    manifest = json.loads(MANIFEST.read_text())
    verified = []
    for entry in manifest.get("files", []):
        path = HERE / entry["path"]
        if not path.exists():
            return not_ready(f"{entry['path']} is missing")
        if entry.get("kind") == "table":
            reader = pd.read_parquet if path.suffix == ".parquet" else pd.read_csv
            frame = reader(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
            if len(frame) != entry["rows"] or frame.shape[1] != entry["columns"]:
                return not_ready(
                    f"{entry['path']} has {len(frame)} rows x {frame.shape[1]} columns; "
                    f"setup wrote {entry['rows']} x {entry['columns']}. Re-run setup.")
            if entry.get("content_hash") and content_hash(frame) != entry["content_hash"]:
                return not_ready(
                    f"{entry['path']} differs from what setup.sh generated (content hash). "
                    "Something rewrote it; re-run setup.")
        elif entry.get("kind") == "artefact":
            if path.stat().st_size == 0:
                return not_ready(f"{entry['path']} is empty")
        verified.append(entry["path"])
    if not verified:
        return not_ready("data/MANIFEST.json lists no files -- setup.sh did not finish")
    print(f"{GREEN}[data] ready{END}  {len(verified)} file(s) verified against data/MANIFEST.json: "
          + ", ".join(verified))
    return 0


if __name__ == "__main__":
    sys.exit(main())

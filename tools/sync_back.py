#!/usr/bin/env python3
"""Map this module repository's tree back onto the monorepo, for a reverse-sync
pull request. Additive/update only — see publish_module.py's SYNC_BACK_SCRIPT
docstring in the monorepo for why deletions are never propagated this way."""
import os
import shutil
import subprocess
from pathlib import Path

MODULE = os.environ["MODULE_NUMBER"]
MONOREPO = Path(os.environ["MONOREPO_CHECKOUT"])
HERE = Path(__file__).resolve().parent.parent

SKIP_TOP = {"README.md", ".git", ".github"}


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=HERE, capture_output=True, text=True, check=True,
    )
    return [p for p in result.stdout.split("\0") if p]


def main():
    changed = []
    for rel in tracked_files():
        top = rel.split("/", 1)[0]
        if top in SKIP_TOP:
            continue
        if top == "notebook":
            dest_rel = f"Module {MODULE}/notebook/" + rel.split("notebook/", 1)[1]
        else:
            dest_rel = f"Module {MODULE}/exercises/{rel}"
        src = HERE / rel
        dest = MONOREPO / dest_rel
        if dest.exists() and dest.read_bytes() == src.read_bytes():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        changed.append(dest_rel)

    print(f"{len(changed)} file(s) changed")
    for c in changed:
        print(f"  {c}")
    Path(os.environ["CHANGED_COUNT_FILE"]).write_text(str(len(changed)))


if __name__ == "__main__":
    main()

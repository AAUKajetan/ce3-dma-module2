#!/usr/bin/env python3
"""Copy the shipped solutions over the labs — to read, or to recover.

    python3 apply.py            put the solutions in place
    python3 apply.py --restore  put the exercises back

Your own attempt is saved to labs/.your_attempt/ first, so nothing is lost.
"""
import pathlib, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
PAIRS = {
    "01_audit_and_align.py": "lab_01.py",
    "02_the_mechanism.py": "lab_02.py",
    "03_fit_and_leak.py": "lab_03.py",
    "04_windows_and_cost.py": "lab_04.py",
}
BACKUP = HERE / "labs" / ".your_attempt"


def main():
    restore = "--restore" in sys.argv
    nothing = []
    BACKUP.mkdir(parents=True, exist_ok=True)
    for lab_name, solution_name in PAIRS.items():
        lab = HERE / "labs" / lab_name
        if restore:
            saved = BACKUP / lab_name
            if saved.exists():
                shutil.copy2(saved, lab)
                print(f"restored {lab_name}")
            else:
                # Silence here used to read as success: nothing was restored, the
                # solution stayed in labs/, and the closing line still told the
                # student to run --restore to get their work back.
                nothing.append(lab_name)
                print(f"NOTHING SAVED for {lab_name} — labs/.your_attempt has no copy "
                      f"of your file. It is written the first time you run apply.py.")
            continue
        source = (HERE / "solutions" / solution_name).read_text()
        # Save whatever is in labs/ now, unless it is already the applied
        # solution -- then there is nothing of yours to save and overwriting the
        # backup would lose it. The old rule was "back up only if no backup
        # exists", which silently kept the first backup for ever: apply, restore,
        # write your own answer, apply again, restore, and your answer was
        # replaced by the file you started from. That has happened.
        # Two tests, not one. "Different from the solution I am about to apply"
        # is not the same as "this is the student's own work": edit a solution
        # while a previously applied one is still in labs/, and the old solution
        # gets written over the backup, destroying it silently. The first cut of
        # this fix tested for the substring "NotSolved", which every solution
        # also carries in its own import line -- so the file it was meant to
        # protect against was exactly the one that defeated it. A file with no
        # `raise NotSolved` left in it is a solution, never an attempt.
        current = lab.read_text()
        if current.strip() != source.strip() and "raise NotSolved" in current:
            shutil.copy2(lab, BACKUP / lab_name)
        # The solutions import nothing from lab_support, so the stub's path
        # preamble is not needed -- but the check imports by file name, so the
        # module must expose the same function names, which it does.
        lab.write_text(source)
        print(f"applied {solution_name} -> labs/{lab_name}")
    if restore:
        if nothing:
            print(f"\n{len(nothing)} file(s) could not be restored, listed above. "
                  "labs/ still holds whatever was there — check before you overwrite it.")
            return 1
        print("\nyour own files are back. 'make check' will show them as you left them")
        return 0
    print("\nrun 'make check' to see them all green, "
          "or 'python3 apply.py --restore' to get your own work back")
    return 0


if __name__ == "__main__":
    sys.exit(main())

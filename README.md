# Module 2 — Cleaning data and building features

Data Mining and Analysis (course code CE3), Aalborg University, Copenhagen.
Edition 2026. Instructor: Valentino Servizi.

This repository holds one module's exercises and its demonstration
notebook. The slides, the examination material and the four other modules
are elsewhere.

## Getting started

```bash
bash setup.sh     # dependencies and the data the labs read
python3 apply.py  # copy the stubs into your working folder
make check        # run every check; each one tells you what it wants
```

A check that passes exits zero. A check that fails exits two and names the
file, the function and the lab. Nothing here needs a network, an account
or a key.

## What is here

| Folder | What it holds |
|---|---|
| `labs/` | The stubs you write |
| `verify/` | The checks that grade them |
| `data/` | The committed slice the labs read |
| `notebook/` | The demonstration, instructor's run |
| `READING.md` | The reading list and this module's examination question |

> **The notebook does not run here.** It was executed against the full
> course archive, which is not distributed. Read it for the measurements
> and the reasoning; the labs are what you run.

> **Solutions are not in this release.** They are published after the
> submission deadline, as a later release of this repository.

## Licence

Code is MIT licensed. Teaching material — text, figures, notebooks — is CC BY-NC-SA 4.0. The data slice is vehicle telemetry and identifies nobody.

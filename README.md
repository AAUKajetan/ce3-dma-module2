# Module 2 — Cleaning data and building features

Data Mining and Analysis (course code CE3), Aalborg University, Copenhagen.
Edition 2026. Instructor: Valentino Servizi.

This repository holds one module's exercises and its demonstration
notebook. The slides, the examination material and the four other modules
are elsewhere.

## Getting started

```bash
bash setup.sh   # dependencies and the data the labs read
make check      # every check; amber means 'not written yet', not 'wrong'
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
| `solutions/` | The reference answers, and `WHY.md` |

> **The solutions ship with this repository**, in `solutions/`, with the
> reasoning written out in `solutions/WHY.md`. Read them when you are
> stuck, not instead of trying — the examinable content is the reasoning,
> and it does not transfer by copying.

```bash
python3 apply.py            # copy the solutions over labs/ (your work is saved first)
python3 apply.py --restore  # get your own attempt back
```

> **The notebook does not run here.** It was executed against the full
> course archive, which is not distributed. Read it for the measurements
> and the reasoning; the labs are what you run.

## Licence

Code is MIT licensed. Teaching material — text, figures, notebooks — is CC BY-NC-SA 4.0. The data slice is vehicle telemetry and identifies nobody.

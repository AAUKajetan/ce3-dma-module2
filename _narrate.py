"""Narration for the solutions and the demonstrations.

    from _narrate import narrator, show_table, save_figure

Every solution's `__main__` uses this, so that `python3 solutions/lab_0K.py`
tells the whole story from the terminal -- what was loaded and its shape, each
intermediate quantity with its unit, the answer, and what the check will look
for -- and writes its figures under exercises/out/. A teacher evaluating a lab
runs `make demo` and reads one page; a student who wants to see the reference
answer at work runs one file.

Figures are plotly: an interactive .html always, and a .png beside it when
kaleido is installed (the slides use the .png; the .html is for the eye). The
figure helper never raises on a missing kaleido -- a demonstration must not fail
because a picture could not be rasterised.
"""
from __future__ import annotations

import html
import logging
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

_START = time.time()


class _Elapsed(logging.Formatter):
    """Seconds since the run began, then the lab, then the sentence."""

    def format(self, record):
        record.elapsed = f"{time.time() - _START:6.2f}s"
        return super().format(record)


def narrator(lab: int) -> logging.Logger:
    """A logger that writes to stdout, unbuffered, prefixed by the lab number.

    INFO is the default: every step a reader would want to see is INFO. DEBUG is
    for the arithmetic inside a step, and `python3 solutions/lab_0K.py -v` turns
    it on.
    """
    logger = logging.getLogger(f"lab{lab}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_Elapsed("%(elapsed)s  lab %(name)s  %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.name = str(lab)
    logger.setLevel(logging.DEBUG if "-v" in sys.argv else logging.INFO)
    # Third-party chatter (mlflow, alembic, urllib3) drowns the narrative.
    for noisy in ("alembic", "mlflow", "urllib3", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def show_table(frame, title: str, max_rows: int = 12, logger=None) -> None:
    """Print a titled table. Long frames are shown head and tail, never hidden."""
    import pandas as pd
    out = logger.info if logger else print
    out(f"--- {title} ({len(frame)} rows) ---")
    with pd.option_context("display.width", 120, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.4g}"):
        if len(frame) <= max_rows:
            text = frame.to_string()
        else:
            half = max_rows // 2
            text = frame.head(half).to_string() + "\n   ...\n" + frame.tail(half).to_string()
    for line in text.splitlines():
        out("    " + line)


def save_figure(fig, name: str, lab: int, logger=None, width: int = 1000,
                height: int = 560) -> pathlib.Path:
    """Write out/lab_0K_<name>.html and, when kaleido is present, the .png twin.

    Returns the html path. Says where it wrote, so the terminal is a table of
    contents for the pictures.
    """
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"lab_{lab:02d}_{name}"
    fig.update_layout(template="plotly_white", width=width, height=height,
                      margin=dict(l=60, r=30, t=60, b=60))
    # div_id pinned: plotly otherwise embeds a fresh random id on every write,
    # and a figure that differs byte for byte on each run is noise in a diff.
    fig.write_html(str(stem.with_suffix(".html")), include_plotlyjs="cdn", div_id=stem.name)
    wrote = [stem.with_suffix(".html").name]
    try:
        fig.write_image(str(stem.with_suffix(".png")), scale=2)
        wrote.append(stem.with_suffix(".png").name)
    except Exception as reason:  # kaleido missing or its browser refused
        wrote.append(f"(no png: {type(reason).__name__})")
    (logger.info if logger else print)(f"figure -> out/{' + '.join(wrote)}")
    return stem.with_suffix(".html")


def demo_index() -> pathlib.Path:
    """One page under out/ linking every figure, grouped by lab. `make demo` ends here."""
    OUT.mkdir(exist_ok=True)
    pages = sorted(OUT.glob("lab_*.html"))
    rows = []
    for page in pages:
        lab = page.name[4:6]
        title = page.stem[7:].replace("_", " ")
        rows.append(f'<li>Lab {int(lab)} — <a href="{html.escape(page.name)}">{html.escape(title)}</a></li>')
    body = "\n".join(rows) or "<li>no figures yet — run the solutions first</li>"
    index = OUT / "index.html"
    index.write_text(
        "<!doctype html><meta charset='utf-8'><title>Demonstrations</title>"
        "<h1>Demonstrations — figures written by the solutions</h1>"
        f"<ul>{body}</ul>")
    print(f"wrote out/index.html — {len(pages)} figure(s)")
    return index

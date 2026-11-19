"""Shared figure export: every figure as vector PDF + 600-dpi PNG (AAAI reproducibility requirement)."""
from __future__ import annotations
import pathlib


def save(fig, png_path, dpi=600, *, tight=True):
    """Save a 600-dpi PNG and vector PDF.

    For figures drawn at their final publication width, pass ``tight=False``:
    this preserves the exact physical page size and therefore the intended
    final font size when LaTeX includes the PDF at that same width.
    """
    p = pathlib.Path(png_path)
    bbox = "tight" if tight else None
    fig.savefig(p, dpi=dpi, bbox_inches=bbox)
    pdf = p.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches=bbox)   # vector
    print(f"WROTE {p}  +  {pdf}")
    return p, pdf

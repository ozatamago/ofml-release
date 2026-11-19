"""Paths of this repository, resolved from the location of this file (no absolute paths).

ROOT          repository root
EXP           ROOT/experiments
OUT_TABLES    where the table scripts write .tex files   (default ROOT/outputs/tables,  env OFML_OUT_TABLES)
OUT_FIGURES   where the figure scripts write .pdf files  (default ROOT/outputs/figures, env OFML_OUT_FIGURES)
JA_FONT_DIR   directory of the Japanese font (TrueType, glyf outlines) used only with --lang ja
              (default ROOT/fonts, env OFML_JA_FONT_DIR); the font is not distributed with this repository
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT_TABLES = Path(os.environ.get("OFML_OUT_TABLES", ROOT / "outputs" / "tables"))
OUT_FIGURES = Path(os.environ.get("OFML_OUT_FIGURES", ROOT / "outputs" / "figures"))
JA_FONT_DIR = Path(os.environ.get("OFML_JA_FONT_DIR", ROOT / "fonts"))
JA_FONT = JA_FONT_DIR / "HaranoAjiGothic-Medium-glyf.ttf"
for _d in (OUT_TABLES, OUT_FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

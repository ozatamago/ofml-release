"""Compare the .tex tables written to outputs/tables/ with the tables of the paper (scripts/expected_tables/).

The expected files are copies of the tables included by the paper's LaTeX source (tab:exp-transfer, tab:gpm-paired,
tab:acq-paired; Japanese and English).  usage: python scripts/compare_tables.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ofml_paths  # noqa: E402

ok = True
for exp in sorted((ROOT / "scripts" / "expected_tables").glob("*.tex")):
    got = ofml_paths.OUT_TABLES / exp.name
    same = got.exists() and got.read_bytes() == exp.read_bytes()
    ok &= same
    print(f"[{'ok' if same else 'DIFFERS'}] {got.relative_to(ROOT) if got.exists() else got} vs {exp.relative_to(ROOT)}")
print("all tables are identical to the paper" if ok else "some tables differ from the paper")
sys.exit(0 if ok else 1)

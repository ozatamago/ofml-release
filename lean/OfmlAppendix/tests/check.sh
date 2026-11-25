#!/usr/bin/env bash
# R1–R4 の自動検査。R5 は SPEC.md の表との人による照合。
set -u
cd "$(dirname "$0")/.."
fail=0
echo "== R1 build"; if lake build 2>&1 | tail -3; then :; fi
lake build >/dev/null 2>&1 || { echo "R1 FAIL"; fail=1; }
echo "== R2 forbidden constructs"
if grep -rnE '\bsorry\b|\badmit\b|^\s*axiom\b|native_decide|implemented_by' OfmlAppendix OfmlAppendix.lean; then echo "R2 FAIL"; fail=1; else echo "R2 ok"; fi
echo "== R3 axioms"
out=$(lake env lean tests/AxiomAudit.lean 2>&1); echo "$out"
if echo "$out" | grep -q "sorryAx\|error"; then echo "R3 FAIL"; fail=1; fi
bad=$(echo "$out" | grep -o "depends on axioms: \[[^]]*\]" | grep -vE "^depends on axioms: \[((propext|Classical.choice|Quot.sound)(, )?)+\]$" || true)
[ -n "$bad" ] && { echo "R3 FAIL: $bad"; fail=1; } || echo "R3 ok"
echo "== R4 non-vacuity"
lake env lean tests/NonVacuous.lean && echo "R4 ok" || { echo "R4 FAIL"; fail=1; }
exit $fail

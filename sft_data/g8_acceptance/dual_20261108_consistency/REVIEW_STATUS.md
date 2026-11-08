# PROBE + TruFor joint acceptance — dual_20261108_consistency

- sessions: 24
- generated: 2026-11-08T17:43:31
- status: **pending human review**
- workbench: `index.html`
- records: `records.json`
- comparison vs `dual_20261108_certainty`: `comparison_vs_certainty/comparison.md`
- comparison vs `dual_20261108_peakcrop`: `comparison_vs_peakcrop/comparison.md`
- automatic result vs `dual_20261108_certainty`: overall 12→20, whole 14→14,
  local 8→7, both 5→5, local false positives 4→1, joint issues 7→0,
  coverage failures 1→0, overall Uncertain 7→1, certainty null 7→1,
  certainty corrections 0→1 with 0 failed.
- vs `dual_20261108_peakcrop`: overall 17→20, whole 16→14, local 11→7,
  both 8→5, local false positives 2→1, joint issues 0→0, overall Uncertain 3→1.
- **pending human review; not SFT; not expanding to new sources.** Remaining:
  overall errors `339e8fae`/`51136`/`53059` (PROBE FP on Real) and `57501_0`
  (Fake→Uncertain); local recall 7/24 with conservative Uncertain; `ADM` local
  Tampered vs truth Not applicable; certainty High 21/23 still high.
- This batch is acceptance material, not SFT data; do not train on it.

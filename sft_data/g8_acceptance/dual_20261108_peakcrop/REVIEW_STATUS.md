# PROBE + TruFor joint acceptance — dual_20261108_peakcrop

- sessions: 24
- generated: 2026-11-08T13:55:25
- status: **pending human review**
- workbench: `index.html`
- records: `records.json`
- comparison vs pre-peak-crop batch: `comparison_vs_triptych_prefix/comparison.md`
- comparison vs `dual_20261108_closing`: `comparison_vs_closing/comparison.md`
- automatic result vs `dual_20261108_closing`: whole 16→16, local 12→11,
  both 7→8, local false positives 5→2, joint issues 1→0, local without TruFor 1→0,
  full-image bbox 3→0, strong mutual wording 3→1, confidence=0.95 18→17.
  Coverage corrections 4; failures 0; preclose checks 6; frame-review flags 1.
- **pending human review; not SFT; not expanding to new sources.** Main remaining
  issues: confidence still often 0.95, local false positives `51136`/`ADM`,
  `51136` local Tampered without a region, and one strong mutual-wording case
  (`ADM`) that this batch predates the clause-scope fix for.
- This batch is acceptance material, not SFT data; do not train on it.

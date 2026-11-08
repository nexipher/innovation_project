# PROBE + TruFor joint acceptance — dual_20261108_closing

- sessions: 24
- generated: 2026-11-08T12:13:34
- status: **pending human review**
- workbench: `index.html`
- records: `records.json`
- comparison vs previous rules: `comparison_vs_dual_20261108/comparison.md`
- comparison vs original control: `comparison_vs_control/comparison.md`
- automatic result: whole correct 11→16, local correct 7→12, both 3→7, call2 9→18,
  overall Uncertain 10→1, preclose checks 3/3 followed by a call, untried experts 5,
  confidence=0.95 10→18, local false positives 3→5, joint issue 1.
- decision: **still pending human review; not training and not expanding to new sources**.
  Confidence calibration, local false positives, single-expert definite conclusions and
  cross-task “相互支持” wording need another targeted iteration.
- This batch is acceptance material, not SFT data; do not train on it.

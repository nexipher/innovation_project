# g8 200-source development evaluation — CPU preflight

- date: 2026-11-10
- status: **prepared; 3-session GPU smoke passed; full run pending**
- batch: `sft_data/g8_dev_eval_20261110/`
- manifest: `sft_data/g8_dev_eval_20261110/manifest.json`
- manifest SHA256: `a4b319556ef075534359d825881734ba78d80b075f1e2e957e63468307dcfa7c`
- code commit at preflight: `f7742e989955766897dd33bc83c8c219584448cf`
  (doc-only commits after this preflight do not change the runtime identity;
  `batch_meta.json` records the exact commit at batch start)
- worktree at preflight: clean
- preflight static config SHA256: `04fc2717e4b70930`
  (`preflight_config.json`; runtime `batch_meta.json` records the actual
  `config_sha256` after building the real pipeline)

## Composition (200 sessions)

| group | n |
|---|---:|
| GenImage Real | 30 |
| AutoSplice Authentic | 30 |
| ADM | 10 |
| BigGAN | 10 |
| Glide | 10 |
| Midjourney | 10 |
| SD14 | 10 |
| SD15 | 10 |
| VQDM | 10 |
| Wukong | 10 |
| AutoSplice Forged | 60 |
| **total** | **200** |

Truth totals: whole-image Real 120 / AI-generated 80; local Not tampered 60 /
Not applicable 80 / Tampered 60. Split: dev 181 / val 19; test not read.

## Sampling rules and pools

- seed `20261110`; per-quota RNG `random.Random(f"{seed}:{quota}")` after sorting
  candidates by `source_id`. Full rule is in the manifest.
- GenImage: only `eval_dev_manifest_20261102.json` and
  `eval_val_manifest_20261102.json`, `no_known_use=True`. Candidates after
  excluding the previous 24 sources: Real 524; ADM 789, BigGAN 786, Glide 782,
  Midjourney 786, SD14 784, SD15 782, VQDM 782, Wukong 784.
- AutoSplice: `autosplice_dev_subset_20261102.json` dev bases only. After
  excluding the previous 24 sources and their bases: 91 bases, 65 forged
  candidates, 26 authentic-only candidates. Select 60 forged bases; 31 bases
  remain, select 30 authentic. Real/forged base sets are disjoint.
  Four authentic inputs come from forged bases that were **not** selected for
  local tampering; this is recorded as `authentic_from_forged_base`.
- Previous 24 exclusions: GenImage source IDs and AutoSplice bases
  `39890/41187/72399/42005/57501/58610/51136/42165/53059`; no q95 hash
  overlap. Every input is one q95 file per source.
- AutoSplice forged inputs use one selected variant per base; 29 selected
  forged bases have multiple variants and share the base-level mask. The mask
  is recorded with path and SHA256; no per-variant mask exists.

## CPU verification

- q95 hashes recomputed: 200/200 match
- source hashes recomputed: 200/200 match
- masks recomputed: 60/60 match
- duplicate sample/variant/path/q95/source: 0
- AutoSplice real/forged base overlap: 0
- previous-24 source/q95/base overlap: 0
- failures: 0

## Runtime configuration to fix at batch start

- model: Qwen3-VL-8B-Instruct, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`,
  bfloat16, non-thinking, greedy (temperature 0, do_sample false,
  max_new_tokens 1024).
- tools explicitly `probe,trufor`; certainty_v1; current controller/halting
  policy v3, max expert calls 5, max model turns 6.
- prompt actual SHA256:
  `6b4756ca21bc7a63325236f2f5dea52bfb1100b8e7ffb480c20fdeded527a102`.
- PROBE weights SHA256:
  `c8caa9bad54c029e4d22e1909c4673e46a239d8419a986e0ab461d423a450156`.
- TruFor weights SHA256:
  `ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844`.
- protocol provenance SHA256:
  `8bcd8b223e834425fcb47f788b3c43979fa7de018b1980fb9bfa3cfef35d9d31`.
- runtime runner writes `batch_meta.json` before the first session and includes
  commit, worktree, prompt, model revision, expert weights/preprocessing,
  budget/decoding, manifest hash and config SHA256. `--resume` refuses changed
  manifest/config/commit; completed records are skipped; failures are preserved
  separately. The runner refuses non-`qwen3_real` sessions (no Mock).

## Run-entry checks (2026-11-10)

- `--dry-run`: 200 samples, hashes 200 q95 / 200 source / 60 mask all match.
- 3-session GPU smoke (first three manifest inputs, real Qwen + PROBE + TruFor):
  3/3 completed, 0 failed, all `Real/Real/Not tampered/High`, 2 calls each,
  63.0 s / 48.6 s / 58.0 s (169.6 s total after model load). Pages, per-sample
  records, `records.json`, `records.jsonl`, `review_records.csv` and `index.html`
  were generated; the smoke output is outside the repo at
  `/root/autodl-tmp/g8_dev_eval_smoke_20261110` (not part of this batch).
- One smoke session had a wording-only coverage correction: the model fixed the
  `report` but left “共同支持” in `reasoning`, so `coverage_check.status` stayed
  `failed`; the final label was unchanged. This is a real correction-scope
  finding for review, not a runner failure.
- `--resume` on the smoke: identity verified, 3/3 completed skipped, 0.1 s.
- `scripts/analyze_dev_eval.py` on the smoke produced `SUMMARY.md`,
  `review_queue.json` and `REVIEW_GUIDE.md`; no analyzer exception.
- Core tests: 829 passed, 1 skipped, 853 deselected.

## Expected GPU time

The previous 24-session consistency batch averaged 31.7 s/session, but that
average includes 1-call sessions (about 12 s); its 2-call GenImage Real sessions
were 34.7–58.9 s. The 3-session smoke (all 2-call Real) averaged **56.5 s**.
Using 56 s/session for 200 sessions gives **11,200 s ≈ 3 h 7 m** after model
load. Plan **3.5–4 h** wall clock to allow for AutoSplice/TruFor sessions,
corrections and normal variation. This is an estimate, not a promise.

## Run entry

```bash
# CPU check only
python scripts/g8_dev_eval.py --dry-run

# real run (only with explicit GPU authorization)
FORENSIC_ALLOW_GPU=1 python scripts/g8_dev_eval.py \
  --manifest sft_data/g8_dev_eval_20261110/manifest.json \
  --out sft_data/g8_dev_eval_20261110

# resume; completed samples are skipped
FORENSIC_ALLOW_GPU=1 python scripts/g8_dev_eval.py --resume

# after run
python scripts/analyze_dev_eval.py --batch sft_data/g8_dev_eval_20261110
```

This is a development evaluation; it is not human acceptance and not SFT data.

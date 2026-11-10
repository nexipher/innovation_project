# g8 200-source development evaluation — CPU preflight

- date: 2026-11-10
- status: **prepared; GPU run pending explicit authorization gate**
- batch: `sft_data/g8_dev_eval_20261110/`
- manifest: `sft_data/g8_dev_eval_20261110/manifest.json`
- manifest SHA256: `a4b319556ef075534359d825881734ba78d80b075f1e2e957e63468307dcfa7c`
- code commit: `df65259232d1f316f8a88f224d95159021cc325b`
- worktree at preflight: clean
- preflight static config SHA256: `19ddbbcad0111e79`
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

## Expected GPU time

Previous 24 sessions: 762 s → 31.75 s/session. Linear estimate for 200:
**6,400 s ≈ 1 h 47 m**. Plan **2–2.5 h** including model load, corrections,
and normal variation. This is an estimate, not a promise.

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

#!/usr/bin/env python3
"""
G4-f / D2: the model pre-screen for visual grounding.

`utils.final_v2.validate_sample` judges the text against the record — evidence
ids, the posterior, the wording of a measurement.  It cannot see the picture,
and the review's largest remaining class is exactly that: a diagnostic crop
that does not contain what the observation describes ("眼睛周围的皮肤纹理"
checked against a crop of background), or a visual claim the image does not
show ("明显的压缩块状结构").

This asks the local vision model the two questions a reviewer would ask first,
so that only the flagged samples need a human:

  1. does the described target appear in the image, where the text says it is?
  2. does the crop actually show that target?

It is a *screener*: it never judges authenticity, and it is not trusted until
it has been validated against the human review.  `--against-dispositions`
compares its verdicts with the recorded dispositions and the reviewer's notes
(the notes are matched for visual-defect wording, which is a proxy — recorded
as such in the report).

Model calls need the GPU (agent.md §3.2); `--dry-run` exercises the whole
pipeline on CPU with a scripted judge.

Usage:
  python scripts/screen_visual_grounding.py --dry-run --limit 5
  python scripts/screen_visual_grounding.py --only-clean          # GPU
  python scripts/screen_visual_grounding.py --against-dispositions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from shutil import move
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.final_v2 import validate_sample

FINAL_V2_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "visual_grounding_screen.json")
BUCKET_FILES = ("sft_tool_positive.json", "sft_no_tool_positive.json",
                "sft_honest_abstention.json")

SCREEN_SYSTEM_PROMPT = (
    "你是取证训练数据的审核助手：只核对文字描述与图像内容是否相符，"
    "不判断图像真伪，也不执行任何取证协议。先描述你看到的，再回答。只输出 JSON。"
)

# Three single-purpose calls.  The first multi-image attempt failed on two
# counts worth remembering: with image A and the artifacts in one prompt the
# judge conflated them, and the "crop" it was shown was a colour-mapped
# residual heat map, so "does the crop show the described part" was answered
# false for 69 of 79 samples — the question was unanswerable as asked.
#
# So: let it look first (no answer text in the prompt), then compare the two
# texts, then ask about a *photo* crop only.
DESCRIBE_PROMPT = """先看图。**不要**做任何真伪判断，也不要推测拍摄或生成方式。

用一句话说明：这张图里有什么、主要对象在画面的什么位置。

只输出 JSON：
{"content": "……"}"""

COMPARE_PROMPT = """下面是一张图像的客观描述，以及模型对这张图给出的最终回答。
请比较两者，**不要判断图像真伪**：

图像实际内容：<<CONTENT>>

模型写的回答：
<<ANSWER>>

Q1: 回答里提到的**部位或对象**（如"眼睛""狗的毛发""建筑"）在图像实际内容里出现过吗？
Q2: 回答里的**异常判断**（如"明显的压缩块状结构""纹理不可能这么均匀"）
    与图像实际内容**矛盾**吗？矛盾答 false；只是描述不够详细请答 true。

只输出 JSON：
{"q1_target_in_image": true|false,
 "q2_anomaly_visible": true|false,
 "missing": "具体缺什么（没有则空字符串）",
 "reason": "一句话"}"""

CROP_PROMPT = """这是同一次分析中保存下来的**照片裁剪图**（不是热力图或可视化产物）。
模型对它的最终回答写在下面。

回答里提到的部位或异常，是否出现在**这张裁剪图**里？
裁剪图很小，只回答它本身显示了什么。

只输出 JSON：
{"crop_content": "这张裁剪图显示了什么（一句话）",
 "q3_crop_shows_target": true|false,
 "missing": "具体缺什么（没有则空字符串）",
 "reason": "一句话"}"""

# Words the reviewer used for the visual-grounding class (区域错配 / 描述无依据).
# A proxy for the human label, not the label itself.
VISUAL_DEFECT_CUES = (
    "区域不一致", "区域错配", "裁剪", "诊断框", "诊断区域", "选区", "调用框", "bbox",
    "不覆盖", "未覆盖", "没有覆盖", "与实际不符", "图不符", "位置", "笼统", "依据不足",
    "缺少依据", "难以核验", "无法核验", "不易核验", "编造", "虚构", "夸大",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def prompt_fingerprint() -> str:
    """
    The questions decide what a record means, so a report made with different
    questions must not be resumed into: the first version's prompt produced 75
    suspects out of 79, and a silent resume would have reported those numbers
    as if the new questions had produced them (the same failure the four-arm
    harness had before G3-d).
    """
    return hashlib.sha256(
        (SCREEN_SYSTEM_PROMPT + DESCRIBE_PROMPT + COMPARE_PROMPT + CROP_PROMPT)
        .encode("utf-8")
    ).hexdigest()[:16]


def load_samples(final_v2_dir: str = FINAL_V2_DIR) -> List[dict]:
    samples = []
    for name in BUCKET_FILES:
        path = os.path.join(final_v2_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for sample in json.load(handle):
                sample["_bucket_file"] = name.replace("sft_", "").replace(".json", "")
                samples.append(sample)
    return samples


def answer_of(sample: dict) -> str:
    """The closing assistant turn — what the sample would teach the model."""
    return next((t["value"] for t in reversed(sample.get("conversations", []))
                 if t.get("from") == "gpt"), "")


def crop_images(sample: dict, limit: int = 2) -> List[str]:
    """
    The photo crops, in order.

    `visual_artifacts` are deliberately excluded: they are colour-mapped
    residual maps, and asking "does this crop show the dog's fur" about a JET
    heat map is a question with no correct answer.
    """
    paths = []
    for token in sample.get("evidence_chain") or []:
        path = token.get("diagnostic_region_image")
        if path and path not in paths:
            paths.append(path)
    return paths[:limit]


# ---------------------------------------------------------------------------
# Prompt and parsing (CPU)
# ---------------------------------------------------------------------------

def build_screen_request(sample: dict) -> Tuple[str, List[str]]:
    """Call 1: describe the image, with the answer deliberately withheld."""
    return (DESCRIBE_PROMPT, [sample["image_path"]])


def build_compare_request(sample: dict, answer: str, content: str) -> Tuple[str, List[str]]:
    """
    Call 2: compare the judge's own description with the model's answer.

    The image goes along: the description anchors the comparison, and the
    judge may look again rather than argue from text alone.
    """
    return (COMPARE_PROMPT
            .replace("<<CONTENT>>", content or "（未能获得描述）")
            .replace("<<ANSWER>>", answer), [sample["image_path"]])


def build_crop_request(sample: dict) -> Optional[Tuple[str, List[str]]]:
    """The crop question, or None when the sample has no diagnostic crop."""
    crops = crop_images(sample)
    if not crops:
        return None
    return (f"{CROP_PROMPT}\n\n模型写的回答如下：\n{answer_of(sample)}\n", crops)


def parse_screen(text: str) -> Optional[dict]:
    """The judge's JSON verdict, repaired the way `Parser.parse_verdict` does."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    for candidate in (match.group(0),
                      match.group(0).replace("'", '"')):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def is_suspect(parsed: Optional[dict], keys=("q1_target_in_image", "q2_anomaly_visible")) -> bool:
    """
    Flag only on an explicit "no" to one of the three questions.

    The first version also flagged a non-empty "unsupported claims" list, and
    every one of the 79 samples tripped it — a judge that is asked whether a
    *judgement* ("纹理不自然") is visible in the picture will say no almost
    always, which makes the screen useless as a filter.  An unreadable answer
    is still a suspect: silence is not a pass.
    """
    if not parsed:
        return True
    for key in keys:
        if parsed.get(key) is False:
            return True
    return str(parsed.get("verdict", "")).lower() == "suspect"


# ---------------------------------------------------------------------------
# Screening (GPU)
# ---------------------------------------------------------------------------

class ScriptedJudge:
    """CPU stand-in for the vision model, used by --dry-run."""

    def __init__(self, verdict: str = "ok"):
        self._verdict = verdict
        self.calls = 0

    def judge(self, text: str, images: List[str]) -> str:
        self.calls += 1
        if "先看图" in text:
            return json.dumps({"content": "scripted: 一条鳄鱼趴在沙地上"},
                              ensure_ascii=False)
        if "照片裁剪图" in text:
            return json.dumps({"crop_content": "scripted",
                               "q3_crop_shows_target": True,
                               "missing": "", "verdict": self._verdict,
                               "reason": "scripted"}, ensure_ascii=False)
        return json.dumps({"content": "scripted",
                           "q1_target_in_image": True,
                           "q2_anomaly_visible": True,
                           "missing": "", "verdict": self._verdict,
                           "reason": "scripted"}, ensure_ascii=False)


def judge_with_qwen(client, text: str, images: List[str]) -> str:
    """One vision call: image A as the primary, the crops attached to the turn."""
    if not images:
        raise ValueError("a screening call needs at least one image; "
                         "the compare call carries the image on purpose")
    primary, crops = images[0], images[1:]
    history = [{"from": "user", "value": text,
                **({"image_paths": crops} if crops else {})}]
    return client.ask(primary, history)


def screen(samples: List[dict], judge, progress: bool = True,
           on_record=None, records: Optional[List[dict]] = None) -> List[dict]:
    records = records if records is not None else []
    done = {r["sample_id"] for r in records}
    for index, sample in enumerate(samples, 1):
        if sample["id"] in done:
            continue
        started = time.perf_counter()
        text, images = build_screen_request(sample)
        described = parse_screen(judge(text, images))
        content = (described or {}).get("content") or ""
        compare_request = build_compare_request(sample, answer_of(sample), content)
        whole = parse_screen(judge(*compare_request))
        crop_request = build_crop_request(sample)
        crop = (parse_screen(judge(*crop_request)) if crop_request else None)
        record = {
            "sample_id": sample["id"],
            "bucket": sample.get("_bucket_file"),
            "images": images + (crop_request[1] if crop_request else []),
            "suspect": is_suspect(whole) or (
                crop is not None and is_suspect(crop, ("q3_crop_shows_target",))),
            "content": content,
            "model_description_missing": not content,
            "crop_content": (crop or {}).get("crop_content"),
            "reason": (whole or {}).get("reason") or (crop or {}).get("reason"),
            "missing": (whole or {}).get("missing") or (crop or {}).get("missing") or "",
            "checks": {
                "q1_target_in_image": (whole or {}).get("q1_target_in_image"),
                "q2_anomaly_visible": (whole or {}).get("q2_anomaly_visible"),
                "q3_crop_shows_target": (crop or {}).get("q3_crop_shows_target"),
            },
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        records.append(record)
        if on_record is not None:
            on_record(record)
        if progress and index % 10 == 0:
            print(f"    [{index}/{len(samples)}] screened")
    return records


# ---------------------------------------------------------------------------
# Validation against the human review (CPU)
# ---------------------------------------------------------------------------

def human_visual_flag(notes: str) -> bool:
    """The reviewer complained about visual grounding in this note (proxy)."""
    return any(cue in (notes or "") for cue in VISUAL_DEFECT_CUES)


def evaluate(records: List[dict], dispositions: Dict[str, dict],
             clean_ids: set) -> dict:
    """
    How the screen did against the review, split by what the automatic checks
    already see.  `clean_ids` is the set the automatic checks pass — the only
    samples where the screen has anything to add.
    """
    subset = [r for r in records if r["sample_id"] in dispositions]
    flagged_human = {r["sample_id"] for r in subset
                     if human_visual_flag(dispositions[r["sample_id"]].get("notes", ""))}
    flagged_model = {r["sample_id"] for r in subset if r["suspect"]}
    clean_passing = {r["sample_id"] for r in subset if r["sample_id"] in clean_ids}
    hits = flagged_model & flagged_human
    return {
        "screened": len(subset),
        "human_visual_flagged": len(flagged_human),
        "model_suspects": len(flagged_model),
        "agreement": len(hits),
        "recall_on_human_visual": round(len(hits) / len(flagged_human), 4) if flagged_human else None,
        "precision_against_notes": round(len(hits) / len(flagged_model), 4) if flagged_model else None,
        "suspects_that_pass_automatic_checks": len(flagged_model & clean_passing),
        "clean_and_not_flagged": len(clean_passing - flagged_model),
        "model_flagged_human_did_not": sorted(flagged_model - flagged_human),
        "human_flagged_model_missed": sorted(flagged_human - flagged_model),
        "note": ("human_visual_flagged is a keyword proxy for the reviewer's "
                 "visual-defect notes, not a label; precision against it is a "
                 "lower bound, since the reviewer did not read every sample "
                 "for this class alone"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v2-dir", default=FINAL_V2_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-clean", action="store_true",
                        help="screen only the samples the automatic checks pass")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--against-dispositions", action="store_true",
                        help="compare with the recorded review (CPU)")
    parser.add_argument("--scripted-verdict", default="ok",
                        help="what the dry-run judge answers")
    args = parser.parse_args()

    samples = load_samples(args.final_v2_dir)
    samples_by_id = {s["id"]: s for s in samples}
    if args.only_clean:
        samples = [s for s in samples if not validate_sample(s)]
    if args.limit:
        samples = samples[:args.limit]
    print(f"samples to screen: {len(samples)}")

    fingerprint = prompt_fingerprint()
    records: List[dict] = []
    if os.path.exists(args.output) and not args.dry_run:
        try:
            with open(args.output, encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            previous = {}
        if previous.get("prompt_fingerprint") == fingerprint:
            records = previous.get("records", [])
            print(f"resuming: {len(records)} already screened")
        elif previous:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root, extension = os.path.splitext(args.output)
            moved = f"{root}.superseded_{stamp}{extension}"
            move(args.output, moved)
            print(f"questions changed — previous report kept as "
                  f"{os.path.basename(moved)}, starting cold")

    if args.dry_run:
        judge = ScriptedJudge(args.scripted_verdict).judge
        mode = "dry_run"
    else:
        try:
            import torch
        except ImportError:
            raise SystemExit("torch is required")
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA not available. Screening needs the local vision model "
                "(agent.md §3.2: obtain user authorization first). "
                "Use --dry-run for CPU plumbing.")
        from mllm.qwen_client import QwenVLClient

        # The screening prompt replaces the forensic system prompt on purpose:
        # this call judges a description against a picture, it is not the agent
        # running its protocol.
        client = QwenVLClient(system_prompt=SCREEN_SYSTEM_PROMPT)
        judge = lambda text, images: judge_with_qwen(client, text, images)   # noqa: E731
        mode = "gpu"

    def persist(_record=None):
        report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                  "mode": mode, "prompt_fingerprint": fingerprint,
                  "records": records}
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)

    screen(samples, judge, on_record=persist, records=records)
    persist()

    suspects = [r for r in records if r["suspect"]]
    print(f"\nscreened {len(records)} ｜ suspects {len(suspects)}")
    for record in suspects[:10]:
        print(f"   {record['sample_id'][:52]:54s} {str(record.get('reason'))[:70]}")

    if args.against_dispositions:
        from scripts.record_review import load_dispositions

        clean_ids = {sample_id for sample_id, sample in samples_by_id.items()
                     if not validate_sample(sample)}
        report = evaluate(records, load_dispositions(), clean_ids)
        print("\n对照人工审核：")
        for key, value in report.items():
            if key.startswith("model_flagged_human_did_not") or key.startswith("human_flagged_model_missed"):
                print(f"   {key}: {len(value)} 条 {value[:5]}")
            else:
                print(f"   {key}: {value}")
        with open(args.output, encoding="utf-8") as handle:
            blob = json.load(handle)
        blob["evaluation"] = report
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(blob, handle, ensure_ascii=False, indent=1)

    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()

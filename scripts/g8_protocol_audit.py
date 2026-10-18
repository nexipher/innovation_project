"""
按协议修订的判据数一遍批次里的模型输出（阶段 4.7 的验收工具）。

b1 的人工审核（18/32 待修）指出了几类**措辞**问题，其中三类可以在 trace 里
直接数出来。协议修订（4.7）是否真的改变了模型行为，不能靠"看起来好多了"：
先数 b1，再数 b2，比数。

三组判据（只看**模型输出轮**，不看我们注入的控制器提示）：

  * `presumption`   —— 疑罪从无 / presumption of innocence（4.7 已把模板里的
                       邀请删掉，只保留禁令；模型若仍援引，是它自己带出来的）
  * `overclaim`     —— 把"没有别的专家测量"说成"无冲突／证据链足够强／已交叉验证"
  * `excluded`      —— 声称压缩／编码／传感器噪声"已被排除或没有影响"

外加一组"该有的"：

  * `scope`         —— 结案里是否有**证据范围**的陈述（几个专家有效测量／
                       是否跨专家对照／未调用哪些专家）

用法：
  python scripts/g8_protocol_audit.py --generation g8 --batch b1
  python scripts/g8_protocol_audit.py --generation g8 --batch b1 --batch b2   # 对照
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from utils.batch import batch_paths  # noqa: E402

#: 每组判据＝(名字, 中文说明, 正则)。只匹配**模型输出轮**。
PATTERNS: Tuple[Tuple[str, str, re.Pattern], ...] = (
    ("presumption", "疑罪从无／presumption of innocence",
     re.compile(r"疑罪从无|presumption\s+of\s+innocence", re.I)),
    ("overclaim", "把没有其他专家测量说成无冲突/已交叉验证/证据链足够强",
     re.compile(r"无其他专家|没有其他专家|其他专家.{0,6}冲突|无冲突|"
                r"证据链.{0,4}足够强|足够强的证据链|已(?:经)?(?:完成)?.{0,4}交叉验证|"
                r"多专家(?:验证|对照).{0,6}(?:完成|一致)", re.I)),
    ("excluded", "声称压缩/编码/传感器噪声已被排除或无影响",
     re.compile(r"(?:无|没有|未发现|不存在).{0,8}(?:压缩|编码|传感器噪声|噪声干扰)|"
                r"(?:压缩|编码|传感器噪声).{0,8}(?:已被|已经)?.{0,4}(?:排除|不影响|无影响)|"
                r"未受.{0,10}(?:压缩|编码|噪声).{0,6}影响", re.I)),
    ("scope", "结案声明了证据范围（专家数/是否跨专家对照/未调用的专家）",
     re.compile(r"仅.{0,6}(?:一个|一项|一项专家|PROBE)|只有.{0,6}(?:一项|一个).{0,6}证据|"
                r"未调用|未获取|单一(?:专家|证据)|交叉验证|跨专家|多专家对照|"
                r"本次(?:只有|仅有).{0,10}(?:测量|证据)", re.I)),
)

#: 结案轮的识别：模型最后一条带 <verdict> 的输出
_VERDICT = re.compile(r"<verdict>", re.I)


def _clauses(value: str) -> Iterable[str]:
    return [part.strip() for part in re.split(r"[。；;！!？?\n]", value or "") if part.strip()]


def audit_trace(trace: dict) -> Dict[str, List[dict]]:
    """一条会话 → 各组命中（句子 + 轮次）。只扫模型输出轮。"""
    hits: Dict[str, List[dict]] = {name: [] for name, _desc, _rx in PATTERNS}
    for index, turn in enumerate(trace.get("conversations") or []):
        if turn.get("from") != "gpt":
            continue
        for clause in _clauses(turn.get("value") or ""):
            for name, _desc, pattern in PATTERNS:
                if pattern.search(clause):
                    hits[name].append({"turn": index, "clause": clause[:220]})
    return hits


def final_answer(trace: dict) -> str:
    """最后一轮带 <verdict> 的模型输出（结案）。"""
    for turn in reversed(trace.get("conversations") or []):
        if turn.get("from") == "gpt" and _VERDICT.search(turn.get("value") or ""):
            return turn["value"]
    return ""


def audit_batch(generation: str, batch: str, records_dir: str | None = None) -> dict:
    records_dir = records_dir or batch_paths(batch, generation)["trajectories"]
    sessions, per_group = 0, Counter()
    sessions_with = Counter()
    final_scope = 0
    examples: Dict[str, List[dict]] = {name: [] for name, _d, _r in PATTERNS}
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith(".json") or name == "metadata.json":
            continue
        with open(os.path.join(records_dir, name), encoding="utf-8") as handle:
            record = json.load(handle)
        trace_path = record.get("trace_path")
        if not trace_path or not os.path.exists(os.path.join(PROJECT_ROOT, trace_path)):
            continue
        with open(os.path.join(PROJECT_ROOT, trace_path), encoding="utf-8") as handle:
            trace = json.load(handle)
        sessions += 1
        hits = audit_trace(trace)
        for group, found in hits.items():
            per_group[group] += len(found)
            if found:
                sessions_with[group] += 1
                examples[group].extend({"trajectory_id": record["trajectory_id"], **item}
                                       for item in found[:2])
        answer = final_answer(trace)
        if answer and any(pattern.search(answer)
                          for _n, _d, pattern in PATTERNS if _n == "scope"):
            final_scope += 1
    return {
        "generation": generation, "batch": batch, "sessions": sessions,
        "hits": dict(per_group), "sessions_with": dict(sessions_with),
        "verdict_states_scope": final_scope,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--batch", action="append", required=True,
                        help="可重复：给了多批就并排列出")
    parser.add_argument("--records-dir", default=None)
    parser.add_argument("--json", action="store_true", help="输出 JSON（供入库/对比）")
    args = parser.parse_args()

    reports = [audit_batch(args.generation, batch, args.records_dir)
               for batch in args.batch]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0

    header = ["判据"] + [f"{r['batch']}（{r['sessions']} 会话）" for r in reports]
    print(" | ".join(header))
    for name, desc, _rx in PATTERNS:
        row = [f"{name}（{desc}）"]
        for report in reports:
            n = report["sessions"]
            row.append(f"命中 {report['hits'].get(name, 0)} 处 / "
                       f"{report['sessions_with'].get(name, 0)} 个会话"
                       + (f"（{100 * report['sessions_with'].get(name, 0) / n:.0f}%）" if n else ""))
        print(" | ".join(row))
    row = ["结论轮声明证据范围"]
    for report in reports:
        n = report["sessions"]
        row.append(f"{report['verdict_states_scope']}/{n}"
                   + (f"（{100 * report['verdict_states_scope'] / n:.0f}%）" if n else ""))
    print(" | ".join(row))

    for report in reports:
        print(f"\n=== {report['batch']} 命中样例（各判据前 2 条）===")
        for name, _desc, _rx in PATTERNS:
            for item in report["examples"].get(name, [])[:2]:
                print(f"  [{name}] {item['trajectory_id']} 轮 {item['turn']}: {item['clause']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

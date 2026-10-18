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
#:
#: 两个检测器在 b2 上暴露过口径错误（初版把"诚实声明"与"视觉观察"也算成问题）：
#:   * `无其他专家` 也匹配了"**无其他专家提供交叉验证**"——那是新协议要求的事实
#:     声明，不是吹嘘；吹嘘的具体形态是"无冲突/已完成验证/证据链足够强"。
#:   * `未发现压缩伪影` 是 RULES 8 允许的**视觉观察**，不是"声称已排除干扰"；
#:     后者是"已被排除/不影响/未受影响"这类**测量断言**。
#: 所以：overclaim 与 excluded 只认断言形态，并在同一子句出现"未/无+验证/观察"
#: 时不算。判据改动记在 plan §4.8（基线随之重算）。
#: 吹嘘"无冲突"的具体形态：必须是"其他专家的**测量结果**无冲突"。
#: 只写"无其他专家支持或矛盾证据"是在陈述**没有别的测量**（诚实声明），不算。
#: 只认"其他专家的**测量结果**无冲突"这一具体形态。"无冲突证据"在已经声明
#: "仅一个专家提供有效测量"之后是诚实的（b2 大量出现），不算吹嘘。
OVERCLAIM_CONFLICT = re.compile(
    r"(?:无|没有|未见|不存在)(?:其他|别的|其它)?专家.{0,6}测量结果.{0,4}(?:冲突|矛盾|异议)")
OVERCLAIM_DONE = re.compile(
    r"(?:已完成|已经完成|完成了|已经做过|做了).{0,6}(?:交叉验证|多专家对照|多专家验证)"
    r"|多专家(?:对照|验证).{0,6}(?:完成|一致|吻合)")
OVERCLAIM_STRENGTH = re.compile(
    r"证据链.{0,6}(?:足够强|充分|完整)|足够强的证据链|当前证据(?:已)?(?:充分|足够充分)")
OVERCLAIM_AVAILABILITY = re.compile(
    r"(?:无|没有)(?:其他|别的)专家(?:可|能)(?:提供|调用|补充|使用)|无其他专家可用")
HONEST_ABSENCE = re.compile(
    r"(?:无|没有|未|尚未|未经).{0,12}(?:交叉验证|多专家对照|多专家验证)")
OBSERVATION = re.compile(r"(?:未发现|未见|没有发现|无明显|不存在明显).{0,10}(?:压缩|噪点|噪声|伪影|块状)")
EXCLUDED_CLAIM = re.compile(
    r"(?:压缩|编码|重编码|传感器?噪声|噪声干扰).{0,12}(?:已被?排除|已排除|不影响|无影响|可忽略|未受影响)"
    r"|未受.{0,10}(?:压缩|编码|噪声).{0,8}影响"
    r"|(?:无|没有|未见|未发现).{0,10}(?:压缩|编码|重编码|噪声).{0,10}影响"
    r"|(?:排除|排除掉).{0,10}(?:压缩|编码|噪声)")

PATTERNS: Tuple[Tuple[str, str, re.Pattern], ...] = (
    ("presumption", "疑罪从无／presumption of innocence",
     re.compile(r"疑罪从无|presumption\s+of\s+innocence", re.I)),
    ("overclaim", "把没有其他专家测量说成无冲突/已交叉验证/证据链足够强",
     re.compile("overclaim")),
    ("excluded", "声称压缩/编码/传感器噪声已被排除或无影响（不是「未发现伪影」的观察）",
     re.compile("excluded")),
    ("scope", "结案声明了证据范围（专家数/是否跨专家对照/未调用的专家）",
     re.compile(r"仅.{0,6}(?:一个|一项|一项专家|PROBE)|只有.{0,6}(?:一项|一个).{0,6}证据|"
                r"未调用|未获取|单一(?:专家|证据)|证据(?:单一|仅一?项)|"
                r"无其他专家(?:的)?测量结果|无(?:其他|别的)专家.{0,6}(?:测量|证据)|"
                r"交叉验证|跨专家|多专家对照|"
                r"本次(?:只有|仅有).{0,10}(?:测量|证据)", re.I)),
)


def _matches(name: str, clause: str) -> bool:
    """按判据的**意图**判定，而不是一个词命中就算。"""
    if name == "overclaim":
        if HONEST_ABSENCE.search(clause):
            return False
        return bool(OVERCLAIM_CONFLICT.search(clause) or OVERCLAIM_DONE.search(clause)
                    or OVERCLAIM_STRENGTH.search(clause)
                    or OVERCLAIM_AVAILABILITY.search(clause))
    if name == "excluded":
        if OBSERVATION.search(clause):
            return False
        return bool(EXCLUDED_CLAIM.search(clause))
    if name == "scope":
        # "无其他专家测量结果冲突"同时命中 scope 句式，但它暗示"别的专家测过"
        # ——那是 overclaim，不是范围声明。判据在这里互斥。
        if _matches("overclaim", clause):
            return False
        pattern = next(p for n, _d, p in PATTERNS if n == "scope")
        return bool(pattern.search(clause))
    raise KeyError(name)

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
                if name in ("overclaim", "excluded", "scope"):
                    matched = _matches(name, clause)
                else:
                    matched = bool(pattern.search(clause))
                if matched:
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
    outcomes = Counter()
    calls = Counter()
    halting = Counter()
    elapsed: List[float] = []
    log_complete = Counter()
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
        # 结果分类：正确 / 错误 / 弃权 / 不可解析——**弃权不并入错误**
        verdict = record.get("final_verdict")
        truth = record.get("ground_truth")
        if verdict == "Uncertain":
            outcomes["abstain"] += 1
        elif record.get("verdict_status") in ("no_parseable_conclusion",
                                              "budget_exhausted", "format_failure",
                                              "loop_protection", "evidence_check_failed"):
            outcomes["no_conclusion"] += 1
        elif verdict == truth:
            outcomes["correct"] += 1
        else:
            outcomes["wrong"] += 1
        calls[(record.get("counters") or {}).get("expert_calls")] += 1
        halting[record.get("halting_reason")] += 1
        if isinstance(record.get("elapsed_s"), (int, float)):
            elapsed.append(float(record["elapsed_s"]))
        metadata = trace.get("metadata") or {}
        log_complete["system_prompt"] += bool((metadata.get("system_prompt") or {}).get("text"))
        log_complete["model_calls"] += bool(metadata.get("model_calls"))
        log_complete["image_size"] += bool(metadata.get("image_size"))
        hits = audit_trace(trace)
        for group, found in hits.items():
            per_group[group] += len(found)
            if found:
                sessions_with[group] += 1
                examples[group].extend({"trajectory_id": record["trajectory_id"], **item}
                                       for item in found[:2])
        answer = final_answer(trace)
        # 结论轮的 scope 也要走**同一判定**（词表命中不够：b1 的"无其他专家测量
        # 结果冲突"会命中 scope 句式，但它是 overclaim，不是范围声明）。
        if answer and any(_matches("scope", clause)
                          for clause in _clauses(answer)):
            final_scope += 1
    return {
        "generation": generation, "batch": batch, "sessions": sessions,
        "hits": dict(per_group), "sessions_with": dict(sessions_with),
        "verdict_states_scope": final_scope,
        "outcomes": dict(outcomes),
        "calls": {str(key): value for key, value in calls.items()},
        "halting": dict(halting),
        "elapsed": {"mean": round(sum(elapsed) / len(elapsed), 1) if elapsed else None,
                     "max": max(elapsed) if elapsed else None},
        "log_complete": {key: f"{value}/{sessions}" for key, value in log_complete.items()},
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

    for report in reports:
        outcomes = report["outcomes"]
        print(f"=== {report['batch']}（{report['sessions']} 会话）===")
        print(f"  结果：正确 {outcomes.get('correct', 0)}｜错误 {outcomes.get('wrong', 0)}"
              f"｜弃权 {outcomes.get('abstain', 0)}｜未形成结论 {outcomes.get('no_conclusion', 0)}"
              "   （弃权单列，不并入错误）")
        print(f"  工具调用分布：{report['calls']}｜停止原因：{report['halting']}")
        print(f"  耗时：均 {report['elapsed']['mean']} s｜最长 {report['elapsed']['max']} s")
        print(f"  日志完整性：系统提示词 {report['log_complete'].get('system_prompt')}"
              f"｜调用明细 {report['log_complete'].get('model_calls')}"
              f"｜图像尺寸 {report['log_complete'].get('image_size')}")
    print()
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

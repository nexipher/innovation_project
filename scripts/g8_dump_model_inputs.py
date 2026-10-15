"""
把**模型实际看到的输入**原样打出来，供人核对（阶段 4.5 小批前的第三件事）。

用户 2026-10-15：*"建议打印几份实际模型输入，确认没有把 bootstrap 统计、内部写法
要求等重新塞回包里。"* 这一条要的是**可以自己看的证据**，不是一句"我检查过了"。

三部分，全部走真实代码路径（不重写任何一句话）：

1. **系统提示词**：`build_forensic_prompt(CURRENT_TOOL_TAGS)` 的原文——工具段由登记
   表生成，读它就能看到每个工具带出去的条件是哪一句。
2. **一次真实会话的完整对话**：CPU 上跑 mock 客户端 + mock 专家（不加载任何权重），
   把 trace 里的每一轮原样打印。精简包、工具缺席提示、状态小结都在里面，它们的
   形状与真实会话完全一致（差别只是数值来自 mock）。
3. **逐个数值的包**：同一专家在不同数值下的包（NPR 低分是关键边界：登记表判
   `Uncertain`，不得被读成强 Real 证据），以及一次**失败**测量的包——失败必须
   没有分数、没有尺度与方向。
4. **机械扫描**：对以上全部文本扫内部标记（bootstrap/AUROC/CI[/阶段编号/用户日期/
   写法要求），逐项打印是否命中。

写法与产物：
    python scripts/g8_dump_model_inputs.py                 # 打印 + 落盘
    python scripts/g8_dump_model_inputs.py --out <file>    # 指定产物路径

产物默认写到 `sft_data/g8_model_inputs/<时间戳>/model_inputs.md`——不是批次目录，
不触发冻结门禁。
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "sft_data", "g8_model_inputs")

# 内部措辞：出现在模型输入里就是泄漏。分三类，因为来源不同：
#   统计数值（bootstrap/CI/AUROC）、我们的流程语言（阶段/批次/退役/登记表）、
#   写给自己的写法要求（"适用条件写…"）。
INTERNAL_MARKERS = (
    "bootstrap", "CI[", "置信区间", "AUROC", "p=", "P=",
    "G2", "G7", "阶段", "批次", "退役", "登记表", "用户 2026", "本轮 val",
    "Retired", "replay only", "700-sample", "writing requirements",
    "适用条件写", "不得写成", "写法",
)


def _synthetic_token(source, evidence_name, value, phenomenon, scope="global",
                     status="measured", reason=None):
    """按真实专家填 token 的方式造一个 token（文字取自真实渲染器）。"""
    token = {
        "evidence_id": f"E-dump-{source}",
        "source": source,
        "evidence_name": evidence_name,
        "phenomenon": phenomenon,
        "reasoning": "（后台字段，不应出现在包里）",
        "strength": value,
        "raw_metric": value,
        "support": "AI-generated",
        "interpretation_text": "（后台字段，不应出现在包里）",
        "measurement_scope": scope,
        "status": status,
        "model_delivered": "slim",
    }
    if reason:
        token["reason"] = reason
    return token


def _tokens():
    from experts.external_base import score_phenomenon

    return {
        "probe 高分（0.93）": _synthetic_token(
            "probe_dino_expert", "aigi_discriminator_score", 0.93,
            score_phenomenon(0.93, "整图 resize(224,224)")),
        "probe 低分（0.02）": _synthetic_token(
            "probe_dino_expert", "aigi_discriminator_score", 0.02,
            score_phenomenon(0.02, "整图 resize(224,224)")),
        "npr 高分（0.98）": _synthetic_token(
            "npr_expert", "npr_upsampling_artefact_score", 0.98,
            score_phenomenon(0.98, "整图 resize(224,224)")),
        "npr 低分（0.02）——JPEG 边界": _synthetic_token(
            "npr_expert", "npr_upsampling_artefact_score", 0.02,
            score_phenomenon(0.02, "整图 resize(224,224)")),
        "univfd（0.88）": _synthetic_token(
            "univfd_expert", "univfd_linear_head_score", 0.88,
            score_phenomenon(0.88, "centre crop 224x224")),
        "npr 执行失败": _synthetic_token(
            "npr_expert", "npr_upsampling_artefact_score", 0.5,
            score_phenomenon(0.5, "整图 resize(224,224)"),
            status="failed", reason="RuntimeError: 权重文件不存在"),
    }


def _session_transcript():
    """CPU 上真跑一次 mock 会话，返回 trace（不加载任何权重）。"""
    import tempfile

    from PIL import Image

    import main
    from utils.logger import SessionLogger

    work = tempfile.mkdtemp(prefix="g8_dump_")
    image = os.path.join(work, "input.png")
    Image.new("RGB", (96, 96), (120, 90, 200)).save(image)
    fsm = main.build_pipeline("mock")
    fsm._logger = SessionLogger(sft_dir=work)
    result = fsm.run(image, "Fake")
    with open(result["sft_data_path"], encoding="utf-8") as handle:
        return json.load(handle)


def _scan(text):
    found = [marker for marker in INTERNAL_MARKERS if marker in text]
    return found


def build_dump() -> str:
    from mllm.message_builder import build_forensic_prompt
    from state_machine.controller import ForensicStateMachine
    from state_machine.evidence_tokenizer import EvidenceTokenizer

    parts = []
    parts.append("# 模型实际看到的输入（阶段 4.5）\n")
    parts.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                 f"（机器本地时间，不是估计值）\n")

    prompt = build_forensic_prompt(list(ForensicStateMachine.CURRENT_TOOL_TAGS))
    parts.append("\n## 1. 系统提示词（原文，逐字）\n")
    parts.append("```\n" + prompt + "\n```\n")

    parts.append("\n## 2. 一次真实会话的全部对话（CPU / mock 客户端与专家）\n")
    parts.append("权重未加载，数值来自 mock；**包的形状、条件句、状态小结与工具"
                 "缺席提示与真实会话同源**（`EvidenceTokenizer.model_visible` / 控制器）。\n"
                 "\n> 读的时候注意：gpt 轮里出现的 `strength=0.75`、"
                 "「物理-语义一致性校验」等字样来自 **mock 客户端自己的模板**，"
                 "不是系统注入的文本；系统注入的是 user 轮（包、状态小结、"
                 "`[System: …]` 提示）。\n")
    trace = _session_transcript()
    for index, turn in enumerate(trace.get("conversations", [])):
        parts.append(f"\n### 第 {index} 轮 · {turn.get('from')}\n")
        parts.append("```\n" + (turn.get("value") or "") + "\n```\n")

    parts.append("\n## 3. 逐个数值的包（同一专家的不同取值）\n")
    payloads = {}
    for label, token in _tokens().items():
        pack = EvidenceTokenizer.model_visible(token)
        payloads[label] = pack
        parts.append(f"\n### {label}\n")
        parts.append("```json\n" + json.dumps(pack, ensure_ascii=False, indent=2)
                     + "\n```\n")

    parts.append("\n## 4. 内部标记扫描\n")
    bodies = {"系统提示词": prompt}
    for index, turn in enumerate(trace.get("conversations", [])):
        bodies[f"会话第 {index} 轮（{turn.get('from')}）"] = turn.get("value") or ""
    for label, pack in payloads.items():
        bodies[f"包：{label}"] = json.dumps(pack, ensure_ascii=False)
    lines = ["| 文本 | 命中的内部标记 |", "| --- | --- |"]
    total = 0
    for label, body in bodies.items():
        found = _scan(body)
        total += len(found)
        lines.append(f"| {label} | {'、'.join(found) if found else '（无）'} |")
    parts.append("\n".join(lines) + "\n")
    parts.append(f"\n合计命中：**{total}**"
                 "（0 才算通过；判定用的是词表，不是我的印象）\n")
    # 只扫**模型输入**，不扫这份文档自己的标题与说明（"阶段 4.5"是我们的话）
    hits = {label: _scan(body) for label, body in bodies.items() if _scan(body)}

    # 额外事实：包里的字段集合，与提示词声明的清单是否一致
    from_marker = set()
    for pack in payloads.values():
        from_marker |= set(pack)
    parts.append(f"\n包里出现过的字段：`{sorted(from_marker)}`\n")

    return "".join(parts), hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(
        OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir if os.path.isdir(out_dir) or not out_dir.endswith(".md")
                else os.path.dirname(out_dir), exist_ok=True)
    text, hits = build_dump()
    path = out_dir if out_dir.endswith(".md") else os.path.join(
        out_dir, "model_inputs.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)

    print(text)
    print(f"\n产物：{os.path.relpath(path, PROJECT_ROOT)}")
    if hits:
        for label, markers in hits.items():
            print(f"内部标记命中：{label} → {markers}")
        return 1
    print("内部标记命中：（无）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

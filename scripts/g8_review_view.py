"""
最小审阅视图（阶段 4.5）。

plan.md §4.5 要的是：**原始图像、实际投递的证据包、完整原始对话、结案与外部控制
原因**，而且"通过后才允许扩大生成"。所以这一页要能在**不看任何训练流水线**的前提
下成立——它直接读批次自己产出的两样东西：

  * 记录 `sft_data/<gen>/trajectories/<batch>/<变体>__<策略>.json`
  * 会话 trace（记录里的 `trace_path`）

不经过准入（`admit_trajectories_g4`）：那是"哪些轨迹可以**训练**"的门禁，它按旧
工具箱的标签体系判定、还要求同变体的无工具基线——拿它当验收门会把整批判成 reject
而审阅者什么也看不到（实测：7/7 reject，理由全是 `unknown tool 'npr_expert'` 与
`no no-tool baseline`）。**验收看行为，训练看准入**，两件事不要混。

产物是一页静态 HTML（无服务器、无外部资源，图像按相对路径引用）：
  `sft_data/<gen>/review/<batch>/index.html` + 每个会话一页。

用法：
  python scripts/g8_review_view.py --generation g8 --batch b1 [--output-dir ...]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from utils.batch import batch_paths  # noqa: E402

CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
       margin: 0 auto; max-width: 1180px; padding: 24px; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 28px; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #8884; padding: 6px 8px; text-align: left;
         vertical-align: top; font-size: 14px; }
th { background: #8881; }
img { max-width: 100%; border: 1px solid #8884; border-radius: 6px; }
.turn { border-left: 3px solid #8886; margin: 10px 0; padding: 6px 12px; }
.turn.user { border-color: #2b7; } .turn.gpt { border-color: #37b; }
.role { font-weight: 600; font-size: 13px; opacity: .8; }
pre { white-space: pre-wrap; word-break: break-word; margin: 6px 0;
      background: #8881; padding: 10px; border-radius: 6px; font-size: 13px; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px;
        background: #8883; font-size: 13px; margin-right: 6px; }
.ok { background: #2b73; } .bad { background: #b223; } .warn { background: #b903; }
.muted { opacity: .7; font-size: 13px; }
"""


def esc(text) -> str:
    return html.escape(str(text))


def _rel(target: str, page_dir: str) -> str:
    absolute = target if os.path.isabs(target) else os.path.join(PROJECT_ROOT, target)
    return os.path.relpath(absolute, page_dir).replace(os.sep, "/")


def _pack_block(value: str) -> str:
    """证据轮是一个 JSON（当时**实际投递**的那段）：格式化后原样呈现。"""
    stripped = value.lstrip()
    if not stripped.startswith("{"):
        return f"<pre>{esc(value)}</pre>"
    try:
        parsed = json.loads(value)
    except ValueError:
        return f"<pre>{esc(value)}</pre>"
    return f"<pre>{esc(json.dumps(parsed, ensure_ascii=False, indent=2))}</pre>"


def session_page(record: dict, trace: dict, page_dir: str, index_name: str) -> str:
    verdict = record.get("final_verdict")
    truth = record.get("ground_truth")
    agree = "ok" if verdict == truth else ("warn" if verdict == "Uncertain" else "bad")
    evidence_check = record.get("evidence_check") or {}
    counters = record.get("counters") or {}
    parts = [
        f'<p><a href="{esc(index_name)}">← 批次目录</a></p>',
        f"<h1>{esc(record.get('trajectory_id'))}</h1>",
        f'<p><span class="pill {agree}">判定 {esc(verdict)}</span>'
        f'<span class="pill">真值 {esc(truth)}</span>'
        f'<span class="pill">{esc(record.get("verdict_status"))}</span>'
        f'<span class="pill">停止原因 {esc(record.get("halting_reason"))}</span>'
        f'<span class="pill">{esc(record.get("treatment"))}</span>'
        f'<span class="pill">{esc(record.get("generator"))}</span></p>',
        "<h2>原始图像（模型看到的变体）</h2>",
        f'<img src="{esc(_rel(record["image_path"], page_dir))}" '
        f'alt="{esc(record.get("image_path"))}">',
        '<p class="muted">源：' + esc(record.get("source_id"))
        + '｜分辨率 ' + esc(record.get("resolution") or "-")
        + '｜容器 ' + esc(record.get("container"))
        + '｜耗时 ' + esc(record.get("elapsed_s")) + " s</p>",
        "<h2>结果与外部控制</h2>",
        "<table><tr><th>项</th><th>值</th></tr>",
        f"<tr><td>模型自己的报告</td><td>{esc(record.get('report'))}</td></tr>",
        f"<tr><td>模型候选 / 采纳</td><td>{esc(record.get('model_candidate'))} / "
        f"{esc(record.get('final_verdict'))}</td></tr>",
        f"<tr><td>置信度</td><td>{esc(record.get('confidence'))}</td></tr>",
        f"<tr><td>控制原因</td><td>{esc(record.get('policy_reasons'))}</td></tr>",
        f"<tr><td>引用检查</td><td>{esc(evidence_check.get('status'))}"
        f"（缺陷 {len(evidence_check.get('findings') or [])}，"
        f"候选 {len(evidence_check.get('candidates') or [])}）</td></tr>",
        f"<tr><td>计数器</td><td>{esc(json.dumps(counters, ensure_ascii=False))}</td></tr>",
        f"<tr><td>工具</td><td>{esc(', '.join(record.get('tools_served') or []))}</td></tr>",
        "</table>",
    ]

    chain = record.get("evidence") or []
    if chain:
        parts.append("<h2>证据链（模型看到的是精简包）</h2>")
        parts.append("<table><tr><th>expert</th><th>evidence_id</th><th>状态</th>"
                     "<th>投递</th><th>score</th><th>说明</th></tr>")
        for token in chain:
            from state_machine.evidence_tokenizer import EvidenceTokenizer

            pack = EvidenceTokenizer.model_visible(token) if \
                token.get("model_delivered") == "slim" else token
            score = pack.get("score") or {}
            parts.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                "<td>{}</td></tr>".format(
                    esc(pack.get("expert") or token.get("source")),
                    esc(pack.get("evidence_id")),
                    esc(pack.get("status")),
                    esc(token.get("model_delivered", "full")),
                    esc(score.get("value")),
                    esc((pack.get("reading") or "")[:400])))
        parts.append("</table>")

    parts.append("<h2>完整原始对话</h2>")
    for index, turn in enumerate(trace.get("conversations") or []):
        role = turn.get("from") or "?"
        parts.append(f'<div class="turn {esc(role)}">'
                     f'<div class="role">#{index} {esc(role)}</div>'
                     + _pack_block(turn.get("value") or "")
                     + "</div>")
    return _page(record.get("trajectory_id"), "".join(parts))


def _page(title: str, body: str) -> str:
    return ("<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
            f"{body}</body></html>")


def build(generation: str, batch: str, output_dir: str,
          records_dir: Optional[str] = None) -> dict:
    paths = batch_paths(batch, generation)
    # 记录目录可显式给出：测试据此重定向，**任何测试都不写进仓库**（批次目录是
    # 冻结与审核的凭据，测试数据混进去就分不出真假了）。
    records_dir = records_dir or paths["trajectories"]
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(name for name in os.listdir(records_dir)
                   if name.endswith(".json") and name != "metadata.json")
    rows, written, missing = [], 0, []
    for name in names:
        with open(os.path.join(records_dir, name), encoding="utf-8") as handle:
            record = json.load(handle)
        trace_path = record.get("trace_path")
        if not trace_path:
            missing.append(name)
            continue
        with open(os.path.join(PROJECT_ROOT, trace_path), encoding="utf-8") as handle:
            trace = json.load(handle)
        page = f"{os.path.splitext(name)[0]}.html"
        with open(os.path.join(output_dir, page), "w", encoding="utf-8") as handle:
            handle.write(session_page(record, trace, output_dir, "index.html"))
        written += 1
        agree = "ok" if record.get("final_verdict") == record.get("ground_truth") else (
            "warn" if record.get("final_verdict") == "Uncertain" else "bad")
        rows.append(
            f'<tr><td><a href="{esc(page)}">{esc(record.get("variant_id"))}</a></td>'
            f'<td>{esc(record.get("treatment"))}</td>'
            f'<td>{esc(record.get("ground_truth"))}</td>'
            f'<td><span class="pill {agree}">{esc(record.get("final_verdict"))}</span></td>'
            f'<td>{esc(record.get("verdict_status"))}</td>'
            f'<td>{esc(record.get("halting_reason"))}</td>'
            f'<td>{esc((record.get("counters") or {}).get("expert_calls"))}</td>'
            f'<td>{esc(record.get("elapsed_s"))}s</td></tr>')

    metadata = {}
    metadata_path = os.path.join(records_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    header = [
        f"<h1>批次 {esc(batch)}（代次 {esc(generation)}）· 最小审阅视图</h1>",
        f'<p class="muted">生成于 {esc(datetime.now().isoformat(timespec="seconds"))}'
        f'｜会话 {len(rows)} 条｜模式 {esc(metadata.get("mode"))}'
        f'｜策略 {esc(metadata.get("policy"))} / {esc(metadata.get("policy_version"))}'
        f"</p>",
    ]
    if metadata:
        header.append(
            f'<p class="muted">模型 {esc((metadata.get("model") or {}).get("checkpoint"))}'
            f'@{esc(str((metadata.get("model") or {}).get("revision"))[:8])}'
            f'｜提示词 {esc((metadata.get("prompts") or {}).get("digest"))}'
            f'｜解码 {esc(metadata.get("decoding"))}<br>专家 '
            + esc(", ".join(f"{name}({entry.get('device')})"
                            for name, entry in (metadata.get("experts") or {}).items()))
            + "</p>")
    header.append("<table><tr><th>变体</th><th>格式</th><th>真值</th><th>判定</th>"
                  "<th>结案来源</th><th>停止原因</th><th>工具调用</th><th>耗时</th></tr>")
    header.append("".join(rows) + "</table>")
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(_page(f"批次 {batch} 审阅", "".join(header)))

    return {"sessions": written, "missing_trace": missing, "index":
            os.path.relpath(os.path.join(output_dir, "index.html"), PROJECT_ROOT)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or batch_paths(args.batch, args.generation)["review"]
    report = build(args.generation, args.batch, output_dir)
    print(f"会话页 {report['sessions']} 个；索引 {report['index']}")
    if report["missing_trace"]:
        print(f"缺 trace 的记录（未出页）：{report['missing_trace']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

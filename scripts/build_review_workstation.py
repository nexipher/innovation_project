#!/usr/bin/env python3
"""
G4-f: the review workstation.

Human review of `final_v2` needs everything on one page: the source image and
the re-encoded variant the model actually saw, the expert artifacts, the
Evidence Bundle with its calibration and applicability, how the run compares
with its tool-free baseline, and the *whole* conversation the sample would
train — including the call turns and the evidence rounds, not just the first
assistant reply.

The workstation is static HTML with no server and no external assets: open
`sft_data/review/index.html` in a browser.  Images are referenced by relative
path, so nothing is copied and the page keeps working from the repository.
Decisions are kept in the browser (`localStorage`) and exported as JSON; feed
the export (or a filled `worklist.csv`) to

  python scripts/record_review.py --import <file> --reviewer <you>

which appends them to `sft_data/review/dispositions.jsonl` through the same
validation the CLI path uses.  Nothing is written back to disk from the page
itself.

Review protocol (plan.md §4.11, revised 2026-10-09): one full pass over all
samples.  `high_risk` marks the ones worth extra care; it does not ask for a
second reader.

Usage:
  python scripts/build_review_workstation.py
  python scripts/build_review_workstation.py --output-dir /tmp/review
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

FINAL_V2_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2")
BUCKET_FILES = ("sft_tool_positive.json", "sft_no_tool_positive.json",
                "sft_honest_abstention.json")
VARIANTS_MANIFEST = os.path.join(PROJECT_ROOT, "sft_data", "variants", "manifest.json")
TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sft_data", "review")

HIGH_RISK_CONFIDENCE = 0.8

BUCKET_LABELS = {
    "tool_positive": "工具正样本（调用过专家）",
    "no_tool_positive": "无工具正样本（未调用专家）",
    "honest_abstention": "合理弃权",
}

# Shared by every bucket.
COMMON_CHECKS = [
    ("observation", "<observation> 里每条视觉陈述都能在图上直接核验吗？"),
    ("evidence_fidelity", "<forensic_evidence> 与真实 token 一致吗（evidence_id、测量范围、原始数值、校准概率）？"),
    ("container", "有没有把 PNG/JPEG 容器当成真假的理由？（这是必须拒绝的错误）"),
    ("format", "四段式与 <verdict> JSON 格式完整、无多余脚手架文字吗？"),
    ("teaching", "训练这条样本会不会教出坏习惯（格式捷径 / 模板化 / 空话）？"),
]

# Bucket-specific: the same checklist for every sample is what made the old
# page misleading — it demanded a Brier gain from 140 abstentions that were
# admitted precisely because they do not claim one.
BUCKET_CHECKS = {
    "tool_positive": [
        ("verdict_correct", "结论与真值一致吗？与停止后验一致吗？"),
        ("gain_real", "相对 no-tool 基线确有增益吗（ΔBrier<0 且结论更正确）？"),
        ("evidence_used", "推理真的用了拿到的证据吗（而非背结论）？"),
        ("no_fabrication", "没有编造未收到的证据或数值吗？"),
    ],
    "no_tool_positive": [
        ("no_fabricated_evidence", "**没有虚构法证证据**吗？（无工具却写出证据段 = 必须拒绝）"),
        ("no_shorcut_real", "会不会教出「没有工具 ⇒ 判 Real」的捷径？（本桶 37R/0F，这是主要风险）"),
        ("verdict_correct", "结论与真值一致吗？视觉依据经得起看图复核吗？"),
    ],
    "honest_abstention": [
        ("abstention_reasoned", "Uncertain 是由弱证据 / 证据冲突 / 专家不适用造成的吗？（不要求 ΔBrier<0）"),
        ("no_lazy_abstention", "不是「一见不确定就弃权」的懒惰模式吗？"),
        ("remaining_uncertainty", "剩余不确定性写清楚了吗（缺什么证据、为何拿不到）？"),
    ],
}

DECISIONS = (("accept", "接受（内容与格式都可训练）"),
             ("format_only", "仅格式（内容不可信，只学格式）"),
             ("revise", "待修（修改后才可训练）"),
             ("reject", "拒绝（严重错误，永不训练）"))


def checklists_for(bucket: str) -> List[Tuple[str, str]]:
    """Common checks first, then the ones that only make sense for this bucket."""
    return COMMON_CHECKS + BUCKET_CHECKS.get(bucket, [])


def checklist_html(bucket: str) -> str:
    """Escaped checklist items; `**bold**` is the only markup they may carry."""
    items = []
    for name, text in checklists_for(bucket):
        escaped = html.escape(text).replace("**", "<strong>", 1)
        if "<strong>" in escaped:
            escaped = escaped.replace("**", "</strong>", 1)
        items.append(f'<li><label><input type="checkbox"> <strong>{html.escape(name)}</strong>'
                     f" — {escaped}</label></li>")
    return "".join(items)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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


def load_trajectories(directory: str = TRAJECTORIES_DIR) -> Dict[str, dict]:
    records = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name == "manifest.json":
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        records[record["trajectory_id"]] = record
    return records


def load_variant_entries(path: str = VARIANTS_MANIFEST) -> Dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {entry["variant_id"]: {**entry, "path": key}
            for key, entry in manifest.get("variants", {}).items()}


def load_split(path: Optional[str] = None) -> dict:
    path = path or os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Review sets
# ---------------------------------------------------------------------------

def high_risk_reasons(sample: dict, trajectory: Optional[dict]) -> List[str]:
    """
    Flags that make a sample worth reading slowly.

    These used to be framed as "needs a second reader"; the review is a single
    pass, so they are attention markers only.
    """
    reasons = []
    verdict = (sample.get("final_verdict") or {}).get("verdict")
    confidence = (sample.get("final_verdict") or {}).get("confidence") or 0.0
    tools = (sample.get("metadata") or {}).get("tools_served") or []
    conflict = (trajectory or {}).get("conflict_score") or 0.0

    if verdict == "Uncertain":
        reasons.append("uncertain")
    if len(tools) > 1:
        reasons.append("multi_tool")
    if confidence >= HIGH_RISK_CONFIDENCE and verdict in ("Real", "Fake"):
        reasons.append("high_confidence")
    if conflict > 0.5 or "conflict_unresolved" in ((trajectory or {}).get("policy_reasons") or []):
        reasons.append("conflict")
    return reasons


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _rel(path: str, html_path: str) -> str:
    """Relative URL from a page to an image, so nothing needs copying."""
    absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    return os.path.relpath(absolute, os.path.dirname(html_path)).replace(os.sep, "/")


def _img(path: str, page_path: str, caption: str, missing: set) -> str:
    if not path:
        return f'<figure class="missing"><figcaption>{html.escape(caption)}（未提供）</figcaption></figure>'
    absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(absolute):
        missing.add(path)
        return f'<figure class="missing"><figcaption>{html.escape(caption)}（文件缺失）</figcaption></figure>'
    return (f'<figure><img src="{_rel(path, page_path)}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def _text(value) -> str:
    """Escaped display text; None renders as an em dash rather than 'None'."""
    if value is None or value == "":
        return "—"
    return html.escape(str(value))


def _pretty(body: str) -> str:
    """Evidence turns arrive as one long JSON line; show them indented."""
    stripped = (body or "").strip()
    if not stripped.startswith("{"):
        return body or ""
    try:
        return json.dumps(json.loads(stripped), ensure_ascii=False, indent=1)
    except ValueError:
        return body


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def conversation_html(sample: dict, page_path: str, missing: set) -> Tuple[str, dict]:
    """
    Every turn, in order, marked with whether training supervises it.

    The previous version rendered only the first assistant turn, which hid the
    call round, the evidence round and the final four-section answer for 198 of
    the 237 samples — the reviewer was shown the least informative turn and
    asked to judge the sample anyway.
    """
    turns = sample.get("conversations") or []
    blocks = []
    assistant = 0
    for index, turn in enumerate(turns):
        role = turn.get("from", "?")
        supervised = role == "gpt"
        assistant += 1 if supervised else 0
        images = turn.get("image_paths") or []
        badge = ('<span class="badge sup">参与训练 · 被监督</span>' if supervised
                 else '<span class="badge mask">掩码 · 不进 loss</span>')
        head = (f"轮 {index} ｜ {'助手' if supervised else '用户/系统'} ｜ {badge}"
                + (f" ｜ 附 {len(images)} 张图" if images else ""))
        figures = "".join(
            _img(path, page_path, f"第 {index} 轮附图：{os.path.basename(path)}", missing)
            for path in images)
        blocks.append(
            f'<div class="turn {"gpt" if supervised else "user"}">'
            f'<div class="turnhead">{head}</div>'
            f'<pre class="turnbody">{html.escape(_pretty(turn.get("value", "")))}</pre>'
            f"{figures}</div>")
    summary = {
        "turns": len(turns),
        "assistant": assistant,
        "masked": len(turns) - assistant,
    }
    return "".join(blocks), summary


# ---------------------------------------------------------------------------
# Evidence Bundle
# ---------------------------------------------------------------------------

def _coords(token: dict) -> str:
    pixels = token.get("region_pixels")
    normalized = token.get("region_normalized_1000")
    space = token.get("coordinate_space")
    if not pixels and not normalized:
        return "—"
    return (f"像素 {pixels} ｜ 归一化(‰) {normalized}"
            + (f" ｜ 坐标空间 {space}" if space else ""))


def _condition_metadata(metadata) -> str:
    if not metadata:
        return "—"
    if not isinstance(metadata, dict):
        return _text(metadata)
    parts = []
    for key, value in metadata.items():
        if isinstance(value, float):
            value = round(value, 4)
        parts.append(f"{key}: {value}")
    return " ｜ ".join(html.escape(str(part)) for part in parts)


def evidence_card(token: dict, page_path: str, missing: set, index: int) -> str:
    """
    One card per token, with the direction chain spelled out.

    `support` is *not* "the direction": it is the expert's claim after the
    rectifier rewrote it, `support_raw` is what the expert said on its own,
    and the binding authority is `direction` + `direction_source`.  Collapsing
    those into one "方向" cell (as the previous table did) made the correction
    invisible — in 158 of 200 tokens the raw expert semantics point the
    opposite way.
    """
    likelihood = token.get("calibrated_likelihood") or {}
    aligned = token.get("semantics_aligned")
    warning = ""
    if aligned is False:
        warning = ('<div class="warn-red">⚠ <strong>semantics_aligned = false</strong>：'
                   '该专家的原始语义方向与校准实测相反（原始值越高，实际越可能是 Real）。'
                   '本 token 的方向由 <code>calibrated_likelihood</code> 决定，'
                   '下列「原始解释」只作历史记录，<strong>不得作为方向依据</strong>。</div>')
    elif aligned is None:
        warning = ('<div class="warn">semantics_aligned 缺失：该 token 没有校准条目，'
                   '方向沿用专家自述（direction_source = expert_claim_uncalibrated）。</div>')

    raw_support = token.get("support_raw")
    rows = [
        ("evidence_name", _text(token.get("evidence_name"))),
        ("专家", f"{_text(token.get('source'))}（{_text(token.get('evidence_id'))}）"),
        ("测量范围", f"{_text(token.get('measurement_scope'))} ｜ {_text(token.get('region_semantics'))}"),
        ("区域（region_pixels / region_normalized_1000）", _text(_coords(token))),
        ("原始数值 / 强度", f"raw_metric {_text(token.get('raw_metric'))} ｜ strength {_text(token.get('strength'))}"),
        ("校准似然", f"<strong>P(Fake) {_text(likelihood.get('Fake'))}</strong> ｜ P(Real) {_text(likelihood.get('Real'))}"),
        ("support（整流后主张）", _text(token.get("support"))),
        ("support_raw（专家原始主张）", _text(raw_support) if raw_support else "—（未被改写）"),
        ("direction ← direction_source（方向权威）",
         f"<strong>{_text(token.get('direction'))}</strong> ← {_text(token.get('direction_source'))}"),
        ("reliability（校准可靠性）", _text(token.get("reliability"))),
        ("applicability", _text(token.get("applicability"))),
        ("适用条件", _text(token.get("applicability_conditions"))),
        ("phenomenon（现象）", _text(token.get("phenomenon"))),
        ("reasoning（专家推理）", _text(token.get("reasoning"))),
        ("原始解释 interpretation_text_raw", _text(token.get("interpretation_text_raw"))),
        ("校准后解释 interpretation_text", _text(token.get("interpretation_text"))),
        ("反解释 counter_explanation", _text(token.get("counter_explanation"))),
        ("condition_metadata", _condition_metadata(token.get("condition_metadata"))),
    ]
    body = "".join(f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>"
                   for label, value in rows)

    figures = ""
    if token.get("diagnostic_region_image"):
        figures += _img(token["diagnostic_region_image"], page_path,
                        f"诊断区域（{token.get('source')}）", missing)
    for artifact in token.get("visual_artifacts") or []:
        figures += _img(artifact, page_path, f"专家产物 {os.path.basename(artifact)}", missing)

    return (f'<div class="card"><h3>Token {index + 1}：{_text(token.get("evidence_name"))}'
            f'<span class="pill">{_text(token.get("source"))}</span></h3>{warning}'
            f'<table class="kv">{body}</table>{figures}</div>')


def evidence_html(tokens: List[dict], page_path: str, missing: set) -> str:
    if not tokens:
        return '<p class="note">本轨迹未调用专家，没有 Evidence Token。</p>'
    cards = "".join(evidence_card(token, page_path, missing, index)
                    for index, token in enumerate(tokens))
    flipped = sum(1 for token in tokens if token.get("semantics_aligned") is False)
    return (f'<p class="note">共 {len(tokens)} 个 token，其中 '
            f'<strong>{flipped}</strong> 个 semantics_aligned=false（方向被校准纠正过）。</p>'
            + cards)


# ---------------------------------------------------------------------------
# Verdict, confidence and the baseline comparison
# ---------------------------------------------------------------------------

def _probability_row(label: str, value, note: str) -> str:
    return f"<tr><th>{html.escape(label)}</th><td>{_text(value)}</td><td class='note'>{html.escape(note)}</td></tr>"


def verdict_html(sample: dict, trajectory: Optional[dict],
                 baseline: Optional[dict]) -> str:
    """What the pipeline concluded, and what each number actually means."""
    verdict = sample.get("final_verdict") or {}
    metadata = sample.get("metadata") or {}
    scores = (trajectory or {}).get("scores") or {}
    baseline_scores = (baseline or {}).get("scores") or {}
    tool_free = not (metadata.get("tools_served") or [])
    posterior = verdict.get("posterior")
    if posterior is None and not tool_free:
        posterior = (trajectory or {}).get("posterior")
    if tool_free:
        # A tool-free session keeps the prior (0.5): there is no evidence to
        # derive a posterior from, and printing 0.5 as one would invent a
        # number the pipeline never computed.
        posterior = None
    model_probability = (trajectory or {}).get("model_probability")
    model_candidate = metadata.get("model_candidate") or (trajectory or {}).get("model_candidate")

    if posterior is None:
        likelihood_rows = (
            _probability_row("停止后验 P(Fake)", None,
                             "无工具轨迹没有后验（工具不可用，停止策略直接采用模型判定）") +
            _probability_row("停止后验 P(Real)", None, "同上"))
        confidence_note = "无工具轨迹：所选标签置信度是模型自报值（basis=model_perception），不是校准概率"
    else:
        likelihood_rows = (
            _probability_row("停止后验 P(Fake)", posterior, "Youden 加权 log-odds 后验") +
            _probability_row("停止后验 P(Real)", round(1 - float(posterior), 4),
                             "1 − P(Fake)"))
        confidence_note = ("工具轨迹：所选标签置信度取自停止后验（与 P(Fake)/P(Real) 同源）")

    delta = ""
    if scores.get("model_brier") is not None and scores.get("posterior_brier") is not None and not tool_free:
        delta_value = round(scores["model_brier"] - scores["posterior_brier"], 4)
        delta = _probability_row("ΔBrier（模型 → 后验）", delta_value,
                                 "负值 = 证据让预测更接近真值；这是工具轨迹的增益判据")

    rows = (
        _probability_row("原始模型判定", f"{model_candidate}（p={model_probability}）",
                         "模型自己选的标签与自报概率，未受后验约束") +
        likelihood_rows +
        _probability_row("所选标签置信度", verdict.get("confidence"), confidence_note) +
        _probability_row("模型 Brier", scores.get("model_brier"), "模型自报概率的 Brier 分数") +
        _probability_row("停止后验 Brier", scores.get("posterior_brier"),
                         "后验概率的 Brier 分数") + delta +
        _probability_row("停止原因", " / ".join(verdict.get("policy_reasons") or []),
                         f"halting_reason: {_text((trajectory or {}).get('halting_reason'))}")
    )

    baseline_row = ""
    if baseline:
        baseline_row = (
            "<tr><th>no-tool 基线</th>"
            f"<td>{_text(baseline.get('final_verdict'))}</td>"
            f"<td>{_text(baseline.get('posterior'))}</td>"
            f"<td>{_text(baseline_scores.get('posterior_brier'))}</td>"
            f"<td>{_text((baseline.get('counters') or {}).get('weighted_cost'))}</td></tr>")
    comparison = (
        "<table><thead><tr><th>来源</th><th>判定</th><th>后验 P(Fake)</th>"
        "<th>后验 Brier</th><th>加权成本</th></tr></thead><tbody>"
        f"<tr><th>本轨迹</th><td>{_text(verdict.get('verdict'))}</td>"
        f"<td>{_text(posterior)}</td>"
        f"<td>{_text(scores.get('posterior_brier'))}</td>"
        f"<td>{_text((metadata.get('counters') or {}).get('weighted_cost'))}</td></tr>"
        f"{baseline_row}</tbody></table>")
    if tool_free:
        comparison += ('<p class="note">无工具轨迹：基线与本轨迹是同一次运行，'
                       'ΔBrier 没有意义，不参与本桶判定。</p>')

    return f"<table class='kv'>{rows}</table>{comparison}"


# ---------------------------------------------------------------------------
# Page shell: decision toolbar, keyboard shortcuts, local storage
# ---------------------------------------------------------------------------

REVIEW_SCRIPT = r"""
/* One decision per sample, kept in the browser; export feeds record_review.py.
   No server: localStorage survives a refresh, the export is the record. */
(function () {
  var KEY = 'finalv2review.v1';
  var state = load();
  var storageOk = true;

  function load() {
    try {
      var parsed = JSON.parse(localStorage.getItem(KEY)) || {};
      if (!parsed.decisions) { parsed.decisions = {}; }
      if (!parsed.reviewer) { parsed.reviewer = ''; }
      return parsed;
    } catch (e) { return { reviewer: '', decisions: {} }; }
  }
  function persist() {
    try { localStorage.setItem(KEY, JSON.stringify(state)); return true; }
    catch (e) { return false; }
  }
  function decisionOf(id) { return (state.decisions || {})[id] || null; }

  function flash(message) {
    var box = document.getElementById('save-status');
    if (box) { box.textContent = message; }
  }
  function count() {
    return Object.keys(state.decisions || {}).filter(function (k) {
      return state.decisions[k] && state.decisions[k].decision;
    }).length;
  }
  function refreshStatus() {
    var total = document.body.getAttribute('data-samples');
    var decided = count();
    flash('本地已记录 ' + decided + (total ? ' / ' + total : '') + ' 条（导出后才写入仓库）');
    var progress = document.getElementById('progress');
    if (progress && total) { progress.textContent = decided + ' / ' + total; }
  }

  function exportAll() {
    var payload = {
      reviewer: (document.getElementById('reviewer') || {}).value || state.reviewer || '',
      exported_at: new Date().toISOString(),
      decisions: state.decisions || {}
    };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'review_decisions.json';
    document.body.appendChild(link); link.click(); link.remove();
  }

  function importFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var parsed = JSON.parse(reader.result);
        var incoming = parsed.decisions || parsed;
        Object.keys(incoming).forEach(function (id) { state.decisions[id] = incoming[id]; });
        if (parsed.reviewer) { state.reviewer = parsed.reviewer; }
        persist(); refreshStatus(); renderIndex();
        flash('已导入 ' + Object.keys(incoming).length + ' 条');
      } catch (e) { flash('导入失败：不是有效的 JSON'); }
    };
    reader.readAsText(file);
  }

  function apply(id, decision, notes) {
    state.decisions[id] = { decision: decision, notes: notes || '', saved_at: new Date().toISOString() };
    storageOk = persist();
    if (!storageOk) { flash('浏览器禁止本地保存：请改用导出/导入，或经 http 服务打开本页'); }
    refreshStatus();
    renderIndex();
    var buttons = document.querySelectorAll('[data-decision]');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].classList.toggle('active', buttons[i].getAttribute('data-decision') === decision);
    }
    var history = document.getElementById('history');
    if (history) {
      history.innerHTML = '<li>' + new Date().toLocaleString() + ' → <strong>' + decision +
        '</strong>' + (notes ? '：' + notes.replace(/</g, '&lt;') : '') + '</li>' + history.innerHTML;
    }
  }

  function nextLink() { return document.getElementById('go-next'); }
  function prevLink() { return document.getElementById('go-prev'); }

  function renderIndex() {
    var rows = document.querySelectorAll('[data-row-sample]');
    for (var i = 0; i < rows.length; i++) {
      var id = rows[i].getAttribute('data-row-sample');
      var entry = decisionOf(id);
      var cell = rows[i].querySelector('[data-cell="decision"]');
      var note = rows[i].querySelector('[data-cell="notes"]');
      var done = entry && entry.decision ? entry.decision : '';
      if (cell) { cell.textContent = done || '—'; }
      if (note) { note.textContent = (entry && entry.notes) || ''; }
      rows[i].classList.toggle('decided', !!done);
      rows[i].classList.toggle('pending', !done);
    }
    var only = document.getElementById('only-pending');
    for (var j = 0; j < rows.length; j++) {
      var decided = rows[j].classList.contains('decided');
      var hide = (only && only.checked && decided) ||
        (document.getElementById('only-high-risk') && document.getElementById('only-high-risk').checked &&
         rows[j].getAttribute('data-high-risk') !== '1');
      rows[j].style.display = hide ? 'none' : '';
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    var id = document.body.getAttribute('data-sample');
    var notes = document.getElementById('notes');
    var reviewer = document.getElementById('reviewer');
    if (reviewer) {
      reviewer.value = state.reviewer || '';
      reviewer.addEventListener('change', function () { state.reviewer = reviewer.value; persist(); });
    }
    if (id && notes) {
      var entry = decisionOf(id);
      notes.value = (entry && entry.notes) || '';
      notes.addEventListener('change', function () {
        var existing = decisionOf(id);
        if (existing) { apply(id, existing.decision, notes.value); }
      });
    }
    var buttons = document.querySelectorAll('[data-decision]');
    for (var i = 0; i < buttons.length; i++) {
      if (id && decisionOf(id) &&
          buttons[i].getAttribute('data-decision') === decisionOf(id).decision) {
        buttons[i].classList.add('active');
      }
      buttons[i].addEventListener('click', function (event) {
        var decision = event.currentTarget.getAttribute('data-decision');
        if (!id) { return; }
        apply(id, decision, notes ? notes.value : '');
        var jump = document.getElementById('auto-advance');
        if (jump && jump.checked && decision !== 'revise' && nextLink()) {
          setTimeout(function () { nextLink().click(); }, 250);
        }
      });
    }
    var exportButton = document.getElementById('export');
    if (exportButton) { exportButton.addEventListener('click', exportAll); }
    var importInput = document.getElementById('import');
    if (importInput) {
      importInput.addEventListener('change', function (event) {
        if (event.target.files.length) { importFile(event.target.files[0]); }
      });
    }
    var onlyPending = document.getElementById('only-pending');
    var onlyRisk = document.getElementById('only-high-risk');
    if (onlyPending) { onlyPending.addEventListener('change', renderIndex); }
    if (onlyRisk) { onlyRisk.addEventListener('change', renderIndex); }

    document.addEventListener('keydown', function (event) {
      var tag = (event.target.tagName || '').toLowerCase();
      if (tag === 'textarea' || tag === 'input' || tag === 'select') { return; }
      var map = { '1': 'accept', '2': 'format_only', '3': 'revise', '4': 'reject' };
      if (id && map[event.key]) {
        var button = document.querySelector('[data-decision="' + map[event.key] + '"]');
        if (button) { button.click(); event.preventDefault(); }
      } else if (event.key === 'ArrowLeft' && prevLink()) { prevLink().click(); }
      else if (event.key === 'ArrowRight' && nextLink()) { nextLink().click(); }
      else if (event.key === 'e') { exportAll(); }
    });

    if (!(function () { try { localStorage.setItem('probe', '1'); localStorage.removeItem('probe'); return true; }
                        catch (e) { return false; } })()) {
      var banner = document.createElement('div');
      banner.className = 'warn-red';
      banner.textContent = '本页无法使用浏览器本地保存（file:// 限制）：请在仓库目录运行 ' +
        'python3 -m http.server 8000 后经 http://localhost:8000/sft_data/review/ 打开，' +
        '否则刷新会丢失勾选。';
      document.body.insertBefore(banner, document.body.firstChild);
    }
    refreshStatus();
    renderIndex();
  });
})();
"""

STYLE = """
 body { font-family: system-ui, sans-serif; margin: 24px; max-width: 1200px; color: #222; }
 h1 { font-size: 18px; } h2 { font-size: 15px; margin-top: 28px; border-bottom: 1px solid #ddd; }
 h3 { font-size: 14px; margin: 4px 0; }
 img { max-width: 420px; max-height: 420px; border: 1px solid #ccc; }
 figure { display: inline-block; margin: 6px 12px 6px 0; vertical-align: top; }
 figcaption { font-size: 12px; color: #555; max-width: 420px; }
 .missing { background: #f6f6f6; color: #999; padding: 20px; font-size: 12px; }
 table { border-collapse: collapse; font-size: 13px; width: 100%; margin-bottom: 10px; }
 th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: top; }
 th { background: #f4f4f4; }
 table.kv th { width: 260px; background: #fafafa; font-weight: 600; }
 .note { color: #666; font-size: 12px; }
 .flag { background: #fff3cd; border-left: 4px solid #e0a800; padding: 8px 12px; margin: 6px 0; }
 .warn { background: #fff9e6; border-left: 4px solid #e0a800; padding: 8px 12px; margin: 6px 0; font-size: 13px; }
 .warn-red { background: #fdecea; border-left: 4px solid #d93025; padding: 8px 12px; margin: 6px 0; font-size: 13px; }
 .ok { color: #1a7f37; }
 .badge { font-size: 11px; padding: 1px 6px; border-radius: 8px; }
 .badge.sup { background: #e6f4ea; color: #137333; border: 1px solid #b7e1c4; }
 .badge.mask { background: #eee; color: #666; border: 1px solid #ddd; }
 .turn { border: 1px solid #e5e5e5; border-radius: 4px; margin: 8px 0; }
 .turn.gpt { border-left: 4px solid #137333; }
 .turn.user { border-left: 4px solid #9aa0a6; }
 .turnhead { background: #fafafa; padding: 4px 8px; font-size: 12px; color: #444; border-bottom: 1px solid #eee; }
 .turnbody { margin: 0; padding: 8px; white-space: pre-wrap; font-size: 12.5px; max-height: 420px; overflow: auto; }
 .card { border: 1px solid #e5e5e5; border-radius: 4px; padding: 10px; margin: 12px 0; }
 .pill { font-size: 11px; background: #eef; border: 1px solid #ccd; padding: 1px 6px; border-radius: 8px; margin-left: 8px; }
 ul.checklist { list-style: none; padding-left: 0; font-size: 13px; }
 ul.checklist li { margin: 4px 0; }
 .toolbar { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd; padding: 8px 0; z-index: 5; }
 .toolbar button { font-size: 13px; padding: 4px 10px; margin-right: 6px; border: 1px solid #bbb;
                   background: #f8f8f8; border-radius: 4px; cursor: pointer; }
 .toolbar button.active { background: #137333; color: #fff; border-color: #137333; }
 .toolbar textarea { width: 100%; height: 46px; font-size: 13px; margin-top: 6px; }
 .status { font-size: 12px; color: #555; }
 tr.decided td[data-cell="decision"] { color: #137333; font-weight: 600; }
 tr.pending td[data-cell="decision"] { color: #999; }
"""


def toolbar_html(sample_id: str, prev_id: Optional[str], next_id: Optional[str]) -> str:
    nav = []
    if prev_id:
        nav.append(f'<a id="go-prev" href="{html.escape(prev_id)}.html">← 上一条</a>')
    if next_id:
        nav.append(f'<a id="go-next" href="{html.escape(next_id)}.html">下一条 →</a>')
    buttons = "".join(
        f'<button data-decision="{key}" title="快捷键 {index + 1}">{html.escape(label)}</button>'
        for index, (key, label) in enumerate(DECISIONS))
    return (f'<div class="toolbar">{" ｜ ".join(nav)} ｜ {buttons}'
            f'<label class="status"><input type="checkbox" id="auto-advance" checked> 决定后自动跳下一条</label>'
            f'<div><textarea id="notes" placeholder="备注（必填于 revise / reject）：问题在哪、为什么"></textarea></div>'
            f'<div class="status"><label>审核人 <input id="reviewer" size="8" placeholder="你的名字"></label>'
            f' ｜ <span id="save-status">—</span>'
            f' ｜ <button id="export">导出全部决定</button>'
            f'<label class="status"> 导入 <input type="file" id="import" accept=".json"></label>'
            f' ｜ 快捷键：1 接受 / 2 仅格式 / 3 待修 / 4 拒绝 / ← → 翻页 / e 导出</div></div>')


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_sample_page(sample: dict, trajectory: Optional[dict],
                       baseline: Optional[dict], source_path: str,
                       index_path: str, page_path: str, missing: set,
                       high_risk: List[str],
                       prev_id: Optional[str] = None,
                       next_id: Optional[str] = None,
                       total_samples: int = 0) -> str:
    metadata = sample.get("metadata") or {}
    verdict = sample.get("final_verdict") or {}
    tokens = sample.get("evidence_chain") or []
    bucket = sample.get("_bucket_file", "")

    images = [_img(sample.get("image_path"), page_path,
                   f"模型实际输入的变体（{metadata.get('treatment')}）", missing),
              _img(source_path, page_path, "数据集源图（未经处理）", missing)]

    conversation, turns = conversation_html(sample, page_path, missing)
    checklist = checklist_html(bucket)
    flags = "".join(f'<div class="flag">高风险标记：{html.escape(reason)}</div>'
                    for reason in high_risk)

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>审核 {html.escape(sample['id'])}</title>
<style>{STYLE}</style></head>
<body data-sample="{html.escape(sample['id'])}" data-samples="{total_samples or ''}">
{toolbar_html(sample['id'], prev_id, next_id)}
<p><a href="../{os.path.basename(index_path)}">← 返回索引</a></p>
<h1>{html.escape(sample['id'])}</h1>
<p>桶：<strong>{html.escape(bucket)}</strong>（{html.escape(BUCKET_LABELS.get(bucket, ''))}）
 ｜ 真值：<strong>{html.escape(sample['ground_truth'])}</strong>
 ｜ 策略：{html.escape(str(metadata.get('policy')))} ｜ 处理：{html.escape(str(metadata.get('treatment')))}
 ｜ 生成器：{html.escape(str(sample.get('source_model')))} ｜ 分区：{html.escape(str(metadata.get('split')))}</p>
{flags}

<h2>图像证据</h2>
{''.join(images)}

<h2>判定与置信度（各自的含义不同，别混用）</h2>
{verdict_html(sample, trajectory, baseline)}

<h2>完整训练对话（共 {turns['turns']} 轮：助手 {turns['assistant']} 轮参与训练，用户/系统 {turns['masked']} 轮掩码）</h2>
<p class="note">训练脚本只对助手轮计算 loss（<code>assistant_spans</code>）：绿色左边框的轮次就是模型要学的内容。
上面这条对话（含工具调用轮、证据轮）就是该样本将要教给模型的全部内容。</p>
{conversation}

<h2>Evidence Bundle（逐 token）</h2>
{evidence_html(tokens, page_path, missing)}

<h2>审核清单（本桶适用，逐项勾选）</h2>
<ul class="checklist">{checklist}</ul>
<p class="note">决定写在页面顶部的工具条，或直接用快捷键。导出后执行：
<code>python scripts/record_review.py --import review_decisions.json --reviewer &lt;你&gt;</code></p>
<h2>本页的修改历史（浏览器本地）</h2>
<ul id="history" class="note"></ul>
<script>window.__review = {{sample: {json.dumps(sample['id'], ensure_ascii=False)}}};</script>
<script>{REVIEW_SCRIPT}</script>
</body></html>
"""


def render_index(samples: List[dict], sets: dict, page_path: str,
                 missing_count: int) -> str:
    rows = []
    for sample in sorted(samples, key=lambda s: s["id"]):
        metadata = sample.get("metadata") or {}
        verdict = sample.get("final_verdict") or {}
        trajectory_posterior = verdict.get("posterior")
        risks = sets["high_risk"].get(sample["id"], [])
        rows.append(
            f'<tr data-row-sample="{html.escape(sample["id"])}" '
            f'data-high-risk="{1 if risks else 0}">'
            f'<td><a href="samples/{html.escape(sample["id"])}.html">{html.escape(sample["id"])}</a></td>'
            f"<td>{html.escape(sample.get('_bucket_file', ''))}</td>"
            f"<td>{html.escape(sample['ground_truth'])}</td>"
            f"<td>{html.escape(str(metadata.get('policy')))}</td>"
            f"<td>{html.escape(str(metadata.get('treatment')))}</td>"
            f"<td>{html.escape(str(sample.get('source_model')))}</td>"
            f"<td>{html.escape(str(verdict.get('verdict')))}</td>"
            f"<td>{_text(trajectory_posterior)}</td>"
            f"<td class='note'>{html.escape(', '.join(risks)) or '—'}</td>"
            f'<td data-cell="decision">—</td>'
            f'<td data-cell="notes" class="note"></td></tr>')
    checklist = "".join(
        f"<li><strong>{html.escape(BUCKET_LABELS.get(bucket, bucket))}</strong>："
        + "".join(f"{html.escape(name)}" + "、" for name, _ in BUCKET_CHECKS.get(bucket, []))[:-1]
        + "</li>" for bucket in BUCKET_LABELS)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>final_v2 审核工作台</title>
<style>{STYLE}</style></head>
<body data-samples="{len(samples)}">
<h1>final_v2 人工审核工作台（单次全量审核）</h1>
<p>样本 {len(samples)} 条 ｜ 桶：{html.escape(str(dict(Counter(s.get('_bucket_file') for s in samples))))}
 ｜ 高风险标记 {len(sets['high_risk'])} 条（仅提示重点，不触发第二次阅读）</p>
<div class="warn">审核不能只看格式。逐条打开样本页，按<strong>本桶的清单</strong>看源图与变体图、
完整训练对话、Evidence Bundle、与 no-tool 基线的对比，再判断这条样本会教给模型什么。</div>
<div class="toolbar">
  <span class="status">进度：<strong id="progress"></strong> ｜ <span id="save-status">—</span></span>
  <label class="status"><input type="checkbox" id="only-pending"> 只看未决</label>
  <label class="status"><input type="checkbox" id="only-high-risk"> 只看高风险</label>
  <label class="status">审核人 <input id="reviewer" size="8" placeholder="你的名字"></label>
  <button id="export">导出全部决定</button>
  <label class="status">导入 <input type="file" id="import" accept=".json"></label>
</div>
<h2>各桶的清单要点</h2><ul class="checklist">{checklist}</ul>
<p class="note">决定直接点样本页的按钮（快捷键 1–4），保存在浏览器本地；导出后运行
<code>python scripts/record_review.py --import review_decisions.json --reviewer &lt;你&gt;</code>
写入仓库。也可以填 <code>worklist.csv</code> 后由同一个 <code>--import</code> 读取。</p>
{f'<div class="warn">有 {missing_count} 张图片文件缺失，已在对应样本页标注。</div>' if missing_count else ''}
<table><thead><tr><th>样本</th><th>桶</th><th>真值</th><th>策略</th><th>处理</th><th>生成器</th>
<th>模型判定</th><th>后验 P(Fake)</th><th>高风险</th><th>你的结论</th><th>备注</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>{REVIEW_SCRIPT}</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(final_v2_dir: str = FINAL_V2_DIR, output_dir: str = OUTPUT_DIR,
          trajectories_dir: str = TRAJECTORIES_DIR,
          variants_manifest: str = VARIANTS_MANIFEST,
          split_path: Optional[str] = None) -> dict:
    samples = load_samples(final_v2_dir)
    if not samples:
        raise SystemExit(f"no samples in {final_v2_dir}")
    trajectories = load_trajectories(trajectories_dir)
    variants = load_variant_entries(variants_manifest)
    split = load_split(split_path)

    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    index_path = os.path.join(output_dir, "index.html")
    missing: set = set()
    high_risk: Dict[str, List[str]] = {}
    ordered = sorted(samples, key=lambda s: s["id"])

    for position, sample in enumerate(ordered):
        metadata = sample.get("metadata") or {}
        trajectory_id = metadata.get("trajectory_id")
        trajectory = trajectories.get(trajectory_id)
        variant = variants.get(trajectory_id.split("__")[-1] if trajectory_id else "")
        baseline = trajectories.get(
            f"no-tool__{trajectory_id.split('__')[-1]}") if trajectory_id else None
        source_path = ""
        if variant:
            source_path = (split.get("sources", {})
                           .get(variant["source_id"], {}).get("path", ""))
        reasons = high_risk_reasons(sample, trajectory)
        if reasons:
            high_risk[sample["id"]] = reasons

        page_path = os.path.join(output_dir, "samples", f"{sample['id']}.html")
        previous = ordered[position - 1]["id"] if position else None
        following = ordered[position + 1]["id"] if position + 1 < len(ordered) else None
        with open(page_path, "w", encoding="utf-8") as handle:
            handle.write(render_sample_page(sample, trajectory, baseline, source_path,
                                            index_path, page_path, missing, reasons,
                                            previous, following, len(ordered)))

    sets = {
        "protocol": "single_pass",
        "full": [s["id"] for s in ordered],
        "high_risk": high_risk,
    }
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(render_index(samples, sets, index_path, len(missing)))
    with open(os.path.join(output_dir, "review_sets.json"), "w", encoding="utf-8") as handle:
        json.dump(sets, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "worklist.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "bucket", "ground_truth", "policy", "treatment",
                         "generator", "model_verdict", "confidence", "posterior_fake",
                         "high_risk", "reviewer", "decision", "notes"])
        for sample in ordered:
            metadata = sample.get("metadata") or {}
            verdict = sample.get("final_verdict") or {}
            writer.writerow([
                sample["id"], sample.get("_bucket_file"), sample["ground_truth"],
                metadata.get("policy"), metadata.get("treatment"),
                sample.get("source_model"), verdict.get("verdict"),
                verdict.get("confidence"), verdict.get("posterior"),
                "|".join(high_risk.get(sample["id"], [])), "", "", "",
            ])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "samples": len(samples),
        "buckets": dict(Counter(s.get("_bucket_file") for s in samples)),
        "high_risk": len(high_risk),
        "high_risk_reasons": dict(Counter(
            r for reasons in high_risk.values() for r in reasons)),
        "missing_images": len(missing),
        "index": os.path.relpath(index_path, PROJECT_ROOT).replace(os.sep, "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v2-dir", default=FINAL_V2_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--trajectories-dir", default=TRAJECTORIES_DIR)
    parser.add_argument("--variants-manifest", default=VARIANTS_MANIFEST)
    parser.add_argument("--split-path", default=None)
    args = parser.parse_args()

    report = build(args.final_v2_dir, args.output_dir, args.trajectories_dir,
                   args.variants_manifest, args.split_path)
    print(f"samples: {report['samples']}  buckets: {report['buckets']}")
    print(f"high risk: {report['high_risk']}  {report['high_risk_reasons']}")
    print(f"missing images: {report['missing_images']}")
    print(f"Open: {report['index']}")


if __name__ == "__main__":
    main()

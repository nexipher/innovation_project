"""
g8 验收审核工作台（阶段 4.6）。

把"看一眼"升级成"能审"：逐条给决定、看进度、筛、导出、导入；同时把**证据的出处**
摆在页面上——原文按 trace 逐字展示、投递包用**当时存下来的那段**、后台字段另折叠、
模型侧调用日志有就有、没有就写"未记录"。

八条要求落在哪里：

1. 审核动作 → 页面里的决定区（通过/待修/拒绝 + 审核人 + 分类 + 备注；待修/拒绝必须
   写原因）＋ 上一条/下一条 ＋ 进度 ＋ 筛选 ＋ 导出/导入；草稿存 `localStorage`，
   刷新不丢。
2. 身份隔离 → 每条样本的**内容指纹由 Python 算好嵌进页面**（`utils.review_store`），
   JS 只做字符串比对，不重算；导入时 generation/batch/schema 不匹配一律拒；
   指纹不符 → 标**已过期**、不继承。
3. 验收≠训练 → 决定只有"本次审核通过"，页面与导入都不写任何训练产物；也不接
   `admit_trajectories_g4`（那会把 32 条全挡在外面）。
4. 原文可核对 → 「已记录的会话原文」；逐轮标来源；助手文本原样；每轮给原始文本折叠
   与整份 trace 下载。
5. 投递与后台分开 → 主区用 **trace 里保存的投递原文**（不再用今天的 `model_visible()`
   重新生成冒充历史输入）；后台 token 全字段另折叠；摘要截断但可展开。
6. 审计信息 → 原图/实际输入变体/尺寸来源/模型与专家版本/停止原因/请求-执行-重复计数
   /引用检查的**具体发现文本**；未记录的项明确标注。
7. 日志完整性 → 每批一张表，逐项标已记录/未记录（b1 缺三项，如实写）。
8. 验证 → 见 `tests/test_g8_review_workstation.py` 与 `tests/test_review_store.py`。

用法：
  python scripts/g8_review_workstation.py --generation g8 --batch b1
  python scripts/g8_review_import.py <导出文件> --reviewer <你>   # 写回仓库（另一步）
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from state_machine.controller import ForensicStateMachine  # noqa: E402
from utils.batch import batch_paths  # noqa: E402
from utils.review_store import (CATEGORIES, DECISIONS, SCHEMA,  # noqa: E402
                                latest, load_ledger, progress, sample_fingerprint)

CSS = """
:root { color-scheme: light dark; --line:#8884; --bg:#8881; }
* { box-sizing: border-box; }
body { font: 15px/1.6 -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
       margin: 0; padding: 16px 26px 60px; max-width: 1280px;
       background: canvas; color: canvastext; }
a { color: #2b7; }
h1 { font-size: 21px; margin: 0 0 6px; } h2 { font-size: 16px; margin: 26px 0 8px; }
h3 { font-size: 14px; margin: 16px 0 6px; opacity: .85; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left;
         vertical-align: top; font-size: 14px; }
th { background: var(--bg); position: sticky; top: 0; }
img { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
/* 链接：未访问/已访问用不同色相区分（蓝 vs 紫），深浅色各一套；
   已访问只是**浏览器里的浏览记录**，与"是否已审核"无关——审核状态另有标签。 */
a:link { color: #0b5fb0; text-decoration-color: #0b5fb066; }
a:visited { color: #8a3fb8; text-decoration-color: #8a3fb866; }
a:hover { text-decoration: underline; }
a:focus-visible { outline: 2px solid #0b5fb0; outline-offset: 2px; border-radius: 3px; }
@media (prefers-color-scheme: dark) {
  a:link { color: #74b9ff; text-decoration-color: #74b9ff66; }
  a:visited { color: #cba0ff; text-decoration-color: #cba0ff66; }
  a:focus-visible { outline-color: #74b9ff; }
}
/* 顶部固定审核栏：滚动时始终可见，不透明背景 + 明确的层叠顺序 */
.topbar { position: sticky; top: 0; z-index: 20; background: canvas;
          border-bottom: 1px solid var(--line); padding: 6px 0 10px;
          margin: -10px 0 14px; }
.topbar .row { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }
.topbar h1 { font-size: 18px; margin: 2px 0; }
.topbar .crumbs { font-size: 13px; }
.topbar textarea { min-height: 38px; max-height: 96px; resize: vertical; }
.topbar .controls { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
@media (max-width: 720px) { .topbar .row { gap: 4px 8px; } .topbar h1 { font-size: 16px; } }
.turn { border-left: 3px solid var(--line); margin: 10px 0; padding: 4px 12px; }
.turn.model { border-color: #37b; } .turn.tool { border-color: #2b7; }
.turn.control { border-color: #c93; }
.role { font-weight: 600; font-size: 13px; }
.role .kind { font-weight: 400; opacity: .75; }
pre { white-space: pre-wrap; word-break: break-word; margin: 6px 0; background: var(--bg);
      padding: 10px; border-radius: 6px; font-size: 13px; max-height: 340px; overflow: auto; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px; background: #8883;
        font-size: 12.5px; margin: 0 6px 4px 0; }
.ok { background: #2b73; } .bad { background: #b223; } .warn { background: #b903; }
.muted { opacity: .72; font-size: 13px; }
.bar { height: 10px; background: var(--bg); border-radius: 6px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: #2b7; }
button { font: inherit; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--line);
         background: var(--bg); cursor: pointer; }
button.primary { background: #2b7; color: #fff; border-color: #2b7; }
button.danger { background: #b22; color: #fff; border-color: #b22; }
button.warn { background: #c93; color: #fff; border-color: #c93; }
input[type=text], textarea, select { font: inherit; background: transparent; color: inherit;
        border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; }
textarea { width: 100%; min-height: 54px; }
.stale { background: #b903; }
details > summary { cursor: pointer; font-size: 13.5px; opacity: .85; }
.banner { border: 1px solid var(--line); border-left: 4px solid #c93; padding: 10px 12px;
          border-radius: 6px; background: var(--bg); }
.filters { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 10px 0; }
tr.hidden { display: none; }
"""

JS = r"""
const META = __META__;
const KEY = "g8review:" + META.generation + ":" + META.batch;
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function loadDraft() {
  try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { return {}; }
}
function saveDraft(draft) {
  localStorage.setItem(KEY, JSON.stringify(draft));
  flash("已存到浏览器（草稿）。**这还没有写回仓库**——请用「导出」把文件交给导入命令。");
}
function flash(message) {
  const box = $("#flash");
  if (!box) return;
  box.textContent = message; box.hidden = false;
  clearTimeout(flash._t); flash._t = setTimeout(() => { box.hidden = true; }, 6000);
}

/** 仓库台账 + 浏览器草稿 → 每条现在的状态与是否过期。 */
function stateOf(id) {
  const sample = META.samples[id] || {};
  const repo = META.repo_decisions[id] || null;
  const draft = loadDraft()[id] || null;
  const pick = draft || repo;
  if (!pick) return { decision: null, stale: false, draft: false, repo: repo, record: null };
  const stale = pick.fingerprint !== sample.fingerprint;
  return { decision: pick.decision, stale: stale, draft: !!draft, repo: repo, record: pick };
}

function renderProgress() {
  const ids = Object.keys(META.samples);
  let decided = 0, stale = 0, draftOnly = 0;
  const by = { pass: 0, revise: 0, reject: 0 };
  for (const id of ids) {
    const state = stateOf(id);
    if (state.record && !state.stale) { decided++; if (state.decision in by) by[state.decision]++; }
    if (state.record && state.stale) stale++;
    if (state.record && state.draft) draftOnly++;
  }
  const pct = ids.length ? Math.round(100 * decided / ids.length) : 0;
  const bar = $("#progress"); if (bar) bar.innerHTML = `<span style="width:${pct}%"></span>`;
  const text = $("#progress-text");
  if (text) text.textContent =
    `已决定 ${decided}/${ids.length}（${pct}%）｜通过 ${by.pass}｜待修 ${by.revise}｜拒绝 ${by.reject}` +
    `｜已过期 ${stale}｜浏览器草稿未导出 ${draftOnly}`;
}

function persist(id, decision, category, note) {
  const sample = META.samples[id];
  const reviewer = ($("#reviewer") && $("#reviewer").value || "").trim();
  if (!reviewer) { flash("请先填审核人。"); return false; }
  if ((decision === "revise" || decision === "reject") && !note.trim()) {
    flash("待修/拒绝必须填写原因。"); return false;
  }
  const draft = loadDraft();
  draft[id] = { fingerprint: sample.fingerprint, decision: decision, reviewer: reviewer,
                category: category || null, note: note,
                decided_at: new Date().toISOString().slice(0, 19) };
  saveDraft(draft);
  return true;
}

function applyFilters() {
  const wantDecision = ($("#f-decision") || {}).value || "";
  const wantTreatment = ($("#f-treatment") || {}).value || "";
  const wantOutcome = ($("#f-outcome") || {}).value || "";
  for (const row of $$("tr[data-id]")) {
    const state = stateOf(row.dataset.id);
    const key = !state.record ? "undecided" : (state.stale ? "stale" : state.decision);
    let show = true;
    if (wantDecision && key !== wantDecision) show = false;
    if (wantTreatment && row.dataset.treatment !== wantTreatment) show = false;
    if (wantOutcome && row.dataset.outcome !== wantOutcome) show = false;
    row.classList.toggle("hidden", !show);
  }
}

function buildExport() {
  const reviewer = (($("#reviewer") || {}).value || "").trim();
  const draft = loadDraft();
  const decisions = {}; const ids = Object.keys(draft);
  ids.forEach(id => {
    const entry = draft[id];
    const fp = META.samples[id] && META.samples[id].fingerprint;
    if (fp && entry.fingerprint === fp) decisions[id] = entry;
    else decisions[id] = Object.assign({}, entry, { stale: true, stale_reason: "内容已变" });
  });
  return { schema: META.schema, generation: META.generation, batch: META.batch,
           reviewer: reviewer, exported_at: new Date().toISOString().slice(0, 19),
           scope: "acceptance_only", decisions: decisions };
}

function doExport() {
  const payload = buildExport();
  const count = Object.keys(payload.decisions).length;
  if (!count) { flash("还没有任何决定可导出。"); return; }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `review_${META.generation}_${META.batch}.json`;
  link.click();
  flash(`已导出 ${count} 条。把文件交给：python scripts/g8_review_import.py <文件> --reviewer ${payload.reviewer || "…"}`);
}

function doImport(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let payload;
    try { payload = JSON.parse(reader.result); }
    catch (e) { flash("导入失败：不是合法 JSON。"); return; }
    if (payload.schema !== META.schema) { flash("导入被拒：schema 不匹配。"); return; }
    if (payload.generation !== META.generation || payload.batch !== META.batch) {
      flash(`导入被拒：文件属于 ${payload.generation}/${payload.batch}，本页是 ` +
            `${META.generation}/${META.batch}（跨代次/跨批次一律拒绝）。`);
      return;
    }
    const draft = loadDraft(); let taken = 0, stale = 0, refused = 0;
    for (const [id, entry] of Object.entries(payload.decisions || {})) {
      const sample = META.samples[id];
      if (!sample) { refused++; continue; }
      if ((entry.decision === "revise" || entry.decision === "reject") && !(entry.note || "").trim()) {
        refused++; continue;
      }
      if (entry.fingerprint !== sample.fingerprint) stale++;
      draft[id] = entry; taken++;
    }
    saveDraft(draft);
    flash(`导入 ${taken} 条（其中 ${stale} 条与当前内容不符，已标过期）；拒绝 ${refused} 条。`);
    renderProgress(); applyFilters();
    $$("tr[data-id]").forEach(row => paintRow(row));
  };
  reader.readAsText(file);
}

/** 一条样本页上的决定区。 */
function initDecide() {
  const box = $("#decide"); if (!box) return;
  const id = box.dataset.id;
  const sample = META.samples[id];
  const state = stateOf(id);
  const set = (decision) => {
    const category = ($("#category") || {}).value || "";
    const note = ($("#note") || {}).value || "";
    if (!persist(id, decision, category, note)) return;
    paint();
  };
  const paint = () => {
    const now = stateOf(id);
    const badge = $("#decide-state");
    if (badge) {
      if (!now.record) badge.innerHTML = '<span class="pill">未决定</span>';
      else {
        const label = { pass: "通过（本次审核）", revise: "待修", reject: "拒绝" }[now.decision] || now.decision;
        badge.innerHTML = `<span class="pill ${now.decision === "pass" ? "ok" : (now.decision === "reject" ? "bad" : "warn")}">${label}</span>` +
          `<span class="pill">审核人 ${now.record.reviewer || ""}</span>` +
          (now.category ? `<span class="pill">分类 ${now.category}</span>` : "") +
          (now.stale ? '<span class="pill stale">已过期：内容已变，需重审</span>' : "") +
          (now.draft ? '<span class="pill warn">浏览器草稿（未写回仓库）</span>' : '<span class="pill">仓库台账</span>');
      }
    }
    if (now.record) {
      if ($("#note")) $("#note").value = now.record.note || "";
      if ($("#category")) $("#category").value = now.record.category || "";
    }
  };
  $("#btn-pass") && ($("#btn-pass").onclick = () => set("pass"));
  $("#btn-revise") && ($("#btn-revise").onclick = () => set("revise"));
  $("#btn-reject") && ($("#btn-reject").onclick = () => set("reject"));
  paint();
}

function paintRow(row) {
  const cell = row.querySelector("[data-state]");
  if (!cell) return;
  const state = stateOf(row.dataset.id);
  if (!state.record) { cell.innerHTML = '<span class="pill">未决定</span>'; return; }
  const label = { pass: "通过", revise: "待修", reject: "拒绝" }[state.decision] || state.decision;
  const cls = state.decision === "pass" ? "ok" : (state.decision === "reject" ? "bad" : "warn");
  cell.innerHTML = `<span class="pill ${cls}">${label}</span>` +
    (state.stale ? '<span class="pill stale">过期</span>' : "") +
    (state.draft ? '<span class="pill warn">草稿</span>' : "");
}

document.addEventListener("DOMContentLoaded", () => {
  const reviewer = localStorage.getItem(KEY + ":reviewer");
  if (reviewer && $("#reviewer")) $("#reviewer").value = reviewer;
  if ($("#reviewer")) $("#reviewer").addEventListener("change", () =>
    localStorage.setItem(KEY + ":reviewer", $("#reviewer").value));
  renderProgress();
  $$("tr[data-id]").forEach(paintRow);
  ["#f-decision", "#f-treatment", "#f-outcome"].forEach(sel => {
    const node = $(sel); if (node) node.addEventListener("change", applyFilters);
  });
  $("#btn-export") && ($("#btn-export").onclick = doExport);
  $("#import-file") && ($("#import-file").onchange = (event) => {
    if (event.target.files[0]) doImport(event.target.files[0]);
  });
  initDecide();
});
"""


# ---------------------------------------------------------------------------
# 逐轮来源标注（④）
# ---------------------------------------------------------------------------

def classify_turn(turn: dict, index: int) -> Tuple[str, str]:
    """
    一轮消息的**来源**——控制器提示不许伪装成人类输入。

    取值与含义：`task` 任务下发（控制器构造）、`model` 模型输出、
    `tool` 工具返回（投递包）、`control` 控制器提示（系统消息/状态小结/纠正）。
    """
    role = turn.get("from")
    value = (turn.get("value") or "").lstrip()
    if role == "gpt":
        return "model", "模型输出"
    if index == 0:
        return "task", "任务下发（控制器构造）"
    if value.startswith("{"):
        return "tool", "工具返回（投递包）"
    if value.startswith("[System:"):
        if "本会话状态" in value:
            return "control", "控制器提示（状态小结）"
        if "执行失败" in value or "不提供" in value or "已跳过执行" in value:
            return "control", "控制器提示（工具/执行）"
        return "control", "控制器提示"
    return "control", "控制器提示（纠正/追问）"


def call_status(record: dict, trace: dict) -> dict:
    """
    调用状态：**四类各自记录、各自可能"未记录"**，不靠"数组是空的"推断。

    这个页面曾经把 `unavailable_tool_calls` 与 `failed_tool_calls` 直接拼成
    "缺席 []｜失败 []"——空数组既可能是"确实没有"，也可能是"这个版本根本没记"，
    而"模型没调用某个专家"更是完全另一件事。四类：

      * `unavailable` 请求了不可用工具（tag + 对应专家 + 轮次 + 提示原文）
      * `failed`      工具执行失败（专家 + 异常原文 + 轮次）
      * `skipped`     重复请求被跳过（专家 + 轮次，计数来自 metadata）
      * `not_called`  本次可用但模型没调用（available 减去已测量/不可用/失败）

    每类都带 `recorded` 标志：键不存在就是**未记录**（旧 trace），不写"无"。
    """
    metadata = trace.get("metadata") or {}
    turns = trace.get("conversations") or []

    def _turn_texts(*needles) -> List[int]:
        found = []
        for index, turn in enumerate(turns):
            value = turn.get("value") or ""
            if all(needle in value for needle in needles):
                found.append(index)
        return found

    measured = {token.get("source") for token in (record.get("evidence") or [])
                if token.get("source")}
    available = metadata.get("available_experts")
    if available is None:
        available = record.get("tools_served")

    out: Dict[str, object] = {"unavailable": [], "failed": [], "skipped": [],
                              "not_called": [], "recorded": {}, "skipped_count": None}

    raw_unavailable = metadata.get("unavailable_tool_calls")
    out["recorded"]["unavailable"] = raw_unavailable is not None
    asked_unavailable = set()
    for tag in raw_unavailable or []:
        turn_indexes = _turn_texts("不提供", tag)
        notice = next((turns[index].get("value") for index in turn_indexes), "")
        source = ForensicStateMachine.EXPERT_KEY_MAP.get(tag, "")
        if source:
            asked_unavailable.add(source)
        out["unavailable"].append({
            "tool": tag, "expert": source or "（未登记标签）", "turns": turn_indexes,
            "reason": (notice.strip() if notice else "未记录提示原文"),
        })

    raw_failed = metadata.get("failed_tool_calls")
    out["recorded"]["failed"] = raw_failed is not None
    asked_failed = set()
    for entry in raw_failed or []:
        source = str(entry).split(":", 1)[0].strip()
        asked_failed.add(source)
        turn_indexes = _turn_texts("执行失败", source) or _turn_texts(source, "失败")
        out["failed"].append({"tool": source, "expert": source, "turns": turn_indexes,
                              "reason": str(entry)})

    skipped_count = metadata.get("skipped_duplicate_calls")
    out["skipped_count"] = skipped_count
    out["recorded"]["skipped"] = skipped_count is not None
    for source in sorted(set(metadata.get("skipped_sources") or []) |
                         {token.get("source") for token in (record.get("evidence") or [])
                          if token.get("source") and skipped_count}):
        turn_indexes = _turn_texts(source, "已跳过执行")
        if turn_indexes:
            out["skipped"].append({"tool": source, "expert": source,
                                   "turns": turn_indexes,
                                   "reason": "该专家已在本会话测量过整图，重复请求未执行"})

    out["recorded"]["not_called"] = available is not None
    if available is not None:
        out["not_called"] = sorted(set(available) - measured - asked_unavailable
                                   - asked_failed)
    return out


def _render_call_status(status: dict) -> str:
    """按调用状态出块：无异常就一句"调用异常：无"，有就逐条列出工具/轮次/原因。"""
    parts: List[str] = ["<h2>调用状态</h2>"]
    abnormal = status["unavailable"] or status["failed"]
    if not status["recorded"]["unavailable"] and not status["recorded"]["failed"]:
        parts.append('<p class="banner">调用异常：<b>未记录</b>'
                     '（本会话 trace 里没有 `unavailable_tool_calls` / '
                     '`failed_tool_calls` 字段，不能当作"没有异常"）。</p>')
    elif not abnormal:
        parts.append('<p>调用异常：<b>无</b>'
                     '<span class="muted">（未请求不可用工具，也没有工具执行失败）</span></p>')
    if status["unavailable"]:
        parts.append("<h3>请求了不可用工具</h3><table>"
                     "<tr><th>工具</th><th>对应专家</th><th>轮次</th><th>原因（记录原文）</th></tr>")
        for item in status["unavailable"]:
            parts.append(f'<tr><td>{html.escape(str(item["tool"]))}</td>'
                         f'<td>{html.escape(str(item["expert"]))}</td>'
                         f'<td>{html.escape(_turns_text(item["turns"]))}</td>'
                         f'<td>{html.escape(str(item["reason"]))}</td></tr>')
        parts.append("</table>")
    if status["failed"]:
        parts.append("<h3>工具执行失败</h3><table>"
                     "<tr><th>专家</th><th>轮次</th><th>记录的原因</th></tr>")
        for item in status["failed"]:
            parts.append(f'<tr><td>{html.escape(str(item["expert"]))}</td>'
                         f'<td>{html.escape(_turns_text(item["turns"]))}</td>'
                         f'<td>{html.escape(str(item["reason"]))}</td></tr>')
        parts.append("</table>")
    if status["recorded"]["skipped"]:
        count = status["skipped_count"]
        if count:
            detail = ("；".join(f'{item["tool"]}（轮次 {_turns_text(item["turns"])}）'
                                for item in status["skipped"])
                      or "轮次未记录")
            parts.append(f'<p>重复请求被跳过：<b>{count}</b> 次 —— {html.escape(detail)}'
                         '<span class="muted">（重复请求不执行、不产生新测量）</span></p>')
        else:
            parts.append('<p>重复请求被跳过：<b>无</b></p>')
    else:
        parts.append('<p>重复请求被跳过：<b>未记录</b></p>')
    if status["recorded"]["not_called"]:
        if status["not_called"]:
            parts.append('<p>未调用专家（本次可用、模型没有调用）：'
                         + "、".join(f"<code>{html.escape(name)}</code>"
                                     for name in status["not_called"])
                         + '<span class="muted">（可用列表取自 trace 的 '
                           '`available_experts`）</span></p>')
        else:
            parts.append('<p>未调用专家：<b>无</b>（可用的都调用过）</p>')
    else:
        parts.append('<p>未调用专家：<b>未记录</b>'
                     '（trace 里没有可用专家列表）</p>')
    return "".join(parts)


def _turns_text(indexes) -> str:
    if not indexes:
        return "未记录"
    return "、".join(f"#{index}" for index in indexes)


def _truncate(text: str, limit: int = 900) -> Tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _pre(text: str, fmt: str = "") -> str:
    return f'<pre class="{fmt}">{html.escape(text)}</pre>'


def _json_block(value: str, limit: int = 1500) -> str:
    """
    一段 JSON：美化后展示（可截断），**同时**给原始文本与完整展开入口。

    截断是给人的第一条视线，完整性靠折叠区——两者都要有，否则"摘要看起来对"
    会取代"原文确实对"。
    """
    stripped = (value or "").lstrip()
    pretty = value
    if stripped.startswith("{"):
        try:
            pretty = json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        except ValueError:
            pretty = value
    shown, cut = _truncate(pretty, limit)
    parts = [_pre(shown + ("\n…（已截断，完整内容见下面两个折叠）" if cut else ""))]
    if cut or "\n" in pretty:
        parts.append("<details><summary>完整内容（美化 JSON）</summary>"
                     + _pre(pretty) + "</details>")
    parts.append("<details><summary>原始文本（未做任何美化/改写）</summary>"
                 + _pre(value or "") + "</details>")
    return "".join(parts)


def session_page(sample: dict, page_dir: str, index_name: str,
                 batch_meta: Optional[dict] = None,
                 neighbors: Optional[Dict[str, str]] = None,
                 meta: Optional[dict] = None) -> str:
    record = sample["record"]
    trace = sample["trace"]
    metadata = trace.get("metadata") or {}
    trajectory_id = record.get("trajectory_id")
    verdict, truth = record.get("final_verdict"), record.get("ground_truth")
    agree = "ok" if verdict == truth else ("warn" if verdict == "Uncertain" else "bad")
    evidence_check = record.get("evidence_check") or {}
    counters = record.get("counters") or {}

    parts: List[str] = []
    # ① 顶部固定审核栏：目录/上下条/样本名/小标签/指纹/审核控件，**只此一套**。
    nav = []
    if neighbors and neighbors.get("prev"):
        nav.append(f'<a href="{html.escape(neighbors["prev"])}">← 上一条</a>')
    if neighbors and neighbors.get("next"):
        nav.append(f'<a href="{html.escape(neighbors["next"])}">下一条 →</a>')
    parts.append(
        '<header class="topbar">'
        f'<div class="row crumbs"><a href="{html.escape(index_name)}">← 批次目录</a>'
        + ("　" + " · ".join(nav) if nav else "")
        + f'<span class="muted">内容指纹 {html.escape(sample["fingerprint"])}</span></div>'
        f'<h1>{html.escape(str(trajectory_id))}</h1>'
        f'<div class="row">'
        f'<span class="pill {agree}">判定 {html.escape(str(verdict))}</span>'
        f'<span class="pill">真值 {html.escape(str(truth))}</span>'
        f'<span class="pill">结案来源 {html.escape(str(record.get("verdict_status")))}</span>'
        f'<span class="pill">停止原因 {html.escape(str(record.get("halting_reason")))}</span>'
        f'<span class="pill">{html.escape(str(record.get("treatment")))}</span>'
        "</div>"
        + _review_controls(trajectory_id)
        + "</header>")

    # ⑥ 审计信息：图、尺寸、版本、计数、检查发现
    variant = record.get("image_path") or ""
    image_size = metadata.get("image_size") or []
    parts.append("<h2>图像与版本</h2>")
    parts.append('<div class="grid">')
    parts.append(f'<div class="card"><h3>实际输入变体（模型看到的）</h3>'
                 f'<img src="{html.escape(_rel(variant, page_dir))}" alt="variant">'
                 f'<p class="muted">路径 {html.escape(variant)}<br>'
                 f'实际尺寸 {html.escape(str(image_size)) if image_size else "未记录"}'
                 f'（来源：trace metadata.image_size）</p></div>')
    source_image = sample.get("source_image") or ""
    parts.append(f'<div class="card"><h3>源图（变体由它编码而来）</h3>'
                 + (f'<img src="{html.escape(_rel(source_image, page_dir))}" alt="source">'
                    if source_image and os.path.exists(
                        os.path.join(PROJECT_ROOT, source_image))
                    else '<p class="muted">未记录</p>')
                 + f'<p class="muted">{html.escape(source_image or "未记录")}</p></div>')
    model = (batch_meta or {}).get("model") or {}
    experts = (batch_meta or {}).get("experts") or {}
    parts.append('<div class="card"><h3>模型与专家版本</h3><ul>'
                 f'<li>模型 {html.escape(str(model.get("checkpoint")))}@'
                 f'{html.escape(str(model.get("revision"))[:8])}（{html.escape(str(model.get("dtype")))}，'
                 f'{html.escape(str(model.get("inference_mode")))}）</li>'
                 + "".join(f'<li>{html.escape(name)}：{html.escape(str(entry.get("class")))}'
                           f'，设备 {html.escape(str(entry.get("device")))}</li>'
                           for name, entry in sorted(experts.items()))
                 + f'<li>提示词 digest {html.escape(str(((batch_meta or {}).get("prompts") or {}).get("digest")))}</li>'
                   '</ul></div>')
    parts.append("</div>")

    parts.append("<h2>控制与计数</h2><table><tr><th>项</th><th>值</th></tr>")
    rows = [
        ("模型自己的报告", record.get("report")),
        ("模型候选 / 采纳", f'{record.get("model_candidate")} / {record.get("final_verdict")}'),
        # 严格区分两种量：模型自报的、未校准的把握 与 专家的原始分数。
        # 不写成一个"置信度"了事，也不渲染成百分比——那会读成 P(真实/伪造)。
        ("模型自报置信度（未校准）",
         f'{record.get("confidence")}'
         "　—　这是模型对所选结论的主观把握，不是 P(Real/Fake)，"
         "也不是专家分数；专家原始分数见下方证据表"),
        ("停止原因（控制端）", f'{record.get("halting_reason")}｜policy_reasons='
                              f'{record.get("policy_reasons")}'),
        ("请求 / 执行 / 重复", f'执行 {counters.get("expert_calls")}｜跳过重复请求 '
                               f'{metadata.get("skipped_duplicate_calls", "未记录")}｜'
                               f'重复动作轮 {metadata.get("repeated_action_turns", "未记录")}｜'
                               f'抑制重复 {metadata.get("suppressed_duplicate_count", "未记录")}'),
        ("模型轮 / 工具 / 权重成本", f'{counters.get("model_turns")}｜{counters.get("expert_calls")}'
                                     f'｜{counters.get("weighted_cost")}'),
        ("格式/动作违规轮", f'格式失败 {metadata.get("format_failure_turns", "未记录")}｜'
                            f'非法动作 {metadata.get("invalid_action_turns", "未记录")}｜'
                            f'协议违规 {metadata.get("protocol_violations", "未记录")}'),
        ("耗时", f'{record.get("elapsed_s")} s'),
        ("记录 schema", record.get("record_schema")),
    ]
    for name, value in rows:
        parts.append(f"<tr><td>{html.escape(name)}</td>"
                     f"<td>{html.escape(str(value))}</td></tr>")
    parts.append("</table>")
    parts.append(_render_call_status(call_status(record, trace)))

    # 协议命中：自动候选，供人工复核（用户 2026-10-18：全部命中都要人看）
    from scripts.g8_protocol_audit import PATTERNS, audit_trace, final_answer

    hits = audit_trace(trace)
    hit_count = sum(len(items) for items in hits.values())
    parts.append("<h2>协议命中（自动候选）</h2>")
    parts.append('<p class="muted">自动命中只是 <b>候选</b>：判据是词表，未命中不等于没问题；'
                 "请以原文为准。" +
                 (f"本会话命中 <b>{hit_count}</b> 处。" if hit_count else "本会话无命中。") + "</p>")
    if hit_count:
        parts.append("<table><tr><th>判据</th><th>轮次</th><th>句子（原文）</th></tr>")
        for name, _desc, _rx in PATTERNS:
            for item in hits[name]:
                parts.append(f'<tr><td>{html.escape(name)}</td>'
                             f'<td>#{item["turn"]}</td>'
                             f'<td>{html.escape(item["clause"])}</td></tr>')
        parts.append("</table>")
    answer = final_answer(trace)
    states_scope = bool(answer) and any(pattern.search(answer)
                                        for name, _d, pattern in PATTERNS if name == "scope")
    parts.append('<p>结案是否声明证据范围（自动判断）：<b>'
                 + ("是" if states_scope else "否")
                 + '</b><span class="muted">（自动判断只覆盖已定义句式，需人工确认）</span></p>')

    parts.append("<h2>引用检查（具体发现，不只是数量）</h2>")
    findings = evidence_check.get("findings") or []
    candidates = evidence_check.get("candidates") or []
    parts.append(f'<p>状态 <b>{html.escape(str(evidence_check.get("status", "未记录")))}</b>'
                 f'｜缺陷 {len(findings)}｜候选 {len(candidates)}</p>')
    if not findings and not candidates:
        parts.append('<p class="muted">本条没有缺陷、也没有候选。'
                     '（检查只覆盖已定义的模式——未命中不等于已证明忠实。）</p>')
    for label, items, severity in (("缺陷", findings, "bad"), ("候选", candidates, "warn")):
        for finding in items:
            parts.append(
                f'<div class="card" style="margin:6px 0">'
                f'<span class="pill {severity}">{label}</span>'
                f'<span class="pill">{html.escape(str(finding.get("kind")))}</span>'
                f'<div>{html.escape(str(finding.get("why") or ""))}</div>'
                + (f'<div class="muted">句子：{html.escape(str(finding.get("sentence")))}</div>'
                   if finding.get("sentence") else "")
                + "</div>")

    # ⑤ 投递包（trace 原文）＋ 后台证据（折叠）
    parts.append("<h2>证据：当时投递了什么，后台又记了什么</h2>")
    parts.append('<p class="muted">主区是 <b>trace 里保存的投递原文</b>（当时真正发出去的'
                 '那段 JSON），不是今天用代码重新生成的。后台完整 token 在其下方的折叠区。</p>')
    chain = record.get("evidence") or []
    delivered = [turn for index, turn in enumerate(trace.get("conversations") or [])
                 if classify_turn(turn, index)[0] == "tool"]
    # 一眼看清"哪些专家给了什么数"：整张表都从**投递原文**里读，不重算。
    measured = [_pack_row(turn.get("value") or "") for turn in delivered]
    measured = [row for row in measured if row]
    if measured:
        parts.append('<h3>本次得到的专家原始分数（未校准；非百分比）</h3>'
                     '<table><tr><th>expert</th><th>状态</th><th>测量范围</th>'
                     '<th>专家原始分数</th><th>该值的方向</th></tr>')
        for row in measured:
            parts.append(
                f'<tr><td>{html.escape(str(row.get("expert")))}</td>'
                f'<td>{html.escape(str(row.get("status")))}</td>'
                f'<td>{html.escape(str(row.get("scope")))}</td>'
                f'<td>{html.escape(str(row.get("value")))}</td>'
                f'<td>{html.escape(str(row.get("direction")))}</td></tr>')
        parts.append("</table>")
    if delivered:
        for index, turn in enumerate(delivered, start=1):
            parts.append(f"<h3>投递 {index}</h3>")
            parts.append(_json_block(turn.get("value") or ""))
    else:
        parts.append('<p class="muted">未记录投递轮（injection=none 或该会话没有投递证据）。</p>')
    if chain:
        parts.append("<details><summary>后台完整证据链（token 全字段，模型看不到这里）"
                     "</summary>")
        for token in chain:
            parts.append(_pre(json.dumps(token, ensure_ascii=False, indent=2)))
        parts.append("</details>")

    # ⑦ 日志完整性（本会话视角）
    parts.append("<h2>日志完整性</h2>")
    parts.append(_completeness_table(metadata, trace))

    # ④ 已记录的会话原文
    parts.append("<h2>已记录的会话原文</h2>")
    parts.append('<p class="muted">下方每一轮都按 trace 原样展示，助手文本不总结、不补写；'
                 '每轮附原始文本与来源标注。<a href="'
                 + html.escape(_rel(record.get("trace_path") or "", page_dir))
                 + '" download>下载整份 trace（JSON）</a></p>')
    for index, turn in enumerate(trace.get("conversations") or []):
        kind, label = classify_turn(turn, index)
        value = turn.get("value") or ""
        images = turn.get("image_paths") or []
        parts.append(f'<div class="turn {kind}">'
                     f'<div class="role">#{index} {html.escape(label)}</div>')
        if images:
            parts.append('<p class="muted">随附图像：'
                         + "、".join(html.escape(path) for path in images) + "</p>")
        parts.append(_json_block(value) if value.lstrip().startswith("{") else _pre(value))
        parts.append("</div>")

    return _page(str(trajectory_id), "".join(parts), meta=meta)


def _completeness_table(metadata: dict, trace: dict) -> str:
    """⑦：逐项标"已记录/未记录"——旧批次缺什么就写什么，不事后重建。"""
    calls = metadata.get("model_calls")
    prompt = metadata.get("system_prompt") or {}
    items = [
        ("初始 system prompt 原文", bool(prompt.get("text")),
         "只有 digest 时不能证明当时用的是哪一版"),
        ("每次模型调用的提示原文", bool(calls) and all(c.get("prompt_text") for c in calls),
         "把每轮真正发出去的提示文本存进 trace"),
        ("每次调用的多模态统计", bool(calls) and all(c.get("input_stats") for c in calls),
         "输入 token / 图像 token / 随附图数"),
        ("客户端内部重试", calls is not None and bool(calls)
         and all("attempt" in c for c in calls),
         "被格式校验拒掉的输出与随后注入的纠正"),
        ("纠正前/后的原始输出", bool((trace.get("final_verdict") or {}).get("evidence_check"))
         or any("original_closing" in json.dumps(t) for t in trace.get("conversations") or []),
         "控制器纠正会留 original_closing；客户端内部重试的留痕见上一行"),
        ("逐轮对话原文", bool(trace.get("conversations")), "本页展示的就是它"),
        ("证据链（后台 token）", bool(trace.get("evidence_chain")), ""),
        ("图像实际尺寸", bool(metadata.get("image_size")), ""),
    ]
    rows = ["<table><tr><th>记录项</th><th>状态</th><th>说明</th></tr>"]
    for name, present, note in items:
        rows.append(f'<tr><td>{html.escape(name)}</td>'
                    f'<td>{"已记录" if present else "未记录"}</td>'
                    f'<td class="muted">{html.escape(note)}</td></tr>')
    rows.append("</table>")
    if not prompt.get("text"):
        rows.append('<p class="banner">本会话的 trace 里<b>没有</b>初始 system prompt 原文、'
                    '也没有每次调用的提示文本与客户端重试记录（阶段 4.6 之前的会话）。'
                    '本页如实标"未记录"，<b>不用今天的代码重建冒充原始日志</b>；'
                    '后续批次会带上这些字段。</p>')
    return "".join(rows)


def _pack_row(payload: str) -> dict:
    """
    从**投递原文**里取一行"专家原始分数"。取不到就返回空——不拿后台 token 顶替，
    那会把模型没看到的字段当成它当时看到的东西。
    """
    stripped = (payload or "").lstrip()
    if not stripped.startswith("{"):
        return {}
    try:
        parsed = json.loads(payload)
    except ValueError:
        return {}
    if not isinstance(parsed, dict) or "expert" not in parsed:
        return {}
    score = parsed.get("score") or {}
    return {"expert": parsed.get("expert"), "status": parsed.get("status"),
            "scope": parsed.get("scope"), "value": score.get("value"),
            "direction": score.get("direction_for_this_value")
            or score.get("direction")}


def _review_controls(trajectory_id: str) -> str:
    """
    审核控件（顶部栏里那**一套**）：决定、审核人、分类、备注、导出、导入、提示。

    决定区在顶部而不是页尾：审核时要边看正文边改决定，控件跟着滚就没法用。
    """
    categories = "".join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
                         for name in CATEGORIES)
    return f'''
<div id="decide" data-id="{html.escape(trajectory_id)}">
  <div class="controls">
    <span id="decide-state"></span>
    <label class="muted">审核人 <input type="text" id="reviewer" placeholder="你的名字" size="10"></label>
    <label class="muted">分类 <select id="category"><option value="">（不需要）</option>{categories}</select></label>
    <button id="btn-pass" class="primary">通过（本次审核）</button>
    <button id="btn-revise" class="warn">待修</button>
    <button id="btn-reject" class="danger">拒绝</button>
    <button id="btn-export">导出决定</button>
    <label class="muted">导入 <input type="file" id="import-file" accept="application/json"></label>
    <span id="flash" class="muted" hidden></span>
  </div>
  <textarea id="note" placeholder="备注／原因（待修、拒绝必填）"></textarea>
  <p class="muted" style="margin:4px 0">这是<b>行为验收</b>：<b>通过</b>只表示本次审核通过，
  <b>不会把它变成可训练样本</b>，也不走训练准入。决定先存浏览器草稿，
  <b>导出后用 <code>python scripts/g8_review_import.py &lt;文件&gt;</code> 才写回仓库</b>；
  内容一变旧决定自动标"已过期"。链接颜色只表示你打开过（浏览记录），与审核状态无关。</p>
</div>
'''


def index_page(samples: List[dict], generation: str, batch: str,
               batch_meta: Optional[dict] = None,
               repo_decisions: Optional[Dict[str, dict]] = None,
               meta: Optional[dict] = None) -> str:
    treatments = sorted({sample["record"].get("treatment") for sample in samples})
    rows = []
    for sample in sorted(samples, key=lambda item: item["record"]["trajectory_id"]):
        record = sample["record"]
        verdict, truth = record.get("final_verdict"), record.get("ground_truth")
        outcome = "ok" if verdict == truth else ("abstain" if verdict == "Uncertain" else "wrong")
        rows.append(
            f'<tr data-id="{html.escape(record["trajectory_id"])}" '
            f'data-treatment="{html.escape(str(record.get("treatment")))}" '
            f'data-outcome="{outcome}">'
            f'<td><a href="{html.escape(sample["page"])}">{html.escape(str(record.get("variant_id")))}</a></td>'
            f'<td>{html.escape(str(record.get("treatment")))}</td>'
            f'<td>{html.escape(str(truth))}</td>'
            f'<td><span class="pill {"ok" if outcome == "ok" else ("warn" if outcome == "abstain" else "bad")}">'
            f'{html.escape(str(verdict))}</span></td>'
            f'<td>{html.escape(str(record.get("halting_reason")))}</td>'
            f'<td>{(record.get("counters") or {}).get("expert_calls")}</td>'
            f'<td data-state></td></tr>')

    model = (batch_meta or {}).get("model") or {}
    missing = _missing_items(samples)
    head = [
        f"<h1>批次 {html.escape(batch)}（代次 {html.escape(generation)}）· 验收审核工作台</h1>",
        '<p class="muted">生成视图：'
        + html.escape(datetime.now().isoformat(timespec="seconds"))
        + f"｜会话 {len(samples)} 条｜模型 "
        + html.escape(str(model.get("checkpoint"))) + "@"
        + html.escape(str(model.get("revision"))[:8])
        + f"｜策略 {html.escape(str((batch_meta or {}).get('policy')))}"
        + f"（{html.escape(str((batch_meta or {}).get('policy_version')))}）</p>",
        '<div class="banner">本页是<b>行为验收</b>：决定与训练无关（不改"可训练"标记，'
        '也不跑训练准入）。决定先存浏览器草稿，<b>导出后由导入命令写回仓库</b>才算数。</div>',
        '<h2>进度</h2>',
        '<div class="bar" id="progress"><span style="width:0%"></span></div>',
        '<p id="progress-text" class="muted"></p>',
        '<h2>筛选</h2><div class="filters">'
        '<label class="muted">决定 <select id="f-decision"><option value="">全部</option>'
        f'{"".join(f"<option value={d}>{d}</option>" for d in DECISIONS)}'
        '<option value="undecided">未决定</option><option value="stale">已过期</option></select></label>'
        '<label class="muted">格式 <select id="f-treatment"><option value="">全部</option>'
        + "".join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
                  for name in treatments) +
        '</select></label>'
        '<label class="muted">结果 <select id="f-outcome"><option value="">全部</option>'
        '<option value="ok">判定=真值</option><option value="abstain">弃权</option>'
        '<option value="wrong">判错</option></select></label>'
        '<label class="muted">审核人 <input type="text" id="reviewer" placeholder="你的名字"></label>'
        '<button id="btn-export">导出决定</button>'
        '<label class="muted">导入：<input type="file" id="import-file" accept="application/json"></label>'
        '<span id="flash" class="muted" hidden></span></div>',
    ]
    if missing:
        head.append('<p class="banner">本批缺失的记录项（各会话页也已逐项标注）：'
                    + html.escape("；".join(missing)) + "</p>")
    head.append('<h2>会话一览</h2><table><tr><th>变体</th><th>格式</th><th>真值</th>'
                '<th>判定</th><th>停止原因</th><th>工具</th><th>审核</th></tr>')
    head.append("".join(rows) + "</table>")

    return _page(f"{batch} 验收审核", "".join(head), meta=meta)


def _missing_items(samples: List[dict]) -> List[str]:
    """批次级"未记录"清单（⑦ 的另一半：批次视角）。"""
    if not samples:
        return []
    trace = samples[0]["trace"]
    metadata = trace.get("metadata") or {}
    missing = []
    if not (metadata.get("system_prompt") or {}).get("text"):
        missing.append("初始 system prompt 原文")
    if not metadata.get("model_calls"):
        missing.append("每次模型调用的提示原文与统计")
    if not metadata.get("image_size"):
        missing.append("图像实际尺寸")
    return missing


def _page(title: str, body: str, meta: Optional[dict] = None) -> str:
    script = ""
    if meta is not None:
        script = ("<script>" + JS.replace("__META__", json.dumps(meta, ensure_ascii=False))
                  + "</script>")
    return ("<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"{body}{script}</body></html>")


def _rel(target: str, page_dir: str) -> str:
    if not target:
        return ""
    absolute = target if os.path.isabs(target) else os.path.join(PROJECT_ROOT, target)
    return os.path.relpath(absolute, page_dir).replace(os.sep, "/")


def _source_loader():
    """
    源图路径：`split_v2.json` 的 `sources[source_id].path`。

    变体清单只记"这个变体属于哪个源"（`source_id`），源文件本身在 split 里；
    两张图都要摆出来才看得见"模型看到的那张是不是源的忠实编码"。
    """
    split_path = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
    try:
        with open(split_path, encoding="utf-8") as handle:
            split = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {key: entry.get("path", "") for key, entry in (split.get("sources") or {}).items()}


def build(generation: str, batch: str, output_dir: str,
          records_dir: Optional[str] = None,
          variants_manifest: Optional[str] = None,
          ledger_path: Optional[str] = None) -> dict:
    paths = batch_paths(batch, generation)
    records_dir = records_dir or paths["trajectories"]
    os.makedirs(output_dir, exist_ok=True)

    batch_meta = {}
    metadata_path = os.path.join(records_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            batch_meta = json.load(handle)

    manifest: Dict[str, dict] = {}
    manifest_path = variants_manifest or paths["variants"]
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        manifest = {entry.get("variant_id"): {**entry, "source_path": key}
                    for key, entry in (raw.get("variants") or {}).items()}

    _SOURCES = _source_loader()
    repo = latest(load_ledger(generation, batch, path=ledger_path))
    repo_decisions = {key: {"fingerprint": value.fingerprint, "decision": value.decision,
                            "reviewer": value.reviewer, "category": value.category,
                            "note": value.note, "decided_at": value.decided_at}
                      for key, value in repo.items()}
    # 页面里的 JS 需要：每条样本的指纹（由 Python 算好，JS 只比对，不重算）
    # ＋ 仓库台账现状（页面显示"仓库 vs 草稿"两种状态）。

    names = sorted(name for name in os.listdir(records_dir)
                   if name.endswith(".json") and name != "metadata.json")
    samples, missing = [], []
    for name in names:
        with open(os.path.join(records_dir, name), encoding="utf-8") as handle:
            record = json.load(handle)
        trace_path = record.get("trace_path")
        if not trace_path or not os.path.exists(os.path.join(PROJECT_ROOT, trace_path)):
            missing.append(name)
            continue
        with open(os.path.join(PROJECT_ROOT, trace_path), encoding="utf-8") as handle:
            trace = json.load(handle)
        samples.append({
            "record": record, "trace": trace, "page": f"{os.path.splitext(name)[0]}.html",
            "fingerprint": sample_fingerprint(record, trace),
            "source_image": _SOURCES.get(record.get("source_id") or "", ""),
        })

    meta = {
        "schema": SCHEMA, "generation": generation, "batch": batch,
        "samples": {sample["record"]["trajectory_id"]: {
            "fingerprint": sample["fingerprint"], "page": sample["page"]}
            for sample in samples},
        "repo_decisions": repo_decisions,
    }
    for position, sample in enumerate(samples):
        neighbors = {
            "prev": samples[position - 1]["page"] if position > 0 else "",
            "next": samples[position + 1]["page"] if position + 1 < len(samples) else "",
        }
        with open(os.path.join(output_dir, sample["page"]), "w", encoding="utf-8") as handle:
            handle.write(session_page(sample, output_dir, "index.html",
                                      batch_meta=batch_meta, neighbors=neighbors,
                                      meta=meta))
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(index_page(samples, generation, batch, batch_meta, repo_decisions,
                                meta=meta))

    # 过期**每次重算**，不信台账里写的：内容有没有变，只有现在这一刻知道。
    # （`check_import` 在导入时标一次，重建页面时再标一次——两次都必须以当前指纹为准。）
    current = {sample["record"]["trajectory_id"]: sample["fingerprint"]
               for sample in samples}
    repo_values = []
    for key, value in repo.items():
        value.stale = value.fingerprint != current.get(key, value.fingerprint)
        if value.stale:
            value.stale_reason = (value.stale_reason
                                  or "内容已变（重建页面时重算，需重审）")
        repo_values.append(value)
    state = progress(repo_values, total=len(samples))
    return {"sessions": len(samples), "missing_trace": missing, "progress": state,
            "index": os.path.relpath(os.path.join(output_dir, "index.html"), PROJECT_ROOT)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--records-dir", default=None)
    parser.add_argument("--variants-manifest", default=None)
    parser.add_argument("--ledger", default=None,
                        help="验收台账路径（默认 sft_data/<gen>/review/<batch>/dispositions.jsonl）")
    args = parser.parse_args()
    output_dir = args.output_dir or batch_paths(args.batch, args.generation)["review"]
    report = build(args.generation, args.batch, output_dir, records_dir=args.records_dir,
                   variants_manifest=args.variants_manifest, ledger_path=args.ledger)
    print(f"会话页 {report['sessions']} 个；索引 {report['index']}")
    print(f"仓库台账进度：{json.dumps(report['progress'], ensure_ascii=False)}")
    if report["missing_trace"]:
        print(f"缺 trace 的记录（未出页）：{report['missing_trace']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

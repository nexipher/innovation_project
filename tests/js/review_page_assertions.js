// 页面上的审核人输入框（替身）：先填上，后面单独测"没填"的情况
const reviewerInput = document.querySelector("#reviewer");
reviewerInput.value = "hj";

// ① 进度保存：写进 localStorage 的草稿能读回来（刷新后仍在）
const id = Object.keys(META.samples)[0];
const draft = loadDraft();
draft[id] = { fingerprint: META.samples[id].fingerprint, decision: "pass",
              reviewer: "hj", category: null, note: "", decided_at: "2026-10-15T22:00:00" };
saveDraft(draft);
assert(loadDraft()[id].decision === "pass", "草稿写入后能读回（刷新不丢）");

// ② 指纹一致 → 决定有效；指纹变了 → 标过期、不继承
assert(stateOf(id).decision === "pass" && stateOf(id).stale === false,
       "指纹一致时决定有效");
const _real = META.samples[id];
META.samples[id] = { fingerprint: "deadbeef", page: _real.page };
assert(stateOf(id).stale === true, "内容指纹变了就标过期");
assert(stateOf(id).decision === "pass", "过期不等于丢弃：决定仍在，只是要重审");
META.samples[id] = _real;

// ③ 导出：带指纹、带用途声明、不带任何训练标记
const payload = buildExport();
assert(payload.schema === META.schema, "导出带 schema");
assert(payload.generation === META.generation && payload.batch === META.batch,
       "导出带代次与批次");
assert(payload.scope === "acceptance_only", "导出声明为验收用途");
assert(JSON.stringify(payload).indexOf("trainable") === -1, "导出里没有训练标记");
assert(payload.decisions[id].fingerprint === META.samples[id].fingerprint, "导出带内容指纹");

// ④ 跨批次导入必须拒绝（整份拒收，草稿不受影响）
const foreign = { schema: META.schema, generation: META.generation, batch: "b9",
                  decisions: { "别的样本": { fingerprint: "x", decision: "pass" } } };
doImport({ _text: JSON.stringify(foreign) });
assert(flashText().indexOf("导入被拒") >= 0, "跨批导入被拒（页面明说）");
assert(loadDraft()["别的样本"] === undefined, "跨批导入没有落进草稿");

// ④b 本批、但样本不存在 → 该条被拒，不整体接受
const partly = { schema: META.schema, generation: META.generation, batch: META.batch,
                 decisions: { "不存在的样本": { fingerprint: "x", decision: "pass" } } };
doImport({ _text: JSON.stringify(partly) });
assert(loadDraft()["不存在的样本"] === undefined, "未知样本不落草稿");
assert(flashText().indexOf("拒绝 1 条") >= 0, "页面报出被拒条数");

// ⑤ 待修/拒绝缺原因 → 不落草稿，并给出提示
persist(id, "revise", "", "");
assert(loadDraft()[id].decision === "pass", "待修没写原因不覆盖原决定");
assert(flashText().indexOf("必须填写原因") >= 0, "缺原因时页面说要填原因");

// ⑤b 没有审核人 → 不落草稿（把审核人清空再试）
reviewerInput.value = "";
persist(id, "reject", "protocol", "两个动作标签");
assert(loadDraft()[id].decision === "pass", "没有审核人不落草稿");
assert(flashText().indexOf("审核人") >= 0, "缺审核人时页面说要填");

finish();

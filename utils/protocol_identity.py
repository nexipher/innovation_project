"""
验收协议身份（2026-10-18 用户指定，plan §4.12b）。

**为什么批次冻结不等于协议冻结**：`FROZEN.json` 里的 `(g8, b2)` 只说明那 32 条
会话是历史材料，不说明外部 Agent 即将使用的**运行配置**与 b2 验收时相同。主模型
revision、系统提示词、专家权重与预处理、模型可见证据包、预算与调用规则，任何一项
变了，b2 的人工验收就不再覆盖新的运行。

这里把"协议"写成一份可版本化的身份：

  * **模型**：checkpoint / revision / dtype / inference_mode / 解码参数；
  * **提示词**：`prompt_digest` + 实际系统提示词 sha256；
  * **专家**：集合、每个专家的类、权重 sha256/字节数、登记表中的预处理；
  * **证据包**：`EvidenceTokenizer.MODEL_VISIBLE_KEYS` 的版本（对键集合取哈希）；
  * **工具与预算**：policy、tools_served、HALTING_POLICY、调用/轮次上限、
    证据注入方式；
  * **协议文件内容哈希**：controller / evidence_tokenizer / message_builder /
    qwen3_client / config / score_semantics（py + json）/ external_base /
    每个专家适配器。

两侧都用**同一个** `canonical_identity()` 归一化：登记时从 b2 自己的
`metadata.json` 取（并核对协议文件自 b2 提交后未变），运行时从**真实构建出来的
管线对象**取（不读 CLI 声明、不读 b2 元数据）。比较是逐字段的，错误信息给出
"验收值 / 当前值"。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

from config import PROJECT_ROOT

#: 验收身份文件的 schema。字段或比较规则变化时递增。
SCHEMA = "acceptance_protocol_identity/1"

#: 协议定义文件：内容哈希进入身份。改任何一个都应触发重新验收。
PROTOCOL_FILES = (
    "state_machine/controller.py",
    "state_machine/evidence_tokenizer.py",
    "mllm/message_builder.py",
    "mllm/qwen3_client.py",
    "config.py",
    "utils/score_semantics.py",
    "experts/score_semantics.json",
    "experts/external_base.py",
)

#: 专家类 → 实现文件。未知类不能登记（无法把它的代码绑进身份）。
EXPERT_FILES = {
    "ProbeDinoExpert": "experts/probe_dino.py",
    "NPRExpert": "experts/npr.py",
    "UnivFDExpert": "experts/univfd.py",
    "NoiseExpert": "experts/noise.py",
    "JPEGExpert": "experts/jpeg.py",
    "FrequencyExpertV2": "experts/frequency_v2.py",
}

#: 身份必须包含的小节；缺一节就不是可比较的身份。
REQUIRED_SECTIONS = ("model", "prompts", "experts", "evidence_pack", "tools",
                     "budget", "decoding", "provenance")


# ---------------------------------------------------------------------------
# 基本工具
# ---------------------------------------------------------------------------

def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha256(identity: dict) -> str:
    """身份内容的稳定哈希：写进日志/批次元数据，便于引用而不必再序列化整份。"""
    return sha256_text(_canonical(identity))[:16]


# ---------------------------------------------------------------------------
# 身份采集
# ---------------------------------------------------------------------------

def pack_identity() -> dict:
    """
    模型可见证据包的版本。

    版本来自 `MODEL_VISIBLE_KEYS` 的**内容**，不是手写常量：键集合变了，版本自然
    变，不依赖谁记得改字符串。`evidence_tokenizer.py` 的文件哈希也在协议文件里，
    读取文本/语义的变化同样会被挡住。
    """
    from state_machine.evidence_tokenizer import EvidenceTokenizer

    keys = list(EvidenceTokenizer.MODEL_VISIBLE_KEYS)
    return {"version": f"slim/{sha256_text(_canonical(keys))[:12]}",
            "keys": keys}


def model_identity(client) -> dict:
    """主模型身份：client 自己的 identity()，没有就是 mock/未记录。"""
    if client is None or not callable(getattr(client, "identity", None)):
        return {"name": getattr(client, "name", None), "mode": "mock"}
    return client.identity()


def prompt_identity() -> dict:
    """系统提示词身份：合并 digest + 实际使用的那份的 sha256。"""
    from mllm.message_builder import FORENSIC_SYSTEM_PROMPT
    from utils.config_fingerprint import prompt_digest

    return {
        "digest": prompt_digest(),
        "forensic_sha256": sha256_text(FORENSIC_SYSTEM_PROMPT),
        "forensic_chars": len(FORENSIC_SYSTEM_PROMPT),
    }


def budget_identity(evidence_injection: str = "text+image") -> dict:
    """停止策略与安全预算：换一个值就是另一种协议。"""
    import config

    return {
        "halting_policy": getattr(config, "HALTING_POLICY", None),
        "max_expert_calls": getattr(config, "MAX_EXPERT_CALLS", None),
        "max_model_turns": getattr(config, "MAX_MODEL_TURNS", None),
        "evidence_injection": evidence_injection,
    }


def registry_preprocess(source_name: str) -> Optional[dict]:
    """登记表里的预处理身份；未登记的专家没有可核对的预处理。"""
    try:
        from utils.score_semantics import SemanticsError, entry

        try:
            record = entry(source_name)
        except SemanticsError:
            return None
    except Exception:
        return None
    preprocess = record.get("preprocess") or {}
    if not preprocess:
        return None
    return {
        "name": preprocess.get("name"),
        "detail": preprocess.get("detail"),
        "measurement_scope": preprocess.get("measurement_scope"),
        "sha256": sha256_text(_canonical(preprocess)),
    }


def expert_identity(experts: Dict[str, object]) -> Dict[str, dict]:
    """
    每个专家的身份：项目已有的身份函数 + 设备 + 权重出处 + 预处理。

    与 `scripts/g8_generate_batch.py` 原来写的批次元数据保持同一形状：登记时
    读的是 b2 的 `metadata.json`，运行时读的是这批对象，两者必须能逐字段对上。
    """
    from utils.config_fingerprint import expert_specs

    specs = expert_specs(experts or {})
    for name, expert in (experts or {}).items():
        specs.setdefault(name, {})
        specs[name]["device"] = getattr(expert, "device", None)
        for attr in ("weight_source", "licence_note", "score_reference"):
            value = getattr(expert, attr, None)
            if value:
                specs[name][attr] = value
        source_name = specs[name].get("source_name") or name
        preprocess = registry_preprocess(source_name)
        if preprocess:
            specs[name]["preprocess"] = preprocess
        specs[name].update(_weights_digest(expert))
    return specs


def _weights_digest(expert) -> dict:
    """专家权重摘要：优先专家自己的 identity()，否则对真实权重文件取 sha256。"""
    info: Dict[str, object] = {}
    if callable(getattr(expert, "identity", None)):
        try:
            info["identity"] = expert.identity()
        except Exception as exc:                     # 身份拿不到不该中断采集
            info["identity_error"] = f"{type(exc).__name__}: {exc}"
    path = getattr(expert, "weights_path", None)
    # 只对普通文件取摘要：mock 的 weights_path 是 /dev/null，既没有意义还会写 sidecar。
    if isinstance(path, str) and os.path.isfile(path):
        from experts.external_base import sha256_file

        try:
            info["weights_path"] = path
            info["weights_sha256"] = sha256_file(path)
            info["weights_bytes"] = os.path.getsize(path)
        except Exception as exc:
            info["weights_error"] = f"{type(exc).__name__}: {exc}"
    return info


def protocol_file_paths(expert_classes: Iterable[str] = ()) -> List[str]:
    """身份覆盖的全部文件（排序、去重）。未知专家类直接拒绝。"""
    paths = list(PROTOCOL_FILES)
    for name in sorted(set(expert_classes)):
        target = EXPERT_FILES.get(name)
        if target is None:
            raise SystemExit(
                f"无法把专家类 {name!r} 绑定进协议身份：它没有登记实现文件。"
                f"已知类：{', '.join(sorted(EXPERT_FILES))}。"
                "新增专家时先在 utils/protocol_identity.py 的 EXPERT_FILES 里登记。")
        paths.append(target)
    return sorted(set(paths))


def provenance_identity(expert_classes: Iterable[str] = ()) -> dict:
    """协议文件内容哈希。读不到文件就拒绝——不能给一份不完整的身份。"""
    files: Dict[str, str] = {}
    for relative in protocol_file_paths(expert_classes):
        absolute = os.path.join(PROJECT_ROOT, relative)
        if not os.path.isfile(absolute):
            raise SystemExit(f"协议文件不存在，无法生成身份：{relative}")
        files[relative] = file_sha256(absolute)
    return {"files": files, "sha256": sha256_text(_canonical(files))}


# ---------------------------------------------------------------------------
# 归一化：两侧共用
# ---------------------------------------------------------------------------

def _normalize_model(identity: dict) -> dict:
    identity = identity or {}
    generation = identity.get("generation") or {}
    return {
        "name": identity.get("name"),
        "checkpoint": identity.get("checkpoint") or identity.get("repo_id"),
        "revision": identity.get("revision"),
        "dtype": identity.get("dtype"),
        "inference_mode": identity.get("inference_mode"),
        # 只比较会影响输出的解码参数；snapshot 路径/分片列表是机器信息，不进身份。
        "generation": {
            "temperature": generation.get("temperature"),
            "do_sample": generation.get("do_sample"),
            "max_new_tokens": generation.get("max_new_tokens"),
        } if generation else None,
    }


def _normalize_experts(experts: Dict[str, dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, spec in sorted((experts or {}).items()):
        source_name = spec.get("source_name") or name
        preprocess = spec.get("preprocess") or registry_preprocess(source_name)
        out[name] = {
            "class": spec.get("class"),
            "source_name": source_name,
            "weights_sha256": spec.get("weights_sha256"),
            "weights_bytes": spec.get("weights_bytes"),
            "preprocess_sha256": (preprocess or {}).get("sha256"),
            "preprocess_detail": (preprocess or {}).get("detail"),
        }
    return out


def canonical_identity(*, model: dict, prompts: dict, experts: Dict[str, dict],
                       tools: dict, budget: dict, decoding: dict,
                       provenance: dict) -> dict:
    """把任意来源的身份归一成同一个可比较结构。"""
    return {
        "schema": SCHEMA,
        "model": _normalize_model(model),
        "prompts": {
            "digest": (prompts or {}).get("digest"),
            "forensic_sha256": (prompts or {}).get("forensic_sha256"),
        },
        "experts": _normalize_experts(experts),
        "evidence_pack": pack_identity(),
        "tools": {
            "policy": (tools or {}).get("policy"),
            "policy_version": (tools or {}).get("policy_version"),
            "tools_served": sorted((tools or {}).get("tools_served") or []),
            "evidence_injection": (tools or {}).get("evidence_injection"),
        },
        "budget": {
            "halting_policy": (budget or {}).get("halting_policy"),
            "max_expert_calls": (budget or {}).get("max_expert_calls"),
            "max_model_turns": (budget or {}).get("max_model_turns"),
            "evidence_injection": (budget or {}).get("evidence_injection"),
        },
        "decoding": dict(decoding or {}),
        "provenance": provenance,
    }


def identity_from_metadata(metadata: dict) -> dict:
    """从 b2 的批次元数据建验收身份（登记时用）。"""
    experts = metadata.get("experts") or {}
    classes = {spec.get("class") for spec in experts.values() if spec.get("class")}
    return canonical_identity(
        model=metadata.get("model") or {},
        prompts=metadata.get("prompts") or {},
        experts=experts,
        tools={
            "policy": metadata.get("policy"),
            "policy_version": metadata.get("policy_version"),
            "tools_served": metadata.get("tools_served") or [],
            "evidence_injection": metadata.get("evidence_injection"),
        },
        budget=metadata.get("budget") or {},
        decoding=metadata.get("decoding") or {},
        provenance=provenance_identity(classes),
    )


def runtime_identity(base, *, tools_policy: str,
                     greedy: bool, inference_mode: str,
                     evidence_injection: str = "text+image") -> dict:
    """
    即将运行的管线对象的**实际**身份。

    `base` 是 `main.build_pipeline()` 返回的对象：模型客户端与专家都已真实构造，
    这里读的是对象，不是命令行声明。
    """
    client = getattr(base, "_mllm", None)
    experts = getattr(base, "_experts", {}) or {}
    classes = {type(expert).__name__ for expert in experts.values()}
    return canonical_identity(
        model=model_identity(client),
        prompts=prompt_identity(),
        experts=expert_identity(experts),
        tools={
            "policy": tools_policy,
            "policy_version": budget_identity()["halting_policy"],
            "tools_served": sorted(experts),
            "evidence_injection": evidence_injection,
        },
        budget=budget_identity(evidence_injection),
        decoding={"greedy": bool(greedy), "inference_mode": inference_mode},
        provenance=provenance_identity(classes),
    )


# ---------------------------------------------------------------------------
# 比较
# ---------------------------------------------------------------------------

def _diff(expected, actual, path: str, problems: List[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected:
                problems.append(f"{child}: 验收未记录，当前为 {actual[key]!r}")
            elif key not in actual:
                problems.append(f"{child}: 验收为 {expected[key]!r}，当前缺失")
            else:
                _diff(expected[key], actual[key], child, problems)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if expected != actual:
            problems.append(f"{path}: 验收 {expected!r}，当前 {actual!r}")
        return
    if expected != actual:
        problems.append(f"{path}: 验收 {expected!r}，当前 {actual!r}")


def compatibility_problems(accepted: dict, actual: dict) -> List[str]:
    """逐字段比较两份归一化身份；返回人类可读的差异。"""
    if not isinstance(accepted, dict):
        return ["验收身份不是对象"]
    if not isinstance(actual, dict):
        return ["当前身份不是对象"]
    missing = [name for name in REQUIRED_SECTIONS if name not in accepted]
    if missing:
        return [f"验收身份缺少小节：{', '.join(missing)}"]
    problems: List[str] = []
    _diff(accepted, actual, "", problems)
    return problems


def assert_compatible(accepted_payload: dict, actual: dict, *,
                      generation: str, batch: str,
                      accepted_path: Optional[str] = None) -> None:
    """外部 Agent 启动前的最后一道检查。不一致就拒绝，并指出每个字段。"""
    identity = (accepted_payload or {}).get("identity") or accepted_payload
    problems = compatibility_problems(identity, actual)
    if not problems:
        return
    shown = problems[:20]
    more = "" if len(problems) <= 20 else f"\n  …（另有 {len(problems) - 20} 项）"
    where = f"（登记文件：{accepted_path}）" if accepted_path else ""
    raise SystemExit(
        f"外部 Agent 评测被拒绝：即将运行的配置与 {generation}/{batch} 的验收协议"
        f"身份不一致{where}。\n  " + "\n  ".join(shown) + more + "\n"
        "门禁比较的是构建出来的真实对象，不是命令行声明。若这是有意的协议变更，"
        "先重新跑 b2、重新人工验收并重新登记，不能用旧验收覆盖新配置。")


# ---------------------------------------------------------------------------
# 验收身份文件的读写
# ---------------------------------------------------------------------------

def accepted_path(generation: str, batch: str) -> str:
    return os.path.join(PROJECT_ROOT, "sft_data", generation, "review", batch,
                        "accepted_protocol.json")


def load_accepted(path: str) -> dict:
    """读验收身份文件；缺失/损坏/schema 不符/缺小节都明确拒绝。"""
    if not os.path.exists(path):
        raise SystemExit(
            f"外部 Agent 评测被拒绝：{path} 不存在——{os.path.basename(path)} "
            "尚未登记。先完成人工验收，再运行 scripts/g8_accept_protocol.py。")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"验收协议身份文件无法读取：{path}（{type(exc).__name__}: {exc}）")
    if not isinstance(payload, dict):
        raise SystemExit(f"验收协议身份文件不是对象：{path}")
    if payload.get("schema") != SCHEMA:
        raise SystemExit(
            f"验收协议身份的 schema 不是 {SCHEMA}：{path}"
            f"（实际 {payload.get('schema')!r}）")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise SystemExit(f"验收协议身份缺少 identity 对象：{path}")
    missing = [name for name in REQUIRED_SECTIONS if name not in identity]
    if missing:
        raise SystemExit(f"验收协议身份缺少小节 {', '.join(missing)}：{path}")
    stored = payload.get("identity_sha256")
    if not stored:
        raise SystemExit(f"验收协议身份缺少 identity_sha256，无法核对文件完整性：{path}")
    if stored != identity_sha256(identity):
        raise SystemExit(
            f"验收协议身份的 identity 与 identity_sha256 不符（文件被改过？）：{path}")
    return payload

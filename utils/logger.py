"""
Session trace logger and operation log manager.

Two responsibilities:
1. SessionLogger — records every conversation turn during a forensic session
   and exports to ShareGPT-format JSON for SFT training.
2. Operation log — appends structured Markdown entries to claude_operation_log.md
   per the project audit specification (agent.md §1.2).
"""

import json
import os
from datetime import datetime
from typing import Optional

from config import SFT_SESSIONS_DIR, OPERATION_LOG_PATH


class SessionLogger:
    """
    Per-session trace logger that collects conversation turns, evidence chain,
    and final verdict, then saves as a ShareGPT-format SFT training sample.
    """

    def __init__(self):
        self._session_id: str = ""
        self._image_path: str = ""
        self._ground_truth: Optional[str] = None
        self._image_size: tuple = (0, 0)
        self._conversations: list = []
        self._evidence_chain: list = []
        self._final_verdict: Optional[dict] = None
        self._total_steps: int = 0
        self._halting_reason: str = ""
        self._mock_mode: str = ""
        self._start_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def init_sft_session(
        self,
        image_path: str,
        ground_truth: Optional[str] = None,
        image_size: tuple = (0, 0),
        mock_mode: str = "",
    ) -> str:
        """
        Initialise a new SFT session. Returns the session ID.

        Args:
            image_path: Path to the image being analysed.
            ground_truth: 'Real' or 'Fake' (from dataset directory structure).
            image_size: (height, width) of the image.
            mock_mode: Which MockMLLM mode was used.
        """
        self._start_time = datetime.now()
        timestamp = self._start_time.strftime("%Y%m%d_%H%M%S")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        self._session_id = f"session_{timestamp}_{image_name}"

        self._image_path = image_path
        self._ground_truth = ground_truth
        self._image_size = image_size
        self._mock_mode = mock_mode

        self._conversations = []
        self._evidence_chain = []
        self._final_verdict = None
        self._total_steps = 0
        self._halting_reason = ""

        # First user message: the task instruction
        self._conversations.append({
            "from": "user",
            "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱开展多轮质证。",
        })

        return self._session_id

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    def add_conversation_turn(self, from_: str, value: str) -> None:
        """Append a conversation turn (from='user' or 'gpt')."""
        self._conversations.append({"from": from_, "value": value})

    def add_evidence(self, evidence: dict) -> None:
        """Record an Evidence Token returned by an expert."""
        self._evidence_chain.append(evidence)

    def add_system_message(self, value: str) -> None:
        """Inject a system-level message (e.g. budget-exhausted warning)."""
        self._conversations.append({"from": "user", "value": value})

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize_sft(self, verdict: dict, total_steps: int, halting_reason: str) -> None:
        """
        Mark the session as complete with the final verdict.

        Args:
            verdict: Parsed verdict dict from MLLM output.
            total_steps: Number of expert-call iterations executed.
            halting_reason: Why the loop terminated (verdict_output | max_steps |
                            info_gain_converged | evidence_conflict).
        """
        self._final_verdict = verdict
        self._total_steps = total_steps
        self._halting_reason = halting_reason

    def save_sft(self) -> str:
        """
        Serialise the full session to a ShareGPT-format JSON file.
        Returns the output file path.
        """
        os.makedirs(SFT_SESSIONS_DIR, exist_ok=True)

        # Detect source model from image path
        source_model = "Unknown"
        if "GenImage_Test" in self._image_path:
            for subdir in ["ADM", "BigGAN", "Glide", "Midjourney",
                           "SD14", "SD15", "VQDM", "Wukong"]:
                if f"/{subdir}/" in self._image_path:
                    source_model = subdir
                    break
        elif "Real" in self._image_path:
            source_model = "Real"

        sft_record = {
            "id": self._session_id,
            "image_path": self._image_path,
            "ground_truth": self._ground_truth,
            "source_model": source_model,
            "final_verdict": self._final_verdict,
            "conversations": self._conversations,
            "evidence_chain": self._evidence_chain,
            "metadata": {
                "image_size": list(self._image_size),
                "total_steps": self._total_steps,
                "halting_reason": self._halting_reason,
                "mock_mode": self._mock_mode,
                "session_timestamp": self._start_time.isoformat() if self._start_time else "",
            },
        }

        filename = f"forensic_sft_{self._session_id}.json"
        filepath = os.path.join(SFT_SESSIONS_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sft_record, f, ensure_ascii=False, indent=2)

        return filepath

    # ------------------------------------------------------------------
    # Accessors (used by state machine for in-flight inspection)
    # ------------------------------------------------------------------

    @property
    def conversation(self) -> list:
        return self._conversations

    @property
    def evidence_chain(self) -> list:
        return self._evidence_chain

    @property
    def session_id(self) -> str:
        return self._session_id


# ------------------------------------------------------------------
# Operation log (claude_operation_log.md)
# ------------------------------------------------------------------

def log_operation(
    action: str,
    changes: list[str],
    files: list[str],
    result: str,
    todo: str = "",
) -> None:
    """
    Append a structured Markdown entry to claude_operation_log.md.

    Follows the format mandated by agent.md §1.2.

    Args:
        action: Short description of what was done.
        changes: List of specific change descriptions.
        files: List of file paths (with status: Created / Modified / Deleted).
        result: Execution result and verification status.
        todo: (Optional) Known limitations or remaining work.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    change_items = "\n".join(f"  {i}. {c}" for i, c in enumerate(changes, 1))
    file_items = "\n".join(f"  - `{f}`" for f in files)

    entry = f"""### {timestamp} - {action}

* **当前操作动作**：{action}
* **核心变更说明**：
{change_items}
* **涉及/修改的文件清单**：
{file_items}
* **执行结果与验证状态**：{result}
* **置信度或遗留待办（TODO）**：{todo or '无'}
---
"""
    os.makedirs(os.path.dirname(OPERATION_LOG_PATH), exist_ok=True)

    with open(OPERATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)

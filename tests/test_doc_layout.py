"""
Documentation layout (2026-10-15 reorganisation).

Two rules the user set, checked mechanically because both decay silently:

  * the root keeps **four daily documents** — anything else belongs under
    `docs/{design,development,archive}/`;
  * every relative link in the **current** documents resolves.  The archived v1
    documents are exempt: they are kept exactly as they were, links included,
    and `archive/legacy_v1_20261015/` is therefore not walked here.
"""

import os
import re

import pytest

from config import PROJECT_ROOT

ROOT_DOCS = {"README.md", "plan.md", "claude_operation_log.md", "KNOWN_ISSUES.md"}
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
SKIP_DIRS = {".git", "archive", "__pycache__", "dataset", "traces", "logs",
             "calibration", "sft_data", "experts", "mllm", "state_machine",
             "utils", "scripts", "tests"}


def _current_docs():
    """Every markdown file that describes the current system."""
    found = []
    for name in sorted(os.listdir(PROJECT_ROOT)):
        if name.endswith(".md") and os.path.isfile(os.path.join(PROJECT_ROOT, name)):
            found.append(os.path.join(PROJECT_ROOT, name))
    docs_root = os.path.join(PROJECT_ROOT, "docs")
    for base, dirs, files in os.walk(docs_root):
        dirs.sort()
        for name in sorted(files):
            if name.endswith(".md"):
                found.append(os.path.join(base, name))
    return found


class TestTheRootKeepsFourDocuments:
    def test_the_root_holds_exactly_the_four_daily_documents(self):
        present = {name for name in os.listdir(PROJECT_ROOT)
                   if name.endswith(".md")
                   and os.path.isfile(os.path.join(PROJECT_ROOT, name))}
        assert present == ROOT_DOCS, (
            f"根目录多出/缺少文档：{sorted(present ^ ROOT_DOCS)}；"
            "其余文档按用途放 docs/design、docs/development 或 docs/archive")

    def test_the_design_and_development_documents_are_where_they_moved_to(self):
        expected = {
            "docs/design/active_forensic_agent_tasks.md",
            "docs/design/Reasoning_Framework.md",
            "docs/design/CURRENT_PROGRAM_ARCHITECTURE.md",
            "docs/design/NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md",
            "docs/development/TESTING.md",
            "docs/development/agent.md",
            "docs/archive/ARCHIVE_INVENTORY.md",
            "docs/archive/archive_inventory.json",
            "docs/archive/NEW_GENERATION_NAMING.md",
        }
        missing = sorted(path for path in expected
                         if not os.path.exists(os.path.join(PROJECT_ROOT, path)))
        assert missing == [], missing


class TestTheCurrentLinksResolve:
    def test_every_relative_link_in_a_current_document_resolves(self):
        problems = []
        checked = 0
        for path in _current_docs():
            for label, url in LINK.findall(open(path, encoding="utf-8").read()):
                if url.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                target = url.split("#")[0].strip()
                if not target:
                    continue
                checked += 1
                resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
                if not os.path.exists(resolved):
                    problems.append(
                        f"{os.path.relpath(path, PROJECT_ROOT)} -> {url}")
        assert checked > 5, "链接检查没读到文档，检查本身失效了"
        assert problems == [], problems

    def test_the_proposal_points_at_the_plan_it_was_merged_into(self):
        """The proposal is kept as a design original; the live plan is plan.md."""
        text = open(os.path.join(PROJECT_ROOT, "docs/design",
                                 "NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md"),
                    encoding="utf-8").read()
        banner = "\n".join(text.split("\n")[:6])
        assert "不再是执行依据" in banner
        assert "plan.md" in banner


@pytest.mark.parametrize("name", sorted(ROOT_DOCS))
def test_the_root_documents_still_exist(name):
    assert os.path.isfile(os.path.join(PROJECT_ROOT, name)), name

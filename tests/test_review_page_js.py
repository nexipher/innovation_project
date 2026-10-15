"""
The review page's browser-side rules, run in Node (stage 4.6).

The page's JS decides whether a draft survives a refresh, whether a rebuilt page
inherits an old decision, and whether a foreign export can be imported.  Those
are exactly the rules a reviewer's hours of work hang on, and "it is only
JavaScript" is not a reason to leave them untested: Node is on the machine, the
page's own script is a value in the Python module, so the test concatenates
stubs + the real script + assertions and runs it.

Skips when Node is absent (the Python-side mirror in `test_review_store.py` still
covers the same invariants, but the page's copy is what runs in the browser).
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core

JS_DIR = os.path.join(PROJECT_ROOT, "tests", "js")

META = {
    "schema": "g8_review_decisions/1",
    "generation": "g8",
    "batch": "b1",
    "samples": {"g8_default__MJ_1_png": {"fingerprint": "aaa111", "page": "p.html"}},
    "repo_decisions": {},
}


def _harness(tmp_path) -> str:
    from scripts.g8_review_workstation import JS

    parts = [
        open(os.path.join(JS_DIR, "review_page_stubs.js"), encoding="utf-8").read(),
        JS.replace("__META__", json.dumps(META, ensure_ascii=False)),
        open(os.path.join(JS_DIR, "review_page_assertions.js"), encoding="utf-8").read(),
    ]
    path = os.path.join(str(tmp_path), "harness.js")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return path


@pytest.mark.skipif(shutil.which("node") is None, reason="机器上没有 node")
class TestThePagesBrowserRules:
    def test_all_assertions_pass(self, tmp_path):
        result = subprocess.run([shutil.which("node"), _harness(tmp_path)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        # 断言真的跑了（不是空跑）
        assert result.stdout.count("ok:") >= 5, result.stdout

    def test_the_harness_uses_the_pages_own_script(self):
        """拼进去的必须是页面自己那段 JS——测一份副本等于没测。"""
        from scripts.g8_review_workstation import JS

        assert "function persist(" in JS and "function stateOf(" in JS
        assert "function buildExport(" in JS

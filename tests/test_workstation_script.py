"""
Run the review page's own JavaScript, in node, with a DOM stub.

Two defects have now reached the user through this page — a nested <script>
that killed the whole block, and an export button that threw
`ReferenceError: BATCH is not defined` on click — and both looked the same from
the outside: "the button does nothing".  Static checks did not catch either.

So this test loads the generated page's script into a small fake DOM, clicks
the export button, and asserts that a decision comes out the other end.  It is
the only test here that proves the page works for the person using it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import pytest

from config import PROJECT_ROOT

INDEX = os.path.join(PROJECT_ROOT, "sft_data", "review", "g7_b1", "index.html")

STUB = r"""
// Minimal DOM for the review page: elements with listeners, an in-memory
// localStorage, and enough of Blob/URL for the download step.
const listeners = {};
function element(id) {
  return {
    id: id, value: '', textContent: '', checked: false,
    style: {}, dataset: {},
    addEventListener: (type, fn) => { (listeners[id] = listeners[id] || {})[type] = fn; },
    appendChild: () => {}, remove: () => {}, setAttribute: () => {},
    getAttribute: (name) => (name === 'data-samples' ? '145' : null),
    querySelectorAll: () => [], querySelector: () => null,
    click: () => { const l = (listeners[id] || {}).click; if (l) l({ preventDefault() {} }); },
  };
}
const store = {
  'finalv2review.v1': JSON.stringify({
    reviewer: 'tester',
    decisions: { 'f2_probe__x_png': { decision: 'accept', notes: '', saved_at: 'now' } },
  }),
};
global.window = { __batch: { id: 'g7_b1', fingerprint: 'TESTFINGERPRINT' } };
global.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: (k) => { delete store[k]; },
};
const elements = {};
global.document = {
  body: { getAttribute: (name) => (name === 'data-samples' ? '145' : null) },
  getElementById: (id) => (elements[id] = elements[id] || element(id)),
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => element('created'),
  addEventListener: (type, fn) => { (listeners.document = listeners.document || {})[type] = fn; },
};
global.Blob = class { constructor(parts) { this.parts = parts; } };
global.URL = { createObjectURL: () => 'blob:x' };
global.FileReader = class { readAsText() {} };

function run(script) { eval(script); }

"""


def _scripts(html: str):
    return re.findall(r"<script>(.*?)</script>", html, re.S)


@pytest.fixture(scope="module")
def page_scripts():
    if not os.path.exists(INDEX):
        pytest.skip("审阅工作台尚未构建")
    with open(INDEX, encoding="utf-8") as handle:
        return _scripts(handle.read())


class TestTheReviewPageRuns:
    def test_the_export_button_yields_a_payload(self, tmp_path, page_scripts):
        """Click it in node and read what it produced."""
        if shutil.which("node") is None:
            pytest.skip("node 不可用")

        runner = tmp_path / "run.js"
        runner.write_text(
            STUB
            + "\nlet payload = null;"
            + "\nglobal.Blob = class { constructor(parts) { payload = parts[0]; } };"
            + "\nfor (const script of SCRIPTS) { run(script); }"
            + "\nif (listeners.document && listeners.document.DOMContentLoaded)"
            + "  listeners.document.DOMContentLoaded({});"
            + "\nelements['export'].click();"
            + "\nconsole.log('PAYLOAD ' + JSON.stringify(JSON.parse(payload)));"
            + "\n",
            encoding="utf-8")
        # SCRIPTS comes from the page itself, in order
        scripts_json = json.dumps(page_scripts)
        environment = f"const SCRIPTS = {scripts_json};\n"
        runner.write_text(environment + runner.read_text(encoding="utf-8"),
                          encoding="utf-8")

        result = subprocess.run(["node", str(runner)], capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip(), "导出没有产出任何内容"
        line = [l for l in result.stdout.splitlines() if l.startswith("PAYLOAD ")][-1]
        payload = json.loads(line[len("PAYLOAD "):])
        assert payload["batch_id"] == "g7_b1"
        # the fingerprint the page itself carries (the injected script wins
        # over the stub's value, which is the point of injecting it)
        assert payload["batch_fingerprint"] == "bb64cb8ec42788df"
        assert payload["reviewer"] == "tester"
        assert "f2_probe__x_png" in payload["decisions"]

    def test_no_page_references_an_undefined_batch_global(self, page_scripts):
        """`BATCH.id` with only `window.__batch` defined is a ReferenceError."""
        for script in page_scripts:
            if "window.__batch" in script:      # the injected one-liner
                continue
            assert "BATCH.id" not in script or "var BATCH" in script
            assert "BATCH.fingerprint" not in script or "var BATCH" in script

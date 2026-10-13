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


class TestTheReviewStoreIsPerBatch:
    """
    One shared localStorage key meant the g7_b2 page counted the reviewer's
    145 g7_b1 decisions as its own progress, and could export them under the
    g7_b2 fingerprint.  Storage is now keyed by batch id **and** content
    fingerprint, progress/export only consider this batch's sample ids, and
    import refuses a payload from another batch.
    """

    STUB_EXTRA = r"""
let imported = null;
global.FileReader = class {
  readAsText() { this.result = IMPORT_TEXT; this.onload({ target: { result: IMPORT_TEXT } }); }
};
function loadPage(html) {
  eval(html.match(/<script>window\.__batch[^<]*<\/script>/)[0].replace(/<\/?script>/g, ''));
  const blocks = html.match(/<script>[\s\S]*?<\/script>/g) || [];
  blocks.forEach((b) => {
    const body = b.replace(/^<script>/, '').replace(/<\/script>$/, '');
    if (body.indexOf('finalv2review') !== -1) { eval(body); }
  });
}
"""

    def _build(self, tmp_path, batch_id, samples):
        import json as jsonlib

        from scripts.build_review_workstation import build

        samples_dir = tmp_path / f"batch_{batch_id}"
        samples_dir.mkdir(exist_ok=True)
        (samples_dir / "sft_tool_positive.json").write_text(
            jsonlib.dumps(samples), encoding="utf-8")
        (tmp_path / "none").mkdir(exist_ok=True)
        (tmp_path / "none.json").write_text("{}", encoding="utf-8")
        out = tmp_path / f"review_{batch_id}"
        build(str(samples_dir), str(out), trajectories_dir=str(tmp_path / "none"),
              variants_manifest=str(tmp_path / "none.json"), batch_id=batch_id)
        return (out / "index.html").read_text(encoding="utf-8")

    def _run(self, tmp_path, html, store=None, import_payload=None):
        import json as jsonlib

        from tests.test_batch_binding import _sample

        _ = _sample  # the fixture factory lives with the batch tests
        runner = tmp_path / "run_case.js"
        runner.write_text(
            "const store = " + jsonlib.dumps(store or {}) + ";\n"
            + "const IMPORT_TEXT = " + jsonlib.dumps(jsonlib.dumps(import_payload)) + ";\n"
            + re.sub(r"const store = \{.*?\n\};", "/* store seeded by the test */", STUB,
                     count=1, flags=re.S)
              .replace("global.window = { __batch: { id: 'g7_b1', fingerprint: 'TESTFINGERPRINT' } };",
                       "global.window = {};")
            + self.STUB_EXTRA
            + "\ndocument.body.appendChild = () => {};"
            + "\nconst TOTAL = " + str(len(jsonlib.loads(
                re.search(r'window\.__samples = (\[.*?\]);', html).group(1)))) + ";"
            + "\ndocument.body.getAttribute = (n) => (n === 'data-samples' ? String(TOTAL) : null);"
            + "\nloadPage(PAGE);"
            + "\nif (listeners.document && listeners.document.DOMContentLoaded)"
            + "  listeners.document.DOMContentLoaded({});"
            + "\nconsole.log('PROGRESS ' + (elements['save-status'] || {}).textContent);"
            + "\nif (listeners['reviewer'] && listeners['reviewer'].change) {"
            + " elements['reviewer'].value = 'tester'; listeners['reviewer'].change({}); }"
            + "\nlet payload = null; global.Blob = class { constructor(parts) { payload = parts[0]; } };"
            + "\nif (imported !== null || IMPORT_TEXT !== 'null') {"
            + " if (listeners['import'] && listeners['import'].change)"
            + "   listeners['import'].change({ target: { files: [{}] } }); }"
            + "\nelements['export'].click();"
            + "\nconsole.log('PAYLOAD ' + JSON.stringify(JSON.parse(payload)));"
            + "\nif (IMPORT_TEXT !== 'null') {"
            + " if (listeners['import'] && listeners['import'].change)"
            + "   listeners['import'].change({ target: { files: [{}] } }); }"
            + "\nconsole.log('STATUS ' + (elements['save-status'] || {}).textContent);"
            + "\nconsole.log('KEYS ' + JSON.stringify(Object.keys(store)));"
            + "\n",
            encoding="utf-8")
        environment = f"const PAGE = {json.dumps(html)};\n"
        runner.write_text(environment + runner.read_text(encoding="utf-8"),
                          encoding="utf-8")
        result = subprocess.run(["node", str(runner)], capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == 0, result.stderr[-2000:]
        payload = json.loads([l for l in result.stdout.splitlines()
                              if l.startswith("PAYLOAD ")][-1][len("PAYLOAD "):])
        status = [l for l in result.stdout.splitlines() if l.startswith("STATUS ")][-1]
        progress = [l for l in result.stdout.splitlines() if l.startswith("PROGRESS ")][-1]
        keys = json.loads([l for l in result.stdout.splitlines()
                           if l.startswith("KEYS ")][-1][len("KEYS "):])
        _ = self
        return payload, status, keys, progress

    def test_the_key_contains_the_batch_and_its_fingerprint(self, tmp_path):
        if shutil.which("node") is None:
            pytest.skip("node 不可用")
        from tests.test_batch_binding import _sample

        html = self._build(tmp_path, "g7_b2", [_sample("a"), _sample("b")])
        payload, _, keys, _ = self._run(tmp_path, html)
        assert payload["batch_id"] == "g7_b2"
        # the key the page wrote is scoped; nothing else was touched
        assert any(key.startswith("finalv2review.v1:g7_b2:") for key in keys), keys

    def test_progress_and_export_ignore_other_batches(self, tmp_path):
        if shutil.which("node") is None:
            pytest.skip("node 不可用")
        from tests.test_batch_binding import _sample

        html = self._build(tmp_path, "g7_b2", [_sample("a"), _sample("b")])
        fingerprint = json.loads(re.search(
            r'window\.__batch = (\{.*?\});', html).group(1))["fingerprint"]
        store = {
            f"finalv2review.v1:g7_b2:{fingerprint}": json.dumps({
                "reviewer": "tester",
                "decisions": {
                    "a": {"decision": "accept", "notes": ""},
                    "f2_probe__old_batch_png": {"decision": "reject", "notes": ""},
                }}),
        }
        payload, status, _, progress = self._run(tmp_path, html, store)
        assert set(payload["decisions"]) == {"a"}, payload["decisions"]
        assert "另有 1 条属于其他批次" in status
        assert "1 / 2" in progress, progress

    def test_an_import_from_another_batch_is_refused(self, tmp_path):
        if shutil.which("node") is None:
            pytest.skip("node 不可用")
        from tests.test_batch_binding import _sample

        html = self._build(tmp_path, "g7_b2", [_sample("a")])
        payload, status, _, _ = self._run(tmp_path, html, store={},
                                          import_payload={
                                           "batch_id": "g7_b1",
                                           "batch_fingerprint": "bb64cb8ec42788df",
                                           "decisions": {"a": {"decision": "accept"}}})
        assert payload["decisions"] == {}
        keys = [line for line in [status] if True]
        assert keys  # status is read; the assertion below is the real check
        assert "导入被拒绝" in status

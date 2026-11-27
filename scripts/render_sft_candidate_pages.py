"""Re-render the 24 candidate session pages from saved records and traces.

The tool runs are not repeated: this only rebuilds the HTML after a workbench
change (for example, the scripted-assistant banner).  Records, traces and
evidence stay exactly as recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from scripts.g8_review_workstation import session_page  # noqa: E402
from utils.review_store import sample_fingerprint  # noqa: E402

DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    batch_meta = json.loads((out / "batch_meta.json").read_text(encoding="utf-8"))
    records = json.loads((out / "records.json").read_text(encoding="utf-8"))
    for record in records:
        trace_path = ROOT / record["trace_path"]
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        page = record["page"]
        page_html = session_page(
            {"record": record, "trace": trace, "page": page,
             "fingerprint": sample_fingerprint(record, trace),
             "source_image": record["source_image"]},
            str(out), "index.html", batch_meta=batch_meta,
            meta={"schema": "sft_candidate/1", "generation": "g8",
                  "batch": out.name, "samples": {}, "repo_decisions": {}})
        (out / page).write_text(page_html, encoding="utf-8")
    print(json.dumps({"rendered": len(records), "out": str(out)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

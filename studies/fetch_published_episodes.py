"""Acquire the public, mutable TARSphere website's authored episode exports."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import yaml

from tarsphere.bench.case_loader import BenchmarkCaseLoader

PAGE = "https://xuyicheng-zju.github.io/TARSphere/benchmark.html"


def parse_index(text: str) -> dict:
    prefix = "window.TARSPHERE_BENCHMARK = "
    if not text.startswith(prefix):
        raise ValueError("Unrecognized benchmark index assignment")
    index = json.loads(text[len(prefix):].strip().removesuffix(";"))
    episodes = index["episodes"]
    ids = [e["id"] for e in episodes]
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError("Expected 360 unique published episode IDs")
    if any(not re.fullmatch(r"[a-z0-9_]+", eid) for eid in ids):
        raise ValueError("Unsafe episode ID")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Fresh external output directory")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        parser.error("timeout must be in (0, 300] seconds")
    root = args.output.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        parser.error("download output must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)

    def fetch(url: str) -> bytes:
        if urlparse(url).netloc != urlparse(PAGE).netloc:
            raise ValueError("Refusing a cross-host source URL")
        with urlopen(url, timeout=args.timeout) as response:
            return response.read()

    page = fetch(PAGE)
    (root / "benchmark.html").write_bytes(page)
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', page.decode())
    index_url = urljoin(PAGE, next(s for s in scripts if "benchmark-data.js" in s))
    detail_url = urljoin(PAGE, next(s for s in scripts if s.startswith("episodes.js")))
    raw_index, detail_script = fetch(index_url), fetch(detail_url)
    (root / "benchmark-data.js").write_bytes(raw_index)
    (root / "episodes.js").write_bytes(detail_script)
    # Fail if the site's actual JSON-link convention changes.
    if 'assets/episodes/${summary.id}.json' not in detail_script.decode():
        raise ValueError("Unrecognized explorer detail URL convention")
    index = parse_index(raw_index.decode())
    (root / "exports").mkdir()
    (root / "definitions").mkdir()
    loader = BenchmarkCaseLoader()

    def acquire(summary: dict) -> dict:
        eid = summary["id"]
        url = urljoin(PAGE, f"assets/episodes/{eid}.json")
        raw = fetch(url)
        (root / "exports" / f"{eid}.json").write_bytes(raw)
        export = json.loads(raw)
        if export["id"] != eid or export["definition"]["episode_id"] != eid:
            raise ValueError(f"Export identity mismatch: {eid}")
        path = root / "definitions" / f"{eid}.yaml"
        path.write_text(yaml.safe_dump(export["definition"], sort_keys=False))
        loader.load_file(path)
        return {"episode_id": eid, "path": f"definitions/{eid}.yaml",
                "url": url, "source": export["source"]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries = list(pool.map(acquire, index["episodes"]))
    (root / "index.json").write_text(json.dumps({"episodes": entries}, indent=2))
    (root / "acquisition.json").write_text(json.dumps({
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_url": PAGE, "index_url": index_url, "detail_script_url": detail_url,
        "episodes": len(entries), "schema_validated": True,
        "scope": "Mutable website export; schema validation does not verify physical assets or execution",
    }, indent=2))
    print(f"Acquired and schema-validated {len(entries)} episodes in {root}")


if __name__ == "__main__":
    main()

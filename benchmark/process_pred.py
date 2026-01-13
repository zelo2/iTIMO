"""
Parse raw model outputs into structured responses under results_parsed/.

Supports multiple sources:
  - prompt_results/ (prompt_eval_*.json)
  - SFT_predictions_fullft/
  - SFT_predictions_lora/
  - legacy SFT_results/

By default, only prediction files are processed (e.g., *_example.json,
*_predictions.json, prompt_eval_*.json). Use --all to process every *.json
under the given roots.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from json_repair import repair_json


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def parse_response(resp):
    """Best-effort repair of model response into a dict."""
    if resp is None:
        return None

    # If already dict, return as-is
    if isinstance(resp, dict):
        return resp

    # If list, prefer the last dict; else try repairing the last string
    if isinstance(resp, list):
        for obj in reversed(resp):
            if isinstance(obj, dict):
                return obj
        for obj in reversed(resp):
            if isinstance(obj, str):
                try:
                    return parse_response(obj)
                except Exception:
                    pass
        return None

    # If string, strip think blocks, repair JSON, then load
    if isinstance(resp, str):
        txt = re.sub(r"<think>.*?</think>\s*", "", resp, flags=re.S | re.I)
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, flags=re.S | re.I)
        if m:
            txt = m.group(1)
        fixed = repair_json(txt)
        return json.loads(fixed)

    return None


def should_process_file(path: Path, *, process_all: bool) -> bool:
    """Filter for prediction files."""
    if process_all:
        return path.suffix == ".json"
    name = path.name
    return (
        name.endswith("_example.json")
        or name.endswith("_predictions.json")
        or (name.startswith("prompt_eval_") and name.endswith(".json"))
    )


# --------------------------------------------------------------------------- #
# File processing
# --------------------------------------------------------------------------- #

def process_one_file(src_path: Path, src_root: Path, dst_root: Path) -> bool:
    """
    Load src_path, parse response, write to mirrored path under dst_root/<root.name>/...
    Returns True if written successfully.
    """
    try:
        with src_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] cannot load {src_path}: {e}")
        return False

    if not isinstance(data, dict):
        print(f"[SKIP] top-level not dict: {src_path}")
        return False

    bad = 0
    for k, v in data.items():
        if isinstance(v, dict) and "response" in v:
            try:
                v["response"] = parse_response(v.get("response"))
            except Exception as e:
                bad += 1
                v.setdefault("parse_error", str(e))  # record error but continue

    rel = src_path.relative_to(src_root)
    dst_path = dst_root / src_root.name / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[WRITE] {dst_path}  (bad={bad})")
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DEFAULT_ROOTS = [
    "prompt_results",
    "SFT_predictions_fullft",
    "SFT_predictions_lora",
    "SFT_results",  # legacy
]


def iter_source_files(roots: List[Path], glob_pattern: str, process_all: bool) -> Iterable[Tuple[Path, Path]]:
    for root in roots:
        if not root.exists():
            print(f"[SKIP ROOT] not found: {root}")
            continue
        files = [p for p in root.rglob(glob_pattern) if should_process_file(p, process_all=process_all)]
        print(f"[ROOT] {root} matched {len(files)} file(s)")
        for p in files:
            yield root, p


def parse_args():
    parser = argparse.ArgumentParser(description="Parse raw prediction JSONs into structured responses.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help=f"Source roots (default: {', '.join(DEFAULT_ROOTS)} if they exist).",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob pattern relative to each root (default: *.json).",
    )
    parser.add_argument(
        "--dst-root",
        default="results_parsed",
        help="Destination root for parsed files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all *.json under roots (ignore filename heuristics).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    roots = [Path(r) for r in (args.roots or DEFAULT_ROOTS)]
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    items = list(iter_source_files(roots, args.glob, process_all=args.all))
    print(f"Total matched files: {len(items)}")

    written = 0
    for src_root, path in items:
        if process_one_file(path, src_root, dst_root):
            written += 1

    print(f"Done. Written {written}/{len(items)} files into {dst_root}.")


if __name__ == "__main__":
    main()

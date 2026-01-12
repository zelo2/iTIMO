import json
import re
from pathlib import Path
from json_repair import repair_json

def parse_response(resp):
    """Repair response string to dict; if already dict, return as-is."""
    if not isinstance(resp, str):
        return resp

    # 1) Strip <think>…</think>
    resp = re.sub(r"<think>.*?</think>\s*", "", resp, flags=re.S | re.I)

    # 2) Prefer objects inside ```json ...``` or ``` ...```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp, flags=re.S | re.I)
    if m:
        resp = m.group(1)

    # 3) Repair + parse
    fixed = repair_json(resp)
    return json.loads(fixed)


def should_process_file(p: Path) -> bool:
    """
    Only process evaluation result files, avoid examples/token_usage, etc.
    Rule: filename ends with _example.json.
    """
    return p.name.endswith("_example.json")


def process_one_file(src_path: Path, src_root: Path, dst_root: Path) -> bool:
    """
    Load src_path, parse response, write to mirrored path under dst_root.
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

    # Compute destination mirrored path
    rel = src_path.relative_to(src_root)
    dst_path = dst_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[WRITE] {dst_path}  (bad={bad})")
    return True


def main():
    src_root = Path("SFT_results")
    dst_root = Path("results_parsed")

    files = [p for p in src_root.rglob("*.json") if should_process_file(p)]
    print(f"Found {len(files)} *_example.json files under {src_root}")

    written = 0
    for p in files:
        if process_one_file(p, src_root, dst_root):
            written += 1

    print(f"Done. Written {written}/{len(files)} files into {dst_root}.")


if __name__ == "__main__":
    main()

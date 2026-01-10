import json
import re
from pathlib import Path
from json_repair import repair_json

def parse_response(resp):
    """把 response 字符串修复并转成 dict。若已是 dict 就原样返回。"""
    if not isinstance(resp, str):
        return resp

    # 1) 去掉 <think>…</think>
    resp = re.sub(r"<think>.*?</think>\s*", "", resp, flags=re.S | re.I)

    # 2) 优先提取 ```json ... ``` 或 ``` ... ``` 里的对象
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp, flags=re.S | re.I)
    if m:
        resp = m.group(1)

    # 3) 纠错 + 解析
    fixed = repair_json(resp)
    return json.loads(fixed)


def should_process_file(p: Path) -> bool:
    """
    只处理评测结果文件，避免误处理 examples/token_usage 等。
    规则：文件名以 _example.json 结尾。
    """
    return p.name.endswith("_example.json")


def process_one_file(src_path: Path, src_root: Path, dst_root: Path) -> bool:
    """
    读取 src_path，解析 response，写到 dst_root 的镜像路径。
    返回 True 表示成功写出了文件。
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
                v.setdefault("parse_error", str(e))  # 记录错误但不中断

    # 计算目标镜像路径
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

# build_rec_from_train.py
# -*- coding: utf-8 -*-
import os
import re
import json
from collections import defaultdict

ROOT = "SFT_data"   # 你的 train/val/test 都在这个目录下


# --------- 工具函数 ---------
def get_hint(entry):
    """安全地取出 example_input.hint 并 strip。"""
    try:
        return (entry.get("example_input", {})).get("hint", "").strip()
    except Exception:
        return ""


def get_insert_index(entry):
    eo = entry.get("example_output", {}) if isinstance(entry, dict) else {}
    for key in (
        "insert_index",
        "replaced_index",
        "removed_index",
        "select_index",
        "selected_index",
    ):
        if key in eo:
            return eo[key]
    return None


def get_selected_cand_id(entry):
    eo = entry.get("example_output", {}) if isinstance(entry, dict) else {}
    for key in (
        "selected_cand_id",
        "selected_candidate_index",
        "selected_cand_idx",
        "cand_id",
    ):
        if key in eo:
            return eo[key]
    return None


def select_diverse_ids(train_data, source_ids, current_id, k=5):
    """
    从同 hint 的 train 样本集合里挑最多 k 个，尽量让 insert_index 和 selected_cand_id 多样化。

    train_data: 只包含 train 的 dict
    source_ids: 同 hint 的所有 train 样本 id 列表
    current_id: 当前样本 id（train/val/test 都可以）
    """
    # 备选（排除自己，如果 current_id 也在 train 里）
    cand = []
    for sid in source_ids:
        if sid == current_id:
            continue
        e = train_data[sid]
        idx = get_insert_index(e)
        cid = get_selected_cand_id(e)
        cand.append((sid, idx, cid))

    used_idx, used_cid = set(), set()
    picked = []

    # 先按 index 分组，尽量“一组只取一个”，同时优先拿没用过的 cand_id
    idx_groups = defaultdict(list)
    for sid, idx, cid in cand:
        idx_groups[idx].append((sid, idx, cid))

    for idx in sorted(idx_groups.keys(), key=lambda x: (x is None, x)):
        for (sid, _idx, cid) in sorted(
            idx_groups[idx], key=lambda t: (t[2] in used_cid, str(t[0]))
        ):
            if len(picked) >= k:
                break
            if _idx not in used_idx and (cid is None or cid not in used_cid):
                picked.append(sid)
                if _idx is not None:
                    used_idx.add(_idx)
                if cid is not None:
                    used_cid.add(cid)
                break
        if len(picked) >= k:
            break

    # 不足 k 个则继续补：优先选择“同时带来新 index 与新 cand_id”的样本
    if len(picked) < k:
        remaining = [t for t in cand if t[0] not in picked]

        def novelty(t):
            _sid, _idx, _cid = t
            score = 0
            if _idx not in used_idx:
                score += 1
            if _cid is None or _cid not in used_cid:
                score += 1
            return score

        for sid, idx, cid in sorted(
            remaining, key=lambda t: (-novelty(t), str(t[0]))
        ):
            if len(picked) >= k:
                break
            picked.append(sid)
            if idx is not None:
                used_idx.add(idx)
            if cid is not None:
                used_cid.add(cid)

    return picked[:k]


# --------- 针对某个 (city, op) 处理 train/val/test ---------
def process_city_op(city, op, k=5):
    """
    对一个 city+op 组合：
        SFT_data/{city}_{op}_train.json
        SFT_data/{city}_{op}_val.json
        SFT_data/{city}_{op}_test.json
    三个文件的 rec_exmaples 都只从 train.json 里选。
    """
    train_path = os.path.join(ROOT, f"{city}_{op}_train.json")
    if not os.path.exists(train_path):
        print(f"[SKIP] {train_path} 不存在，跳过 {city}-{op}")
        return

    print(f"\n=== Processing {city} - {op} ===")
    # ---- 1) 读 train，基于 train 建立 hint 分组 ----
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    hint_to_ids = defaultdict(list)
    for sid, entry in train_data.items():
        hint = get_hint(entry)
        hint_to_ids[hint].append(sid)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Unique hints in train: {len(hint_to_ids)}")

    # ---- 2) 对 train/val/test 逐个文件生成 rec_exmaples ----
    for split in ["train", "val", "test"]:
        path = os.path.join(ROOT, f"{city}_{op}_{split}.json")
        if not os.path.exists(path):
            print(f"  [{split}] {path} 不存在，跳过。")
            continue

        print(f"  [{split}] building rec_exmaples from TRAIN pool ...")
        with open(path, "r", encoding="utf-8") as f:
            data_split = json.load(f)

        augmented = {}
        for sid, entry in data_split.items():
            h = get_hint(entry)
            pool = hint_to_ids.get(h, [])  # 只用 train 中同 hint 的 ids
            rec_list = select_diverse_ids(train_data, pool, current_id=sid, k=k)
            new_entry = dict(entry)
            # 注意字段名保持为 rec_exmaples（跟你之前代码一致）
            new_entry["rec_exmaples"] = rec_list
            augmented[sid] = new_entry

        with open(path, "w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)

        print(f"    -> {split}: {len(augmented)} items updated and saved to {path}")


if __name__ == "__main__":
    # 自动扫描 SFT_data 下所有 *_train.json，推断出 city 和 op
    train_files = [
        f for f in os.listdir(ROOT)
        if f.endswith("_train.json")
    ]
    if not train_files:
        print("No *_train.json found in SFT_data/")
        exit(0)

    print("Found train files:")
    for name in sorted(train_files):
        print("  -", name)

    # 从文件名解析 city 和 op，比如 Melb_ADD_train.json
    pattern = re.compile(r"(.+)_([A-Z]+)_train\.json")

    city_op_set = set()
    for name in train_files:
        m = pattern.match(name)
        if not m:
            continue
        city, op = m.group(1), m.group(2)
        city_op_set.add((city, op))

    for city, op in sorted(city_op_set):
        process_city_op(city, op, k=5)

    print("\nAll city/op splits processed.")

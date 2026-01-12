# build_rec_from_train.py
# -*- coding: utf-8 -*-
import os
import re
import json
from collections import defaultdict

ROOT = "SFT_data"   # train/val/test live here


# --------- Utility functions ---------
def get_hint(entry):
    """Safely extract example_input.hint and strip."""
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
    Pick up to k ids from train with the same hint, maximizing diversity on insert_index and selected_cand_id.

    train_data: dict containing only train samples
    source_ids: ids in train sharing the same hint
    current_id: current sample id (train/val/test)
    """
    # candidates (exclude self if current_id in train)
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

    # group by index; prefer one per group and new cand_id
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

    # If still short, fill with items bringing new index and cand_id
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


# --------- Process one (city, op) for train/val/test ---------
def process_city_op(city, op, k=5):
    """
    For a given city+op:
        SFT_data/{city}_{op}_train.json
        SFT_data/{city}_{op}_val.json
        SFT_data/{city}_{op}_test.json
    rec_exmaples for all splits are selected from train.json.
    """
    train_path = os.path.join(ROOT, f"{city}_{op}_train.json")
    if not os.path.exists(train_path):
        print(f"[SKIP] {train_path} not found, skip {city}-{op}")
        return

    print(f"\n=== Processing {city} - {op} ===")
    # ---- 1) read train, build hint groups ----
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    hint_to_ids = defaultdict(list)
    for sid, entry in train_data.items():
        hint = get_hint(entry)
        hint_to_ids[hint].append(sid)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Unique hints in train: {len(hint_to_ids)}")

    # ---- 2) build rec_exmaples for train/val/test ----
    for split in ["train", "val", "test"]:
        path = os.path.join(ROOT, f"{city}_{op}_{split}.json")
        if not os.path.exists(path):
            print(f"  [{split}] {path} not found, skip.")
            continue

        print(f"  [{split}] building rec_exmaples from TRAIN pool ...")
        with open(path, "r", encoding="utf-8") as f:
            data_split = json.load(f)

        augmented = {}
        for sid, entry in data_split.items():
            h = get_hint(entry)
            pool = hint_to_ids.get(h, [])  # only ids with same hint from train
            rec_list = select_diverse_ids(train_data, pool, current_id=sid, k=k)
            new_entry = dict(entry)
            # keep field name rec_exmaples (to stay consistent with existing code)
            new_entry["rec_exmaples"] = rec_list
            augmented[sid] = new_entry

        with open(path, "w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)

        print(f"    -> {split}: {len(augmented)} items updated and saved to {path}")


if __name__ == "__main__":
    # Auto-scan SFT_data for *_train.json to infer city and op
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

    # Parse city/op from filename, e.g., Melb_ADD_train.json
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

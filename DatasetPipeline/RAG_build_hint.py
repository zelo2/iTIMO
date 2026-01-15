# build_rec_from_train.py
# -*- coding: utf-8 -*-
import json
import re
from collections import defaultdict
from pathlib import Path

# Default to released benchmark splits under Benchmark/iTIMO_dataset/iTIMO-*/
# Fallback to legacy SFT_data/ if someone still keeps that layout around.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "Benchmark" / "iTIMO_dataset"
LEGACY_ROOT = REPO_ROOT / "Benchmark" / "SFT_data"


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
def process_city_op(city, op, base_dir: Path, k=5):
    """
    For a given city+op:
        {base_dir}/{city}_{op}_train.json
        {base_dir}/{city}_{op}_val.json
        {base_dir}/{city}_{op}_test.json
    rec_examples for all splits are selected from train.json.
    """
    train_path = base_dir / f"{city}_{op}_train.json"
    if not train_path.exists():
        print(f"[SKIP] {train_path} not found, skip {city}-{op}")
        return

    print(f"\n=== Processing {city} - {op} ===")
    # ---- 1) read train, build hint groups ----
    with train_path.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    hint_to_ids = defaultdict(list)
    for sid, entry in train_data.items():
        hint = get_hint(entry)
        hint_to_ids[hint].append(sid)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Unique hints in train: {len(hint_to_ids)}")

    # ---- 2) build rec_examples for train/val/test ----
    for split in ["train", "val", "test"]:
        path = base_dir / f"{city}_{op}_{split}.json"
        if not path.exists():
            print(f"  [{split}] {path} not found, skip.")
            continue

        print(f"  [{split}] building rec_examples from TRAIN pool ...")
        with path.open("r", encoding="utf-8") as f:
            data_split = json.load(f)

        augmented = {}
        for sid, entry in data_split.items():
            h = get_hint(entry)
            pool = hint_to_ids.get(h, [])  # only ids with same hint from train
            rec_list = select_diverse_ids(train_data, pool, current_id=sid, k=k)
            new_entry = dict(entry)
            new_entry["rec_examples"] = rec_list
            augmented[sid] = new_entry

        with path.open("w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)

        print(f"    -> {split}: {len(augmented)} items updated and saved to {path}")


if __name__ == "__main__":
    # Auto-scan Benchmark/iTIMO_dataset/ (preferred) and legacy SFT_data/ for *_train.json
    candidate_roots = [DATA_ROOT, LEGACY_ROOT]
    train_files = []
    for root in candidate_roots:
        if not root.exists():
            continue
        train_files.extend(root.glob("**/*_train.json"))

    if not train_files:
        print("No *_train.json found under iTIMO_dataset/ or SFT_data/")
        exit(0)

    print("Found train files:")
    for path in sorted(train_files):
        print("  -", path)

    # Parse city/op from filename, e.g., Melb_ADD_train.json
    pattern = re.compile(r"(.+)_([A-Z]+)_train\.json")

    city_op_dirs = {}
    for path in train_files:
        m = pattern.match(path.name)
        if not m:
            continue
        city, op = m.group(1), m.group(2)
        # Prefer the first hit (iTIMO_dataset is scanned before legacy SFT_data)
        city_op_dirs.setdefault((city, op), path.parent)

    for (city, op), base_dir in sorted(city_op_dirs.items()):
        process_city_op(city, op, base_dir=base_dir, k=5)

    print("\nAll city/op splits processed.")

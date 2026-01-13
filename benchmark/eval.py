# -*- coding: utf-8 -*-
"""
Batch evaluation for itinerary modification predictions.

What it computes:
  1) index / POI / edit accuracy
  2) Hint Pass Rate for popularity / category / spatial
  3) Average of three-axis Hint Pass Rate (hint_pass_rate_avg)
  4) All-axes pass rate (check_one(...)[ "ok" ])

POI accuracy rules:
  - If gold and pred both provide selected_cand_id and they match → POI correct
  - Else, if gold and pred both provide full POI (selected_poi / poi_label),
    compare after converting lon/lat to float; exact match → POI correct
  - If either condition is met, count as POI hit

Usage (run inside benchmark/):
    # parsed results (recommended)
    python eval.py --root results_parsed --glob "*_example.json"

    # raw prompt results (auto-parse responses to dict)
    python eval.py --root prompt_results --glob "*.json" --auto-parse

Examples expected in CWD (symlink or copy):
    Melb_ADD_examples.json, Melb_DELETE_examples.json, Melb_REPLACE_examples.json, ...
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from json_repair import repair_json

from hint_satis_check import check_one   # uses check_one from hint_satis_check


# ========= Helpers: extract gold / pred =========

def get_gold_index(label: Dict[str, Any]):
    """
    Extract gold index from label.
    Tries: insert_index / replaced_index / removed_index / index_label
    """
    if not isinstance(label, dict):
        return None
    for k in ("insert_index", "replaced_index", "removed_index", "index_label"):
        if k in label:
            return label[k]
    return None


def get_gold_poi(label: Dict[str, Any]):
    """
    Extract gold POI tuple/list from label.
    Prefer selected_poi / poi_label, do not fall back to candidate_id.
    """
    if not isinstance(label, dict):
        return None
    if "selected_poi" in label:
        return label["selected_poi"]
    if "poi_label" in label:
        return label["poi_label"]
    return None


def get_gold_cand_id(label: Dict[str, Any]):
    """
    Extract gold candidate id from label (if any).
    """
    if not isinstance(label, dict):
        return None
    for k in ("selected_cand_id", "cand_id", "candidate_id"):
        if k in label:
            return label[k]
    return None


def get_pred_index(resp: Dict[str, Any]):
    """
    Extract predicted index from model response (multiple key variants).
    """
    if not isinstance(resp, dict):
        return None

    candidate_keys = [
        "insert_index", "insertIdx", "insert_position",
        "replaced_index", "replace_index", "replaceIdx",
        "removed_index", "remove_index", "delete_index",
    ]
    for k in candidate_keys:
        if k in resp:
            return resp[k]

    for k in ("index", "position", "target_index"):
        if k in resp:
            return resp[k]

    return None


def get_pred_poi(resp: Dict[str, Any]):
    """
    Extract predicted POI tuple/list from response:
      - Prefer selected_poi / poi_label
      - Fallback to poi
    """
    if not isinstance(resp, dict):
        return None
    if "selected_poi" in resp:
        return resp["selected_poi"]
    if "poi_label" in resp:
        return resp["poi_label"]
    if "poi" in resp:
        return resp["poi"]
    return None


def get_pred_cand_id(resp: Dict[str, Any]):
    """
    Extract predicted candidate id from response (if any).
    """
    if not isinstance(resp, dict):
        return None
    for k in ("selected_cand_id", "cand_id", "candidate_id"):
        if k in resp:
            return resp[k]
    return None


def normalize_poi(poi: Any):
    """
    Normalize POI for comparison:
    - list/tuple: [name, cat, lon, lat, pop]; try to convert lon/lat to float
    - dict with 'poi' key: extract poi then process
    - otherwise return as-is
    """
    # If dict, try to take 'poi' field
    if isinstance(poi, dict) and "poi" in poi:
        poi = poi["poi"]

    if isinstance(poi, (list, tuple)):
        res = list(poi)
        # lon/lat typically at indices 2 and 3; try float conversion
        for idx in (2, 3):
            if idx < len(res):
                try:
                    res[idx] = float(str(res[idx]).strip())
                except Exception:
                    pass
        return res
    return poi


def safe_rate(num: int, den: int):
    return (num / den) if den else None


def json_repair_load(x: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort conversion of model response to dict:
      - if dict, return as-is
      - if list, prefer the last dict; else try repairing last string
      - if string, json-repair then json.loads
      - else, return None
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        for obj in reversed(x):
            if isinstance(obj, dict):
                return obj
        for obj in reversed(x):
            if isinstance(obj, str):
                try:
                    return json.loads(repair_json(obj))
                except Exception:
                    pass
        return None
    if isinstance(x, str):
        try:
            return json.loads(repair_json(x))
        except Exception:
            return None
    return None


# ========= Single file: Accuracy + Hint Pass =========

def eval_one_file(path: Path, examples: Optional[Dict[str, Any]], *, auto_parse: bool):
    """
    For one results_parsed/..._example.json:
      - index / poi / edit accuracy
      - popularity / category / spatial Hint Pass + All-axes Pass

    POI acc rules:
      1) If gold/pred both have candidate_id and match → correct
      2) Else if gold/pred both have full POI and normalize(...) matches → correct
      3) Either (1) or (2) counts as correct
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # accuracy section
    total_idx = correct_idx = 0
    total_poi = correct_poi = 0
    total_edit = correct_edit = 0

    # hint section
    hint_all_total = 0
    hint_all_pass = 0
    hint_axis_total = {"popularity": 0, "category": 0, "spatial": 0}
    hint_axis_pass = {"popularity": 0, "category": 0, "spatial": 0}

    skipped = 0  # invalid structure or missing label/response
    parse_failed = 0  # auto-parse failures

    for sid, rec in data.items():
        if not isinstance(rec, dict):
            skipped += 1
            continue

        label = rec.get("label")
        resp_raw = rec.get("response")
        resp = resp_raw

        # optional json-repair parsing for raw string/list responses
        if auto_parse and not isinstance(resp, dict):
            parsed = json_repair_load(resp)
            if parsed is not None:
                resp = parsed
            else:
                parse_failed += 1

        # ---------- Accuracy: requires label and dict resp ----------
        if isinstance(label, dict) and isinstance(resp, dict):
            gold_idx = get_gold_index(label)
            pred_idx = get_pred_index(resp)

            gold_poi = get_gold_poi(label)
            pred_poi = get_pred_poi(resp)

            gold_cid = get_gold_cand_id(label)
            pred_cid = get_pred_cand_id(resp)

            idx_ok = poi_ok = None

            # index accuracy
            if gold_idx is not None and pred_idx is not None:
                try:
                    gi = int(gold_idx)
                    pi = int(pred_idx)
                except Exception:
                    gi = gold_idx
                    pi = pred_idx
                total_idx += 1
                idx_ok = (gi == pi)
                if idx_ok:
                    correct_idx += 1

            # POI accuracy (candidate id preferred, else POI content)
            has_gold_cid = gold_cid is not None
            has_pred_cid = pred_cid is not None
            has_gold_poi = gold_poi is not None
            has_pred_poi = pred_poi is not None

            # Count if either candidate_id or POI can be aligned
            if (has_gold_cid and has_pred_cid) or (has_gold_poi and has_pred_poi):
                total_poi += 1
                poi_ok = False

                # 1) compare candidate_id first
                if has_gold_cid and has_pred_cid:
                    try:
                        if int(gold_cid) == int(pred_cid):
                            poi_ok = True
                    except Exception:
                        if gold_cid == pred_cid:
                            poi_ok = True

                # 2) if candidate_id missing/mismatch, compare POI content
                if (not poi_ok) and has_gold_poi and has_pred_poi:
                    poi_ok = (normalize_poi(gold_poi) == normalize_poi(pred_poi))

                if poi_ok:
                    correct_poi += 1

            # edit accuracy (index & POI both correct)
            if idx_ok is not None and poi_ok is not None:
                total_edit += 1
                if idx_ok and poi_ok:
                    correct_edit += 1
        else:
            skipped += 1

        # ---------- Hint Pass: depends on examples + resp ----------
        if examples is None or resp is None:
            continue

        sid_str = str(sid)
        ex_rec = examples.get(sid_str)
        if ex_rec is None and sid_str.isdigit():
            ex_rec = examples.get(str(int(sid_str)))
        if not isinstance(ex_rec, dict):
            continue

        example_input = ex_rec.get("example_input")
        example_output = ex_rec.get("example_output")
        if example_input is None:
            continue

        # resp may be dict / str / list; check_one handles it
        res_hint = check_one(example_input, example_output, resp)
        if not isinstance(res_hint, dict):
            continue

        # All-axes pass
        hint_all_total += 1
        if res_hint.get("ok", False):
            hint_all_pass += 1

        axes = res_hint.get("axes", {})
        if isinstance(axes, dict):
            for ax in ("popularity", "category", "spatial"):
                axr = axes.get(ax)
                if not isinstance(axr, dict):
                    continue
                if axr.get("hinted", False):
                    hint_axis_total[ax] += 1
                    if axr.get("ok", False):
                        hint_axis_pass[ax] += 1

    # summarize accuracy
    res: Dict[str, Any] = {
        "file": str(path),
        "n_records": len(data),
        "n_used_for_index": total_idx,
        "index_acc": total_idx and (correct_idx / total_idx) or None,
        "n_used_for_poi": total_poi,
        "poi_acc": total_poi and (correct_poi / total_poi) or None,
        "n_used_for_edit": total_edit,
        "edit_acc": total_edit and (correct_edit / total_edit) or None,
        "n_skipped": skipped,
        "n_parse_failed": parse_failed,
    }

    # summarize Hint Pass
    res["hint_all_total"] = hint_all_total
    res["hint_all_pass"] = hint_all_pass
    res["hint_all_pass_rate"] = safe_rate(hint_all_pass, hint_all_total)

    axis_rates = []
    for ax in ("popularity", "category", "spatial"):
        tot = hint_axis_total[ax]
        ok = hint_axis_pass[ax]
        rate = safe_rate(ok, tot)
        res[f"{ax}_hint_total"] = tot
        res[f"{ax}_hint_pass"] = ok
        res[f"{ax}_hint_pass_rate"] = rate
        if rate is not None:
            axis_rates.append(rate)

    res["hint_pass_rate_avg"] = (sum(axis_rates) / len(axis_rates)) if axis_rates else None

    return res


# ========= Path parsing & main loop =========

CITY_CODES = {"Melb", "Toro", "Florence"}
OPS = {"ADD", "DELETE", "REPLACE"}


def derive_setting(rag_mode: Optional[str], icl_num: Optional[int]) -> str:
    if rag_mode and rag_mode != "none":
        return "rag"
    if icl_num == 0:
        return "zero-shot"
    if icl_num is not None and icl_num > 0:
        return "few-shots"
    return ""


def parse_setting_from_path(path: Path, root: Path) -> Dict[str, Any]:
    """
    Parse path metadata for both legacy and prompt_eval file names.
    Examples:
      results_parsed/Melb/DELETE/model/foo_icl_3_example.json
      results_parsed/prompt_eval_openai_deepseek-chat_Melb_ADD_rag-none_icl-3_test_example.json
    """
    rel = path.relative_to(root)
    parts = rel.parts

    fname = rel.name
    stem = fname[:-5] if fname.endswith(".json") else fname
    tokens = stem.split("_")

    meta: Dict[str, Any] = {
        "city": "",
        "op": "",
        "model": "",
        "provider": "",
        "rag_mode": "",
        "split": "",
        "icl_num": None,
        "setting": "",
    }

    # Legacy nested folder: city/op/model/filename
    if len(parts) >= 4:
        meta["city"] = parts[0]
        meta["op"] = parts[1]
        meta["model"] = parts[2]

    # prompt_eval_<provider>_<model>_<city>_<op>_rag-<rag>_icl-<icl>_<split>[_example]
    if stem.startswith("prompt_eval_") and len(tokens) >= 5:
        meta["provider"] = tokens[2] if len(tokens) > 2 else ""
        meta["model"] = tokens[3] if len(tokens) > 3 else meta["model"]
        meta["city"] = tokens[4] if len(tokens) > 4 else meta["city"]
        meta["op"] = tokens[5] if len(tokens) > 5 else meta["op"]

    # Fallback: pull city/op from tokens if still missing
    if not meta["city"]:
        for t in tokens:
            if t in CITY_CODES:
                meta["city"] = t
                break
    if not meta["op"]:
        for t in tokens:
            if t in OPS:
                meta["op"] = t
                break

    # rag mode and icl num
    for tok in tokens:
        if tok.startswith("rag-"):
            meta["rag_mode"] = tok.split("-", 1)[1] or "rag"
        elif tok == "rag":
            meta["rag_mode"] = "rag"
        if tok.startswith("icl"):
            try:
                meta["icl_num"] = int(tok.split("-", 1)[1])
            except Exception:
                nums = [s for s in tok.split("-") if s.isdigit()]
                if nums:
                    try:
                        meta["icl_num"] = int(nums[0])
                    except Exception:
                        pass

    # split (test/val/train/full) if present
    for tok in reversed(tokens):
        if tok in {"train", "val", "test", "full"}:
            meta["split"] = tok
            break

    if not meta["rag_mode"]:
        meta["rag_mode"] = "none"

    meta["setting"] = derive_setting(meta["rag_mode"], meta["icl_num"])
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate itinerary modification predictions.")
    parser.add_argument(
        "--root",
        default="results_parsed",
        help="Root folder containing parsed prediction JSONs.",
    )
    parser.add_argument(
        "--glob",
        default="*_example.json",
        help="Glob pattern (relative to --root) to select prediction files.",
    )
    parser.add_argument(
        "--examples-dir",
        default=".",
        help="Directory containing <City>_<OP>_examples.json (symlink or copy).",
    )
    parser.add_argument(
        "--auto-parse",
        action="store_true",
        help="Attempt json-repair on string/list responses before computing accuracy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    files = sorted(root.rglob(args.glob))

    print(f"Found {len(files)} result files under {root} matching '{args.glob}'")

    # cache each (city, op) examples to avoid repeated disk reads
    examples_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

    all_results = []

    for path in files:
        meta = parse_setting_from_path(path, root)
        city = meta.get("city", "")
        op = meta.get("op", "")
        model = meta.get("model", "")
        setting = meta.get("setting", "")
        icl_num = meta.get("icl_num")

        key = (city, op)
        if key in examples_cache:
            examples = examples_cache[key]
        else:
            examples_path = Path(args.examples_dir) / f"{city}_{op}_examples.json"
            if examples_path.exists():
                with examples_path.open("r", encoding="utf-8") as f:
                    examples = json.load(f)
            else:
                print(f"[WARN] examples file not found for {city}/{op}: {examples_path}")
                examples = None
            examples_cache[key] = examples

        res = eval_one_file(path, examples, auto_parse=args.auto_parse)

        res.update(meta)
        all_results.append(res)

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "None"

        file_name = Path(res["file"]).name
        n_skipped = res.get("n_skipped", 0)
        n_parse_failed = res.get("n_parse_failed", 0)

        idx_acc = res.get("index_acc")
        poi_acc = res.get("poi_acc")
        edit_acc = res.get("edit_acc")

        all_pass_rate = res.get("hint_all_pass_rate")
        pop_rate = res.get("popularity_hint_pass_rate")
        cat_rate = res.get("category_hint_pass_rate")
        spa_rate = res.get("spatial_hint_pass_rate")
        hint_avg = res.get("hint_pass_rate_avg")

        provider = res.get("provider") or ""
        rag_mode = res.get("rag_mode") or ""
        split = res.get("split") or ""

        meta_desc = f"[{city}/{op}/{split}] {model}"
        if provider:
            meta_desc += f" ({provider})"
        meta_desc += f" ({setting or 'setting?'}, rag={rag_mode or 'none'}, icl={icl_num})"

        print(
            f"{file_name} | skipped={n_skipped}, parse_failed={n_parse_failed} | "
            f"{meta_desc}: "
            f"idx={fmt(idx_acc)}, poi={fmt(poi_acc)}, edit={fmt(edit_acc)}, "
            f"HintAll={fmt(all_pass_rate)}, "
            f"pop={fmt(pop_rate)}, cat={fmt(cat_rate)}, spa={fmt(spa_rate)}, "
            f"HintAvg={fmt(hint_avg)}"
        )

    out_json = root / "accuracy_hint_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to {out_json}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
一次性评估：
  1) index / poi / edit 三个 accuracy
  2) popularity / category / spatial 的 Hint Pass Rate
  3) 三轴平均 Hint Pass Rate（hint_pass_rate_avg）
  4) All-axes pass rate（check_one(...)[ "ok" ] 的通过率）

POI acc 判定规则：
  - 若 gold / pred 都给了 selected_cand_id，且相等 → 认为 POI 正确；
  - 否则，若 gold / pred 都给了完整 POI（selected_poi / poi_label），
    则在归一化经纬度为 float 后比较，只要完全一致 → 认为 POI 正确；
  - 只要满足上述任意一种，即记为 POI 命中。

用法：直接在 benchmark 目录下运行：
    python eval.py

假定目录结构：
    benchmark/
      Melb_ADD_examples.json
      Melb_DELETE_examples.json
      Melb_REPLACE_examples.json
      ...
      results_parsed/
        Melb/DELETE/xxx/xxx_example.json
        ...
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from hint_satis_check import check_one   # 这里用的是 5d9e 那个脚本里的 check_one


# ========= 工具函数：gold / pred 提取 =========

def get_gold_index(label: Dict[str, Any]):
    """
    从 label 里取“正确 index”。
    会依次尝试: insert_index / replaced_index / removed_index / index_label
    """
    if not isinstance(label, dict):
        return None
    for k in ("insert_index", "replaced_index", "removed_index", "index_label"):
        if k in label:
            return label[k]
    return None


def get_gold_poi(label: Dict[str, Any]):
    """
    从 label 里取“正确 POI 的完整 tuple/list”。
    优先用 selected_poi / poi_label，不再退化成 candidate_id。
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
    从 label 里取“正确候选 id”（如果存在）。
    """
    if not isinstance(label, dict):
        return None
    for k in ("selected_cand_id", "cand_id", "candidate_id"):
        if k in label:
            return label[k]
    return None


def get_pred_index(resp: Dict[str, Any]):
    """
    从模型 response 里取预测 index（兼容多种命名）。
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
    从模型 response 里取预测 POI 的完整 tuple/list：
      - 优先 selected_poi / poi_label
      - 兜底 poi 字段
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
    从模型 response 里取预测的候选 id（如果存在）。
    """
    if not isinstance(resp, dict):
        return None
    for k in ("selected_cand_id", "cand_id", "candidate_id"):
        if k in resp:
            return resp[k]
    return None


def normalize_poi(poi: Any):
    """
    归一化 POI 表示，用于比较：
    - 列表/tuple: [name, cat, lon, lat, pop]，会尝试把 lon/lat 转成 float
    - 字典且含 'poi' 字段 → 先取出 poi 再处理
    - 其它类型原样返回
    """
    # 如果是字典，尝试取 'poi' 字段
    if isinstance(poi, dict) and "poi" in poi:
        poi = poi["poi"]

    if isinstance(poi, (list, tuple)):
        res = list(poi)
        # 通常经纬度在下标 2,3，尝试转成 float
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


# ========= 单文件：Accuracy + Hint Pass =========

def eval_one_file(path: Path, examples: Optional[Dict[str, Any]]):
    """
    对单个 results_parsed/..._example.json 做：
      - index / poi / edit accuracy
      - popularity / category / spatial Hint Pass + All-axes Pass

    POI acc 的规则：
      1) 若 gold/pred 都有 candidate_id 且相等 → 视为正确；
      2) 否则，若 gold/pred 都有完整 POI，normalize 后完全相等 → 视为正确；
      3) 满足 (1) 或 (2) 之一即计为 POI 正确。
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # accuracy 部分
    total_idx = correct_idx = 0
    total_poi = correct_poi = 0
    total_edit = correct_edit = 0

    # hint 部分
    hint_all_total = 0
    hint_all_pass = 0
    hint_axis_total = {"popularity": 0, "category": 0, "spatial": 0}
    hint_axis_pass = {"popularity": 0, "category": 0, "spatial": 0}

    skipped = 0  # 完全结构不对 / 没 label / 没 response 的样本

    for sid, rec in data.items():
        if not isinstance(rec, dict):
            skipped += 1
            continue

        label = rec.get("label")
        resp = rec.get("response")

        # ---------- Accuracy：需要 label 和 dict 类型 resp ----------
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

            # POI accuracy（候选 id 优先，其次比 POI 内容）
            has_gold_cid = gold_cid is not None
            has_pred_cid = pred_cid is not None
            has_gold_poi = gold_poi is not None
            has_pred_poi = pred_poi is not None

            # 只要在 cand_id 或 POI 上能对齐一种，就计入分母
            if (has_gold_cid and has_pred_cid) or (has_gold_poi and has_pred_poi):
                total_poi += 1
                poi_ok = False

                # 1) 先看 candidate_id 是否一致
                if has_gold_cid and has_pred_cid:
                    try:
                        if int(gold_cid) == int(pred_cid):
                            poi_ok = True
                    except Exception:
                        if gold_cid == pred_cid:
                            poi_ok = True

                # 2) 如果 candidate_id 不一致 / 缺失，再比较 POI 内容
                if (not poi_ok) and has_gold_poi and has_pred_poi:
                    poi_ok = (normalize_poi(gold_poi) == normalize_poi(pred_poi))

                if poi_ok:
                    correct_poi += 1

            # edit accuracy（index & poi 同时对）
            if idx_ok is not None and poi_ok is not None:
                total_edit += 1
                if idx_ok and poi_ok:
                    correct_edit += 1
        else:
            skipped += 1

        # ---------- Hint Pass：依赖 examples + resp ----------
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

        # resp 可以是 dict / str / list，check_one 会自己处理
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

    # 汇总 accuracy
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
    }

    # 汇总 Hint Pass
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


# ========= 路径解析 & 主循环 =========

def parse_setting_from_path(path: Path, root: Path):
    """
    解析路径拿元信息：
    results_parsed/city/op/model/model_think[_rag]_icl_example.json
      → city, op, model, setting(zero-shot/few-shots/rag), icl_num
    """
    rel = path.relative_to(root)
    parts = rel.parts  # [city, op, model, filename]

    city = parts[0] if len(parts) >= 1 else ""
    op = parts[1] if len(parts) >= 2 else ""
    model = parts[2] if len(parts) >= 3 else ""
    fname = parts[3] if len(parts) >= 4 else path.name

    stem = fname[:-5] if fname.endswith(".json") else fname
    tokens = stem.split("_")

    setting = ""
    icl_num = None

    if "rag" in tokens:
        setting = "rag"
        try:
            idx = tokens.index("rag")
            icl_num = int(tokens[idx + 1])
        except Exception:
            pass
    else:
        nums = [t for t in tokens if t.isdigit()]
        if nums:
            try:
                icl_num = int(nums[0])
            except Exception:
                icl_num = None
        if icl_num == 0:
            setting = "zero-shot"
        elif icl_num is not None and icl_num > 0:
            setting = "few-shots"

    return city, op, model, setting, icl_num


def main():
    root = Path("results_parsed")
    files = sorted(root.rglob("*_example.json"))

    print(f"Found {len(files)} parsed result files under {root}")

    # cache 每个 (city, op) 对应的 examples，避免重复读盘
    examples_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

    all_results = []

    for path in files:
        city, op, model, setting, icl_num = parse_setting_from_path(path, root)

        key = (city, op)
        if key in examples_cache:
            examples = examples_cache[key]
        else:
            examples_path = Path(f"{city}_{op}_examples.json")
            if examples_path.exists():
                with examples_path.open("r", encoding="utf-8") as f:
                    examples = json.load(f)
            else:
                print(f"[WARN] examples file not found for {city}/{op}: {examples_path}")
                examples = None
            examples_cache[key] = examples

        res = eval_one_file(path, examples)

        res.update({
            "city": city,
            "op": op,
            "model": model,
            "setting": setting,
            "icl_num": icl_num,
        })
        all_results.append(res)

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "None"

        file_name = Path(res["file"]).name
        n_skipped = res.get("n_skipped", 0)

        idx_acc = res.get("index_acc")
        poi_acc = res.get("poi_acc")
        edit_acc = res.get("edit_acc")

        all_pass_rate = res.get("hint_all_pass_rate")
        pop_rate = res.get("popularity_hint_pass_rate")
        cat_rate = res.get("category_hint_pass_rate")
        spa_rate = res.get("spatial_hint_pass_rate")
        hint_avg = res.get("hint_pass_rate_avg")

        print(
            f"{file_name} | skipped={n_skipped} | "
            f"[{city}/{op}] {model} ({setting}, icl={icl_num}): "
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

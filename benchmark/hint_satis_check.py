# -*- coding: utf-8 -*-
"""
Hint satisfaction checker (multi-op: ADD / DELETE / REPLACE).

Main entry:
    check_one(example_input, example_output, response_obj)

Arguments:
- example_input: a record["example_input"] from examples.json
- example_output: the corresponding record["example_output"]
- response_obj: the rec["response"] for this sample in prediction file
    * can be dict / str / list([... , {...}]); function handles json-repair

Supported repair ops:
- ADD:    INSERT a POI into need_to_modify itinerary
- DELETE: DELETE a POI from need_to_modify itinerary
- REPLACE: REPLACE one POI in need_to_modify itinerary with another

Important:
- *_ADD_examples.json: trajectory was perturbed by ADD; gold repair is typically DELETE (example_output has removed_index).
- *_DELETE_examples.json: trajectory had a POI removed; gold repair is typically ADD (example_output has insert_index + selected_poi).
- *_REPLACE_examples.json: trajectory had a POI replaced; gold repair is REPLACE (replaced_index + selected_poi).

We do NOT rely on filenames; we infer from fields in example_output / response:
    insert_index / removed_index / replaced_index etc.

check_one returns:
{
  "ok": bool,            # overall pass if all hinted axes are satisfied
  "axes": {
      "popularity": { ... },
      "category":  { ... },
      "spatial":   { ... },
  },
  "meta": {
      "op_pred": str or None,
      "idx_pred": int or None,
      "low_km": float or None,
      "high_km": float or None,
  }
}
"""

import json, math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

from json_repair import repair_json
import numpy as np


__all__ = ["check_one"]


# ---------- Utilities ----------

def json_repair_load(x):
    """Safely convert response to dict:
       - if dict, return as-is;
       - if list (e.g., [analysis, final]), take last dict or repair last JSON string;
       - if string, json-repair then loads;
       - else return None.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        # prefer the last dict
        for obj in reversed(x):
            if isinstance(obj, dict):
                return obj
        # then try repairing a string to json
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


def to_float(s):
    try:
        return float(s)
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def parse_threshold_km(v, fallback=None):
    """Support '0.3km' / '0.3' / 0.3 / None"""
    if v is None:
        return fallback
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    s = s.replace("km", "").strip()
    try:
        return float(s)
    except Exception:
        return fallback


def normalize_pop(val: str):
    return str(val).strip().lower()


def hellinger(p: Dict[str, int], q: Dict[str, int], keys: List[str]) -> float:
    P = np.array([p.get(k, 0) for k in keys], dtype=float)
    Q = np.array([q.get(k, 0) for k in keys], dtype=float)
    if P.sum() == 0 and Q.sum() == 0:
        return 0.0
    if P.sum() == 0:
        P = np.ones_like(P)
    if Q.sum() == 0:
        Q = np.ones_like(Q)
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2)))


def rank_buckets(counts: Dict[str, int]) -> List[frozenset]:
    """Group same counts into buckets sorted by count desc (ignore within-tie swaps)."""
    by_cnt = defaultdict(set)
    for k, v in counts.items():
        by_cnt[int(v)].add(k)
    if not by_cnt:
        return []
    levels = sorted(by_cnt.keys(), reverse=True)
    return [frozenset(by_cnt[c]) for c in levels]


def ranking_changed(cnt_before: Dict[str, int], cnt_after: Dict[str, int]) -> bool:
    return rank_buckets(cnt_before) != rank_buckets(cnt_after)


def parse_axes_from_hint(hint: str) -> Tuple[bool, bool, bool]:
    """Return (popularity_hinted, category_hinted, spatial_hinted)."""
    s = (hint or "").lower()
    pop = ("popularity" in s)
    # cat: category / categories / diversity all count as category/diversity mentioned
    cat = ("category" in s) or ("categories" in s) or ("diversity" in s)
    # spa: spatial / distance
    spa = ("spatial" in s) or ("distance" in s)
    return pop, cat, spa


def sign_with_eps(x: float, eps: float = 1e-6) -> int:
    """Sign with tolerance: >eps -> +1, <-eps -> -1, else -> 0"""
    if x is None:
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def to_int_safe(x):
    """Safe int conversion; on failure return None"""
    try:
        return int(x)
    except Exception:
        return None


# ---------- Stats BEFORE / AFTER ----------

def pop_counts(traj: List[List[Any]]) -> Dict[str, int]:
    """Count high/medium/low"""
    keys = ["high", "medium", "low"]
    c = Counter()
    for poi in traj:
        if len(poi) >= 5:
            c[normalize_pop(poi[4])] += 1
    return {k: c.get(k, 0) for k in keys}


def cat_diversity(traj: List[List[Any]]) -> float:
    """Simple category diversity: #unique(cat) / len(traj)"""
    cats = []
    for poi in traj:
        if len(poi) >= 2:
            cats.append(str(poi[1]).strip().lower())
    uniq = len(set(cats))
    if uniq <= 1:
        return 0.0
    return uniq / max(len(traj), 1)


def spatial_counts(traj: List[List[Any]], low_km: float, high_km: float) -> Dict[str, int]:
    """Count adjacent leg distance buckets low/medium/high"""
    keys = ["low", "medium", "high"]
    c = Counter()
    if len(traj) >= 2 and low_km is not None and high_km is not None:
        for i in range(len(traj) - 1):
            lon1, lat1 = to_float(traj[i][2]), to_float(traj[i][3])
            lon2, lat2 = to_float(traj[i + 1][2]), to_float(traj[i + 1][3])
            if None in (lon1, lat1, lon2, lat2):
                continue
            d = haversine_km(lat1, lon1, lat2, lon2)
            if d < low_km:
                c["low"] += 1
            elif d > high_km:
                c["high"] += 1
            else:
                c["medium"] += 1
    return {k: c.get(k, 0) for k in keys}


# ---------- Parse operation type from label / response ----------

def get_op_and_index_from_label(label: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse gold repair op and index from example_output:
      - insert_index      -> op = "ADD"
      - removed_index     -> op = "DELETE"
      - replaced_index    -> op = "REPLACE"
      - fallback: index_label
    """
    if not isinstance(label, dict):
        return None, None

    # explicit fields first
    if "insert_index" in label:
        return "ADD", to_int_safe(label.get("insert_index"))
    if "removed_index" in label:
        return "DELETE", to_int_safe(label.get("removed_index"))
    if "replaced_index" in label:
        return "REPLACE", to_int_safe(label.get("replaced_index"))

    # fallback: index_label
    if "index_label" in label:
        return None, to_int_safe(label.get("index_label"))

    return None, None


def get_poi_from_label(label: Dict[str, Any]):
    """Get gold POI from label (used for ADD / REPLACE)"""
    if not isinstance(label, dict):
        return None
    if "selected_poi" in label:
        return label["selected_poi"]
    if "poi_label" in label:
        return label["poi_label"]
    return None


def get_op_and_index_from_resp(resp: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse predicted repair op and index from model response.
    If unable to determine op, return (None, None).
    """
    if not isinstance(resp, dict):
        return None, None

    # check most explicit keys first:
    # 1) REPLACE
    for k in ("replaced_index", "replace_index", "replaceIdx"):
        if k in resp:
            return "REPLACE", to_int_safe(resp.get(k))
    # 2) DELETE
    for k in ("removed_index", "remove_index", "delete_index"):
        if k in resp:
            return "DELETE", to_int_safe(resp.get(k))
    # 3) ADD / INSERT
    for k in ("insert_index", "insertIdx", "insert_position"):
        if k in resp:
            return "ADD", to_int_safe(resp.get(k))

    # fallback: if only index/position-like keys, op unknown -> None
    for k in ("index", "position", "target_index"):
        if k in resp:
            return None, to_int_safe(resp.get(k))

    return None, None


def get_poi_from_resp(resp: Dict[str, Any]):
    """Get POI to INSERT/REPLACE from response"""
    if not isinstance(resp, dict):
        return None
    if "selected_poi" in resp:
        return resp["selected_poi"]
    if "poi_label" in resp:
        return resp["poi_label"]
    if "poi" in resp:
        return resp["poi"]
    return None


def build_after(traj: List[List[Any]],
                op: Optional[str],
                idx: Optional[int],
                sel_poi: Optional[List[Any]]) -> Optional[List[List[Any]]]:
    """
    Build modified trajectory using op / idx / sel_poi:
      - op="ADD": insert sel_poi at idx
      - op="DELETE": remove point at idx
      - op="REPLACE": replace point at idx with sel_poi
    Return new trajectory; if invalid, return None.
    """
    if not isinstance(traj, list) or len(traj) == 0:
        return None
    if idx is None:
        return None

    n = len(traj)
    i = max(0, min(idx, n))   # clamp

    after = list(traj)

    if op == "ADD":
        # must have valid sel_poi
        if not isinstance(sel_poi, list) or len(sel_poi) < 5:
            return None
        after = after[:i] + [sel_poi] + after[i:]
        return after

    if op == "DELETE":
        if n == 0:
            return None
        # after clamp, if i==n, remove last element
        if i >= n:
            i = n - 1
        after = after[:i] + after[i+1:]
        return after

    if op == "REPLACE":
        if not isinstance(sel_poi, list) or len(sel_poi) < 5:
            return None
        if n == 0:
            return None
        if i >= n:
            i = n - 1
        after = after[:i] + [sel_poi] + after[i+1:]
        return after

    # op unknown
    return None


# ---------- Single-sample check ----------

def check_one(example_input: Dict[str, Any],
              example_output: Optional[Dict[str, Any]],
              response_obj: Any) -> Dict[str, Any]:
    """
    Check hint satisfaction for one sample.

    Inputs:
      - example_input: example_input from examples.json
      - example_output: corresponding gold label (can be None)
      - response_obj: prediction (dict / str / list etc.)

    Returns:
      {
        "ok": bool,            # all hinted axes satisfied + unmentioned axes invariant
        "axes": { ... },      # popularity / category / spatial details
        "meta": { ... },      # debug info
      }
    """

    # 1) get starting trajectory, hint, distance thresholds
    traj = example_input.get("need_to_modify itinerary") or \
           example_input.get("need_to_modify Itinerary") or []
    hint = example_input.get("hint") or ""
    low_txt = example_input.get("threshold_low") or example_input.get("threshold_low_km")
    high_txt = example_input.get("threshold_high") or example_input.get("threshold_high_km")
    low_km = parse_threshold_km(low_txt, fallback=None)
    high_km = parse_threshold_km(high_txt, fallback=None)

    # 2) parse LLM response
    resp = json_repair_load(response_obj)
    if resp is None:
        return {"ok": False, "reason": "response JSON parse fail", "axes": {}, "meta": {}}

    op_pred, idx_pred = get_op_and_index_from_resp(resp)
    sel_poi_pred = get_poi_from_resp(resp)

    after_pred = build_after(traj, op_pred, idx_pred, sel_poi_pred)

    # If after_pred cannot be built, mark as failed
    if after_pred is None:
        return {
            "ok": False,
            "reason": "cannot build after_pred",
            "axes": {},
            "meta": {"op_pred": op_pred, "idx_pred": idx_pred, "low_km": low_km, "high_km": high_km},
        }

    # 3) BEFORE / AFTER_PRED
    pop_b = pop_counts(traj)
    cd_b = cat_diversity(traj)
    spa_b = spatial_counts(traj, low_km, high_km)

    pop_p = pop_counts(after_pred)
    cd_p = cat_diversity(after_pred)
    spa_p = spatial_counts(after_pred, low_km, high_km)

    # 4) Build gold trajectory for direction check if example_output provides info
    pop_g = spa_g = None
    cd_g = None
    pop_high_g = spa_high_g = None

    pop_delta_gold = spa_delta_gold = cd_delta_gold = None

    op_gold = None
    idx_gold = None

    if isinstance(example_output, dict):
        op_gold, idx_gold = get_op_and_index_from_label(example_output)
        sel_poi_gold = get_poi_from_label(example_output)

        after_gold = build_after(traj, op_gold, idx_gold, sel_poi_gold)
        if after_gold is not None:
            pop_g = pop_counts(after_gold)
            spa_g = spatial_counts(after_gold, low_km, high_km)
            cd_g = cat_diversity(after_gold)

    # share(High) helper
    def high_share(counts: Dict[str, int]) -> float:
        tot = sum(counts.values())
        if tot <= 0:
            return 0.0
        return counts.get("high", 0) / float(tot)

    pop_high_b = high_share(pop_b)
    pop_high_p = high_share(pop_p)
    spa_high_b = high_share(spa_b)
    spa_high_p = high_share(spa_p)

    if pop_g is not None:
        pop_high_g = high_share(pop_g)
        pop_delta_gold = pop_high_g - pop_high_b
    if spa_g is not None:
        spa_high_g = high_share(spa_g)
        spa_delta_gold = spa_high_g - spa_high_b
    if cd_g is not None:
        cd_delta_gold = cd_g - cd_b

    pop_delta_pred = pop_high_p - pop_high_b
    spa_delta_pred = spa_high_p - spa_high_b
    cd_delta_pred = cd_p - cd_b

    # 5) Hellinger + ranking change
    H_pop = hellinger(pop_b, pop_p, ["high", "medium", "low"])
    H_spa = hellinger(spa_b, spa_p, ["low", "medium", "high"])
    rk_pop_changed = ranking_changed(pop_b, pop_p)
    rk_spa_changed = ranking_changed(spa_b, spa_p)

    # 6) parse which axes are mentioned in hint
    pop_hint, cat_hint, spa_hint = parse_axes_from_hint(hint)

    # ---------- popularity axis ----------

    pop_ok = True
    pop_dir_match = None
    changed_pop = (H_pop > 0.1) or rk_pop_changed

    if pop_hint:
        if pop_delta_gold is not None:
            sg = sign_with_eps(pop_delta_gold)
            sp = sign_with_eps(pop_delta_pred)
            if sg == 0:
                pop_dir_match = (sp == 0)
            else:
                pop_dir_match = (sg == sp)
            pop_ok = changed_pop and bool(pop_dir_match)
        else:
            # No gold: only require distribution shift
            pop_ok = changed_pop
    else:
        # Unhinted axes: enforce invariance
        pop_ok = (H_pop <= 0.1) and (not rk_pop_changed)

    # ---------- category axis ----------

    cat_ok = True
    cat_dir_match = None
    if cat_hint:
        changed_cat = (abs(cd_p - cd_b) > 1e-12)
        if cd_delta_gold is not None:
            sg = sign_with_eps(cd_delta_gold)
            sp = sign_with_eps(cd_delta_pred)
            if sg == 0:
                cat_dir_match = (sp == 0)
            else:
                cat_dir_match = (sg == sp)
            cat_ok = changed_cat and bool(cat_dir_match)
        else:
            cat_ok = changed_cat
    else:
        cat_ok = (abs(cd_p - cd_b) <= 1e-12)

    # ---------- spatial axis ----------

    spa_ok = True
    spa_dir_match = None
    changed_spa = (H_spa > 0.1) or rk_spa_changed

    if spa_hint:
        if spa_delta_gold is not None:
            sg = sign_with_eps(spa_delta_gold)
            sp = sign_with_eps(spa_delta_pred)
            if sg == 0:
                spa_dir_match = (sp == 0)
            else:
                spa_dir_match = (sg == sp)
            spa_ok = changed_spa and bool(spa_dir_match)
        else:
            spa_ok = changed_spa
    else:
        spa_ok = (H_spa <= 0.1) and (not rk_spa_changed)

    ok = bool(pop_ok and cat_ok and spa_ok)

    return {
        "ok": ok,
        "axes": {
            "popularity": {
                "hinted": pop_hint,
                "ok": pop_ok,
                "H": H_pop,
                "rank_changed": rk_pop_changed,
                "before": pop_b,
                "after": pop_p,
                "high_before": pop_high_b,
                "high_after": pop_high_p,
                "high_gold": pop_high_g,
                "delta_gold": pop_delta_gold,
                "delta_pred": pop_delta_pred,
                "dir_match_with_gold": pop_dir_match,
            },
            "category": {
                "hinted": cat_hint,
                "ok": cat_ok,
                "CD_before": cd_b,
                "CD_after": cd_p,
                "CD_gold": cd_g,
                "CD_delta_gold": cd_delta_gold,
                "CD_delta_pred": cd_delta_pred,
                "dir_match_with_gold": cat_dir_match,
            },
            "spatial": {
                "hinted": spa_hint,
                "ok": spa_ok,
                "H": H_spa,
                "rank_changed": rk_spa_changed,
                "before": spa_b,
                "after": spa_p,
                "high_before": spa_high_b,
                "high_after": spa_high_p,
                "high_gold": spa_high_g,
                "delta_gold": spa_delta_gold,
                "delta_pred": spa_delta_pred,
                "dir_match_with_gold": spa_dir_match,
            },
        },
        "meta": {
            "op_pred": op_pred,
            "idx_pred": idx_pred,
            "low_km": low_km,
            "high_km": high_km,
        },
    }

import random
import re, ast
import numpy as np
import json
from math import radians, sin, cos, sqrt, atan2
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key = "YOUR_API_KEY"
import pandas as pd
import csv
from itertools import groupby
import datetime
import sys
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Dataset_Pipline import position_POI_extraction
from Dataset_Pipline.template import functions, prompts
from urllib.parse import unquote_plus
import time, httpx
from openai import OpenAI, NotFoundError, APIStatusError, RateLimitError, APIConnectionError
from benchmark.api_key import api_key


deepseek_api_key = "YOUR API KEY"

client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=deepseek_api_key
    )


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

def make_ds_client():
    return OpenAI(
        base_url="https://api.deepseek.com",
        api_key=deepseek_api_key
    )


def ds_chat_create(payload: dict):
    """
    Robust chat.completions.create:
      - build a fresh client per request
      - 404 -> rebuild client then retry
      - 5xx/429/APIConnectionError -> exponential backoff + retry
    """

    client = make_ds_client()
    return client.chat.completions.create(**payload)



def log_payload_brief(payload, round_idx=None):
    try:
        print(f"[dbg] round={round_idx} model={payload.get('model')} "
              f"msgs={len(payload.get('messages', []))} "
              f"tools={len(payload.get('tools', []) if payload.get('tools') else 0)} "
              f"stop={payload.get('stop')}")
    except Exception:
        pass


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openaiAPIcall(**kwargs):
    return client.chat.completions.create(**kwargs)


def unix_time_convert(unix_timestamp):
    datatime_object = datetime.datetime.fromtimestamp(unix_timestamp, tz=ZoneInfo("Australia/Melbourne"))
    formatted_time = datatime_object.strftime('%H:%M')

    return formatted_time


def new_data_open(city_name):
    max_len = -1
    if city_name in ['Melb', 'Toro']:
        base_dir = REPO_ROOT / "data4perturb" / city_name
        train_data = pd.read_csv(base_dir / "train.csv")
        test_data = pd.read_csv(base_dir / "test.csv")
        val_data = pd.read_csv(base_dir / "val.csv")

        data = pd.concat([train_data, test_data])
        data = pd.concat([data, val_data])
        data = data.reset_index(drop=True)

        return data

    if city_name == 'Florence':
        # WARNING: MUST ENCODE it via "utf-8"
        base_path = REPO_ROOT / "data4perturb" / "Florence"
        data = pd.read_csv(base_path / f"Trajectories-{city_name.upper()}.csv",
                           encoding='utf-8')
        poi_info = pd.read_csv(base_path / f"PoIs-{city_name.upper()}.csv",
                               encoding='utf-8')
        # cat_info = pd.read_csv(f"{base_path}/Categories-{city_name.upper()}.csv")

        max_length = 21

        return data, poi_info


def remove_consecutive_duplicates(lst):
    """Remove consecutive duplicate elements from list"""
    return [key for key, _ in groupby(lst)]


def haversine_distance(lat1, lon1, lat2, lon2):
    # lat1 = eval(lat1)
    # lon1 = eval(lon1)
    # lat2 = eval(lat2)
    # lon2 = eval(lon2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return distance


def geo_info_collect(city_name):
    if city_name != 'Florence':
        if city_name in ['Buda', 'Delh', 'Edin', 'Glas', 'Osak', 'Pert', 'Toro', 'Vien']:
            path = REPO_ROOT / "og_dataset" / "data-ijcai15" / "poiList-ijcai15" / f"POI-{city_name}.csv"
        elif city_name == 'Melb':
            path = REPO_ROOT / "og_dataset" / "data-cikm16" / f"POI-{city_name}.csv"
        poi_info = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            count = 0
            for row in reader:
                row = row[0].split(';')
                if count == 0:
                    columns = row
                else:
                    row[0] = eval(row[0])
                    poi_info.append(row)
                count += 1
        poi_info = pd.DataFrame(poi_info, columns=columns)
        poi_info['poiName'] = poi_info['poiName'].astype(str).apply(unquote_plus)
        poi_info['poiID'] -= 1
        dis_mat = np.zeros([len(poi_info), len(poi_info)])
        for i in range(len(poi_info)):
            for j in range(len(poi_info)):
                if i < j:
                    cur_lat = eval(poi_info.loc[i, 'lat'])
                    cur_long = eval(poi_info.loc[i, 'long'])

                    next_lat = eval(poi_info.loc[j, 'lat'])
                    next_long = eval(poi_info.loc[j, 'long'])

                    dis_mat[i, j] = haversine_distance(lat1=cur_lat,
                                                       lat2=next_lat,
                                                       lon1=cur_long,
                                                       lon2=next_long)

                    dis_mat[j, i] = dis_mat[i, j]

        return dis_mat, poi_info


def spatial_distance_classification(traj, dis_mat):
    traj_set = traj['seqID'].unique()
    poi_set = traj['poiID'].unique()  # Double check it is ranging from 0-?
    pair_list = []
    for traj_id in traj_set:
        cur_traj = traj[traj['seqID'] == traj_id].sort_values(by='startTime', ascending=True)
        cur_traj = cur_traj['poiID'].values.tolist()

        for i in range(len(cur_traj) - 1):
            pair_list.append([cur_traj[i], cur_traj[i + 1]])

    seen = set()
    unique_pairs = []
    for p in pair_list:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_pairs.append(p)

    distance_list = []
    for p in unique_pairs:
        distance_list.append(dis_mat[p[0]][p[1]])
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)
    d1 = np.quantile(distance_list, 1 / 3)
    d2 = np.quantile(distance_list, 2 / 3)

    return d1, d2


def candidate_generation(traj, poi_info, og_traj, cat_mark=None):
    candidate_pool = list(set(traj['poiID'].unique()) - set(og_traj))
    candidate_desc = []
    for i in candidate_pool:
        cur_poi_info = poi_info[poi_info['poiID'] == i]
        poplevel = traj[traj['poiID'] == i]['popLevel'].unique()
        visit_behavior = [cur_poi_info['poiName'].item(), cur_poi_info[cat_mark].item(),
                          cur_poi_info['long'].item(),
                          cur_poi_info['lat'].item(), poplevel[0]]

        candidate_desc.append(visit_behavior)
    return candidate_desc


def _require_keys(args, required):
    missing = [k for k in required if k not in args or args[k] is None]
    if missing:
        return {
            "error": "missing_arguments",
            "message": "Required keys are missing.",
            "missing": missing,
            "required": required,
            "received": list(args.keys())
        }
    return None


# --- geo_distance_segments (as above) ---
def _maybe_json(v):
    """If v is str, try json.loads; otherwise return as-is. On failure, return as-is."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _as_waypoints(v):
    """
    Expected input: [{"lat": <num>, "lon": <num>}, ...]
    Tolerances:
      - If entire list is stringified: json.loads once
      - If each element is stringified: json.loads per element
    Rules:
      - Always return Python list (no set/dict)
      - Each element must be {"lat": float, "lon": float}
    On failure: return None
    """
    v = _maybe_json(v)
    if not isinstance(v, list):
        return None

    out = []
    for i, d in enumerate(v):
        d = _maybe_json(d)
        if not isinstance(d, dict):
            return None
        if "lat" not in d or "lon" not in d:
            return None
        try:
            lat = float(d["lat"])
            lon = float(d["lon"])
        except Exception:
            return None
        out.append({"lat": lat, "lon": lon})
    return out  # ensure list


def _haversine_km(a, b):
    R = 6371.0088
    φ1, φ2 = math.radians(a["lat"]), math.radians(b["lat"])
    dφ = math.radians(b["lat"] - a["lat"])
    dλ = math.radians(b["lon"] - a["lon"])
    x = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    x = min(1.0, max(0.0, x))
    return 2 * R * math.asin(math.sqrt(x))


def _spatial_class(L_km):
    """
    Value are based on "dis_mark".
    """
    # if L_km < 0.91:  return "Low"
    # if L_km <= 1.74: return "Medium"
    if L_km < 0.2: return "Low"
    if L_km <= 0.45: return "Medium"
    return "High"


def tool_geo_distance_segments(args):
    # 1) parse top-level
    if not isinstance(args, dict):
        try:
            args = json.loads(args or "{}")
        except Exception:
            args = {}

    # 2) required fields
    required = ["waypoints_before", "waypoints_after"]
    missing = [k for k in required if k not in args or args[k] is None]
    if missing:
        return {
            "error": "missing_arguments",
            "message": "Required keys missing",
            "missing": missing,
            "required": required,
            "received": list(args.keys())
        }

    # 3) normalize to list[dict]
    wb = _as_waypoints(args["waypoints_before"])
    wa = _as_waypoints(args["waypoints_after"])
    if wb is None or wa is None:
        return {
            "error": "invalid_arguments",
            "message": "waypoints_* must be a JSON array (or Python list) of objects with numeric {lat, lon}. Do NOT stringify elements.",
            "received_types": {
                "waypoints_before": type(args["waypoints_before"]).__name__,
                "waypoints_after": type(args["waypoints_after"]).__name__
            }
        }

    # 4) strict type/structure check (avoid set/dict)
    if not isinstance(wb, list) or not isinstance(wa, list):
        return {"error": "invalid_arguments", "message": "waypoints_* must be lists"}
    for prefix, wps in (("before", wb), ("after", wa)):
        for idx, p in enumerate(wps):
            if not isinstance(p, dict):
                return {"error": "invalid_arguments", "message": f"Waypoint {prefix}[{idx}] must be object"}
            if not all(k in p for k in ("lat", "lon")):
                return {"error": "invalid_arguments", "message": f"Waypoint {prefix}[{idx}] missing lat/lon"}
            if not isinstance(p["lat"], (int, float)) or not isinstance(p["lon"], (int, float)):
                return {"error": "invalid_arguments", "message": f"Waypoint {prefix}[{idx}] lat/lon must be numeric"}

    # 5) compute segments
    def segments(wps):
        if len(wps) < 2:
            return [], []
        dists, cats = [], []
        for i in range(len(wps) - 1):
            d = round(_haversine_km(wps[i], wps[i + 1]), 6)
            dists.append(float(d))
            cats.append(_spatial_class(d))
        return dists, cats

    db, cb = segments(wb)
    da, ca = segments(wa)

    # 6) segment count must equal len(points)-1
    if len(db) != max(0, len(wb) - 1) or len(da) != max(0, len(wa) - 1):
        return {"error": "inconsistent_result", "message": "segment count must equal len(waypoints)-1"}

    return {
        "before": {"distances_km": db, "classes": cb},
        "after": {"distances_km": da, "classes": ca}
    }


# ---------------------------Categorical Diversity---------------------------
def categories_from_itinerary(args):
    import json
    # parse
    if not isinstance(args, dict):
        try:
            args = json.loads(args or "{}")
        except Exception:
            args = {}

    # required: itinerary (2D list)
    if "itinerary" not in args or args["itinerary"] is None:
        return {
            "error": "missing_arguments",
            "message": "categories_from_itinerary requires 'itinerary' list.",
            "required": ["itinerary"],
            "received": list(args.keys())
        }

    it = args["itinerary"]
    if not isinstance(it, list):
        return {"error": "invalid_arguments", "message": "itinerary must be a list of rows."}

    cats = []
    for idx, row in enumerate(it):
        if not (isinstance(row, list) and len(row) >= 2):
            return {"error": "invalid_arguments",
                    "message": f"row {idx} must be a list with at least 2 elements [name, category, ...]."}
        cat = row[1]
        if not isinstance(cat, str):
            return {"error": "invalid_arguments",
                    "message": f"row {idx} category must be a string; got {type(cat).__name__}."}
        cats.append(cat)

    return {
        "itinerary_len": len(it),
        "categories": cats
    }


def _norm_cat(s): return " ".join(str(s).strip().split()).title()


def tool_cd_from_categories(args):
    if not isinstance(args, dict):
        try:
            args = json.loads(args or "{}")
        except Exception:
            args = {}

    # err = _require_keys(args, ["categories_before","categories_after"])
    # if err: return err
    required = ["categories_before", "categories_after"]
    missing = [k for k in required if k not in args or args[k] is None]
    if missing:
        return {
            "error": "missing_arguments",
            "message": "cd_from_categories requires BOTH arrays.",
            "missing": missing,
            "required": required,
            "received": list(args.keys())
        }

    cb = args["categories_before"];
    ca = args["categories_after"]
    if not isinstance(cb, list) or not isinstance(ca, list):
        return {"error": "invalid_arguments", "message": "categories_* must be lists of strings."}

    def cd_and_set(lst):
        n = len(lst)
        uniq = {_norm_cat(x) for x in lst}
        k = len(uniq)
        cd = 0.0 if n == 0 else (0.0 if k == 1 else k / n)
        return round(float(cd), 12), sorted(uniq)

    cd_b, set_b = cd_and_set(cb)
    cd_a, set_a = cd_and_set(ca)
    return {
        "categories_raw_before": list(cb),
        "categories_set_before": set_b,
        "categories_raw_after": list(ca),
        "categories_set_after": set_a,
        "cd_before": cd_b,
        "cd_after": cd_a,
        "cd_disruption": abs(cd_b - cd_a) > 1e-12
    }


# --------------------------------Tau Hellinger--------------------------------
def tool_stats_from_categories(args):
    """
    Compute distribution/Hellinger/τ-b for categorical labels (before vs after).

    Key points:
    - Case-insensitive mapping of labels_before/after to domain; normalize to the canonical form in domain.
    - If any label is not in domain (case-insensitive), return {"error":"invalid_arguments", ...}.
    - Preserve return structure and field semantics.
    """
    import json, math
    from collections import Counter

    # -------- parse and required checks --------
    if not isinstance(args, dict):
        try:
            args = json.loads(args or "{}")
        except Exception:
            args = {}

    required = ["labels_before", "labels_after", "domain"]
    missing = [k for k in required if k not in args or args[k] is None]
    if missing:
        return {
            "error": "missing_arguments",
            "message": "stats_from_categories requires labels_before, labels_after, domain.",
            "missing": missing,
            "required": required,
            "received": list(args.keys())
        }

    lb, la, domain = args["labels_before"], args["labels_after"], args["domain"]
    if not (isinstance(lb, list) and isinstance(la, list) and isinstance(domain, list)):
        return {"error": "invalid_arguments", "message": "labels_* and domain must be lists."}

    # -------- build case-insensitive domain map and validate --------
    def _norm_token(x):
        return str(x).strip().lower()

    if any(str(d).strip() == "" for d in domain):
        return {"error": "invalid_arguments", "message": "domain contains empty token."}

    dom_map = {}  # normalized key -> canonical token from domain
    for tok in domain:
        key = _norm_token(tok)
        # duplicate token (case-insensitive) means domain conflict
        if key in dom_map and dom_map[key] != str(tok):
            return {
                "error": "invalid_arguments",
                "message": "domain contains duplicate values ignoring case.",
                "duplicate_key": key,
                "domain": domain
            }
        dom_map[key] = str(tok)

    # -------- normalize labels_* to canonical domain tokens --------
    def _canonize(seq, name):
        out = []
        for i, x in enumerate(seq):
            k = _norm_token(x)
            if k not in dom_map:
                return {
                    "error": "invalid_arguments",
                    "message": f"{name} contains value not in domain (case-insensitive).",
                    "index": i,
                    "label": x,
                    "allowed": domain
                }
            out.append(dom_map[k])
        return out

    tmp = _canonize(lb, "labels_before")
    if isinstance(tmp, dict) and "error" in tmp:
        return tmp
    lb = tmp

    tmp = _canonize(la, "labels_after")
    if isinstance(tmp, dict) and "error" in tmp:
        return tmp
    la = tmp

    # -------- counts in domain order --------
    cb = {k: int(Counter(lb).get(k, 0)) for k in domain}
    ca = {k: int(Counter(la).get(k, 0)) for k in domain}

    # distribution
    sb, sa = sum(cb.values()), sum(ca.values())
    pb = {k: (0.0 if sb == 0 else round(cb[k] / sb, 12)) for k in domain}
    pa = {k: (0.0 if sa == 0 else round(ca[k] / sa, 12)) for k in domain}

    # competition ranks (ties share rank)
    def comp_ranks(counts):
        items = sorted(((k, counts[k]) for k in domain), key=lambda x: (-x[1], x[0]))
        ranks = {};
        rank = 0;
        prev = None;
        seen = 0
        for k, c in items:
            seen += 1
            if c != prev:
                rank = seen;
                prev = c
            ranks[k] = rank
        return ranks

    rb, ra = comp_ranks(cb), comp_ranks(ca)

    # Hellinger
    H = math.sqrt(sum((math.sqrt(pb[k]) - math.sqrt(pa[k])) ** 2 for k in domain)) / math.sqrt(2.0)
    H = float(round(H, 12))

    # -------- Kendall's tau related calculations --------
    C = D = T_x = T_y = 0
    n = len(domain)
    for i in range(n):
        xi, yi = cb[domain[i]], ca[domain[i]]
        for j in range(i + 1, n):
            xj, yj = cb[domain[j]], ca[domain[j]]
            dx = (xi > xj) - (xi < xj)
            dy = (yi > yj) - (yi < yj)
            if dx == 0 and dy == 0:
                T_x += 1;
                T_y += 1
            elif dx == 0:
                T_x += 1
            elif dy == 0:
                T_y += 1
            elif dx == dy:
                C += 1
            else:
                D += 1

    denom_std = math.sqrt((C + D + T_x) * (C + D + T_y))
    tau_b_std = 0.0 if denom_std == 0 else float(round((C - D) / denom_std, 12))

    # Goodman–Kruskal gamma (non-tied pairs only)
    gamma = None
    if (C + D) > 0:
        gamma = float(round((C - D) / (C + D), 12))

    # pairwise sign agreement (including ties)
    def pairwise_signs(counts):
        signs = {}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = counts[domain[i]], counts[domain[j]]
                s = (a > b) - (a < b)
                signs[(i, j)] = s
        return signs

    s_before = pairwise_signs(cb)
    s_after = pairwise_signs(ca)
    total_pairs = len(s_before)  # n*(n-1)/2
    matches = sum(1 for p in s_before if s_before[p] == s_after[p])
    tau_pairwise = 1.0 if total_pairs == 0 else float(round(matches / total_pairs, 12))
    all_pairs_match = (matches == total_pairs)

    # tau policy (kept same as original; overridable via args["tau_policy"])
    tau_policy = str(args.get("tau_policy", "b_sign_override")).lower()
    if tau_policy == "b":
        tau_val = tau_b_std
    elif tau_policy == "b_sign_override":
        tau_val = 1.0 if all_pairs_match else tau_b_std
    elif tau_policy == "gamma":
        tau_val = 0.0 if gamma is None else gamma
    else:  # "pairwise_sign"
        tau_val = tau_pairwise

    # disruption decision (tolerant thresholds)
    thr = args.get("thresholds", {}) or {}
    try:
        hell_thr = float(thr.get("hellinger", 0.1))
    except Exception:
        hell_thr = 0.1
    try:
        tau_thr = float(thr.get("tau_b", 1.0))
    except Exception:
        tau_thr = 1.0

    disruption = (H > hell_thr) or (tau_val < tau_thr)

    return {
        "domain": domain,
        "counts_before": cb, "counts_after": ca,
        "distribution_before": pb, "distribution_after": pa,
        "ranks_before": rb, "ranks_after": ra,
        "hellinger": H,
        "tau_b": tau_val,
        "tau_b_std": tau_b_std,
        "gamma": gamma,
        "pairwise_signs_match": matches,
        "pairwise_signs_total": total_pairs,
        "disruption": disruption
    }


def handle_tool_call(call):
    name = call.function.name
    args = call.function.arguments
    print("-------------------Args-------------------")
    print(f"Args:{args}")

    try:
        if name == "geo_distance_segments":
            print("-------------------Computing Spatial Distance-------------------")
            print(f"Result:{tool_geo_distance_segments(args)}")
            return tool_geo_distance_segments(args)
        elif name == "stats_from_categories":
            print("-------------------Computing H and Tau-b-------------------")
            print(f"Result:{tool_stats_from_categories(args)}")
            return tool_stats_from_categories(args)
        elif name == "cd_from_categories":
            print("-------------------Computing CD-------------------")
            print(f"Result:{tool_cd_from_categories(args)}")
            return tool_cd_from_categories(args)
        elif name == "categories_from_itinerary":
            print("-------------------Extracting Category-------------------")
            print(f"Result:{categories_from_itinerary(args)}")
            return categories_from_itinerary(args)

    except Exception as e:
        return {"error": "tool_runtime_error", "tool": name, "message": str(e), "received": list(args.keys())}


# ---------- Multi-round tool-calling loop (supports frequent calls) ----------
def run_with_tools(model, messages, tools, max_tool_rounds=800, stop_tokens=["</final_json>"], temperature=0):
    """
    Returns: content, usage_summary
    usage_summary structure:
    {
      "totals": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
      "rounds": [
        {
          "round": int,
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "finish_reason": str|None,
          "tools_called": [str, ...],
          "prompt_tokens_details": {...},         # if SDK provides
          "completion_tokens_details": {...}      # if SDK provides
        },
        ...
      ],
      "by_tool": {"tool_name": count, ...},
      "finish_reasons": {"stop": n, "tool_calls": m, ...}
    }
    """
    usage_summary = {
        "totals": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "rounds": [],
        "by_tool": {},
        "finish_reasons": {}
    }

    for round_idx in range(1, max_tool_rounds + 1):

        # If this round will not use tools, omit the tools field (better compatibility)
        tools_payload = tools if (tools and isinstance(tools, list) and len(tools) > 0) else None

        payload = {
            "model": model,
            "messages": messages,
            "tool_choice": "auto",
            "stop": stop_tokens,
            "temperature": temperature,
            # "max_tokens": 16000,
        }

        if tools_payload is not None:
            payload["tools"] = tools_payload

        # log_payload_brief(payload, round_idx=None)  # optional debug

        resp = ds_chat_create(payload)  # robust request wrapper
        # print(resp)
        # assert 1 == 0

        # Collect and normalize usage details
        usage = getattr(resp, "usage", None)
        u = {}
        if usage is not None:
            try:
                u = usage.to_dict()  # OpenAI/compatible SDK
            except Exception:
                if isinstance(usage, dict):
                    u = usage
                else:
                    u = {}

        # Normalize different field names
        prompt_tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        completion_tokens = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        total_tokens = int(u.get("total_tokens") or (prompt_tokens + completion_tokens))

        # Optional details (if SDK provides)
        prompt_details = u.get("prompt_tokens_details") or {}
        completion_details = u.get("completion_tokens_details") or {}

        choice = resp.choices[0]
        msg = choice.message
        finish_reason = getattr(choice, "finish_reason", None)
        tcs = getattr(msg, "tool_calls", None) or []
        tool_names = []
        for tc in tcs:
            try:
                tool_names.append(tc.function.name)
            except Exception:
                tool_names.append("UNKNOWN_TOOL")

        # Per-round log
        usage_summary["rounds"].append({
            "round": round_idx,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "tools_called": tool_names,
            "prompt_tokens_details": prompt_details,
            "completion_tokens_details": completion_details,
        })

        # Accumulate totals
        usage_summary["totals"]["prompt_tokens"] += prompt_tokens
        usage_summary["totals"]["completion_tokens"] += completion_tokens
        usage_summary["totals"]["total_tokens"] += total_tokens

        usage_summary["finish_reasons"][finish_reason] = usage_summary["finish_reasons"].get(finish_reason, 0) + 1
        for name in tool_names:
            usage_summary["by_tool"][name] = usage_summary["by_tool"].get(name, 0) + 1
        # ==============================================

        # Main flow
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            messages.append(msg)
            for call in tcs:
                result = handle_tool_call(call)
                messages.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": call.function.name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            continue

        # normal return: content + usage
        return msg.content, usage_summary

    raise RuntimeError("Exceeded max tool-calling rounds")


def extract_json(text: str):
    """
    Robust JSON extraction:
    1) strip <think>…</think>
    2) after <final_json> do quote-aware brace balancing
    3) if still missing, scan triple-quoted blocks then full text
    4) return dict (prefer containing key signals) or raise
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty model output.")

    PREFERRED_KEYS = {
        "Perturbed Itinerary", "Intents", "cd_before", "cd_after", "cd_disruption",
        "popularity_disruption", "spatial_disruption",
        "popularity_distribution_before", "spatial_distances_before",
        "spatial_categories_before", "popularity_tau_b", "spatial_tau_b"
    }
    ZWS = "\ufeff\u200b\u200c\u200d\u2060"

    def _normalize(x):
        if isinstance(x, set):    return list(x)
        if isinstance(x, tuple):  return [_normalize(v) for v in x]
        if isinstance(x, list):   return [_normalize(v) for v in x]
        if isinstance(x, dict):   return {k: _normalize(v) for k, v in x.items()}
        return x

    def _try_parse(s: str):
        s = s.lstrip(ZWS).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        py = re.sub(r'\btrue\b', 'True', s)
        py = re.sub(r'\bfalse\b', 'False', py)
        py = re.sub(r'\bnull\b', 'None', py)
        try:
            obj = ast.literal_eval(py)
        except Exception:
            return None
        return _normalize(obj)

    def _balance_from(t: str, start: int):
        """Quote-aware balancing: handle quotes and escapes."""
        if start < 0 or start >= len(t) or t[start] not in "{[":
            return None
        stack, in_str, esc = [], False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if not stack:
                        continue
                    top = stack[-1]
                    if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                        stack.pop()
                        if not stack:
                            return t[start:i + 1]
        return None

    def _score(d: dict):
        hits = sum(1 for k in d.keys() if k in PREFERRED_KEYS)
        return (1 if hits > 0 else 0, hits, -len(d))  # prefer key hits, then count, then shorter length

    # drop <think> section first
    s = re.sub(r"(?is)<think>.*?</think>", "", text)

    # 1) try after <final_json>
    if "<final_json>" in s:
        tail = s.split("<final_json>", 1)[1]
        start = tail.find("{")
        if start != -1:
            clip = _balance_from(tail, start)
            if clip:
                obj = _try_parse(clip)
                if isinstance(obj, dict):
                    return obj
                # not a dict: keep searching

    best = None  # (score_tuple, obj)

    # 2) scan triple-quoted code blocks
    for m in re.finditer(r"```[\w-]*\s*([\s\S]*?)\s*```", s, re.IGNORECASE):
        block = m.group(1)
        starts = []
        p1, p2 = block.find("{"), block.find("[")
        if p1 != -1: starts.append(p1)
        if p2 != -1: starts.append(p2)
        for k in sorted(starts, key=lambda x: (block[x] != "{", x)):  # prioritize '{'
            clip = _balance_from(block, k)
            if not clip:
                continue
            obj = _try_parse(clip)
            if isinstance(obj, dict):
                sc = _score(obj)
                if best is None or sc > best[0]:
                    best = (sc, obj)
                    if sc[0] == 1:  # key hit found
                        return obj

    # 3) fallback: scan full text (up to 300 starts)
    tried = 0
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        if ch == "[":
            nxt = s.find("{", i)
            if nxt != -1 and nxt - i < 1000:
                continue
        clip = _balance_from(s, i)
        if not clip:
            continue
        obj = _try_parse(clip)
        if isinstance(obj, dict):
            sc = _score(obj)
            if best is None or sc > best[0]:
                best = (sc, obj)
                if sc[0] == 1:
                    return obj
        tried += 1
        if tried >= 300:
            break

    if best is not None:
        return best[1]

    head = repr(s[:200]);
    tail = repr(s[-200:])
    raise ValueError("No JSON object found (missing or truncated after <final_json>). "
                     f"Head: {head}\nTail: {tail}")


def iTIMO_perturbation(city_name, op=None, checkpoint=-1):
    if not op:
        print("Error")
        return "Error"

    if op == 'ADD':
        used_mark = 'inserted'
    elif op == 'REPLACE':
        used_mark = 'after'
    elif op == 'DELETE':
        used_mark = 'deleted'

    if city_name in ['Melb', 'Toro']:
        if city_name == 'Melb':
            # Path Initialization
            memory_path = f"data-cikm16/perturbation_data/{op}/memory_buffer_{op}.json"
            token_usage_path = f"data-cikm16/perturbation_data/{op}/token_usage_{op}.json"
            result_path = f"data-cikm16/perturbation_data/Melbourne_{op}.json"
            infer_time_path = f"data-cikm16/perturbation_data/{op}/infer_time_{op}.json"
            cat_mark = 'subTheme'
            dis_mark = {'low': 0.3, 'medium': 0.77}
        elif city_name == 'Toro':
            memory_path = f"data-ijcai15/Toro/perturbation_data/{op}/memory_buffer_{op}.json"
            token_usage_path = f"data-ijcai15/Toro/perturbation_data/{op}/token_usage_{op}.json"
            result_path = f"data-ijcai15/Toro/perturbation_data/Toronto_{op}.json"
            infer_time_path = f"data-ijcai15/Toro/perturbation_data/{op}/infer_time_{op}.json"
            cat_mark = 'theme'
            dis_mark = {'low': 0.91, 'medium': 1.74}
            # _spatial_class Function

        traj = new_data_open(city_name)
        traj['poiID'] -= 1
        dis_mat, poi_info = geo_info_collect(city_name)

        '''Data Preprocessing'''
        pop = traj['poiFreq'].unique()
        q1 = np.quantile(pop, 1 / 3)
        q2 = np.quantile(pop, 2 / 3)

        for j in range(len(traj)):
            # print(traj.loc[j, 'dateTaken'], type(traj.loc[j, 'dateTaken']))
            traj.loc[j, 'startTime'] = eval(traj.loc[j, 'dateTaken'])[0]
            traj.loc[j, '#photo'] = len(eval(traj.loc[j, 'dateTaken']))
            if traj.loc[j, 'poiFreq'] <= q1:
                traj.loc[j, 'popLevel'] = 'low'
            elif traj.loc[j, 'poiFreq'] >= q2:
                traj.loc[j, 'popLevel'] = 'high'
            else:
                traj.loc[j, 'popLevel'] = 'medium'
    else:
        if city_name == 'Florence':
            memory_path = f"LearNext-DATASET/Florence/perturbation_data/{op}/memory_buffer_{op}.json"
            token_usage_path = f"LearNext-DATASET/Florence/perturbation_data/{op}/token_usage_{op}.json"
            result_path = f"LearNext-DATASET/Florence/perturbation_data/Florence_{op}.json"
            infer_time_path = f"LearNext-DATASET/Florence/perturbation_data/{op}/infer_time_{op}.json"
            cat_mark = 'theme'
            dis_mark = {'low': 0.20, 'medium': 0.45}
            # _spatial_class Function

        traj, poi_info = new_data_open(city_name)
        poi_info = poi_info.rename(columns={'PoIName_Italian': 'poiName'})
        traj['poiID'] -= 1
        poi_info['poiID'] -= 1
        '''Data Preprocessing'''
        pop = traj['poiFreq'].unique()
        q1 = np.quantile(pop, 1 / 3)
        q2 = np.quantile(pop, 2 / 3)

        for j in range(len(traj)):
            if traj.loc[j, 'poiFreq'] <= q1:
                traj.loc[j, 'popLevel'] = 'low'
            elif traj.loc[j, 'poiFreq'] >= q2:
                traj.loc[j, 'popLevel'] = 'high'
            else:
                traj.loc[j, 'popLevel'] = 'medium'

    # d1, d2 = spatial_distance_classification(traj, dis_mat)
    # print(d1, d2)
    traj_set = traj['seqID'].unique()
    traj_set = traj_set.tolist()

    target_intent_count = np.random.default_rng(2025).integers(1, 4, size=len(traj_set))

    # random.seed(2025)
    # traj_set = random.sample(traj_set, 50)
    # print(traj_set)
    # if city_name == 'Toro':
    #     '''seed 2025-Toronto'''
    #     traj_set = [3881, 807, 5530, 5640, 1378, 624, 58, 4132, 4198, 5073, 1938,
    #                 744, 4308, 4493, 852, 634, 923, 1865, 4756, 683, 1135, 453, 2686,
    #                 5959, 745, 1592, 1525, 380, 1934, 5214, 4451, 5324, 3898, 71, 472,
    #                 623, 898, 2198, 1620, 1802, 929, 2036, 904, 515, 920, 1066, 2604,
    #                 4004, 813, 99]
    # elif city_name == 'Melb':
    #     '''seed 2025-Melbourne'''
    #     traj_set = [3454, 605, 1043, 3185, 1021, 3334, 113, 2433, 2477, 3629, 1468, 574, 2586, 2700,
    #                 660, 382, 757, 3184,
    #                 1288, 2811, 472, 3950, 298, 3447, 3256, 575, 3866, 1229, 3588, 1136, 295, 45,
    #                 1397, 3935, 779, 2691,
    #                 3046, 2321, 3282, 329, 3799, 712, 3426, 1235, 1278, 798, 1578, 3865, 336, 731]
    # elif city_name == 'Florence':
    #     '''seed 2025-Florence'''
    #     traj_set = ['3332', '481', '3', '5125', '6', '5759', '997',
    #                 '3133', '1', '2206', '5924', '2229', '3409', '1363',
    #                 '395', '2267', '2377', '582', '248', '16', '4662',
    #                 '1229', '5974', '591', '2444', '291', '3688', '154',
    #                 '3309', '17', '400', '10', '5500', '1120',
    #                 '4674', '15', '147', '4276', '1332', '5968',
    #                 '4763', '5719', '4733', '14', '2350', '5619',
    #                 '2734', '2112', '3083', '5562']
    #
    # target_intent_count = [3, 2, 3, 3, 2, 3, 1, 3, 1, 3, 1, 3, 2, 1, 1,
    #                        3, 1, 3, 1, 3, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1,
    #                        2, 1, 1, 2, 3, 1, 2,
    #                        1, 3, 2, 1, 2, 3, 1, 1, 2, 2, 2, 2, 1]

    mark = False
    result = {}
    infer_time = {}
    token_usage = {}
    used_poi = []
    used_pos = []
    tools = functions.tools

    if checkpoint != -1:
        with open(result_path, encoding='utf-8') as f:
            result = json.load(f)
        with open(memory_path, encoding='utf-8') as f:
            memory_dic = json.load(f)
            used_pos = memory_dic['POSITION']
            used_poi = memory_dic['POI']
        with open(token_usage_path, encoding='utf-8') as f:
            token_usage = json.load(f)
        with open(infer_time_path, encoding='utf-8') as f:
            infer_time = json.load(f)

        used_traj = []

        for key, value in result.items():
            used_traj.append(key)

        print(len(traj_set))
        for x in range(len(traj_set)):
            traj_set[x] = str(traj_set[x])
        traj_set = list(set(traj_set) - set(used_traj))
        print(traj_set, "\n",used_traj)
        print(len(traj_set))
        # assert 1 == 0


    for current_round, traj_id in enumerate(traj_set):
        if city_name in ['Melb', 'Toro']:
            traj_id = int(traj_id)
        '''Initialize Memory'''
        if current_round == -1:
            memory = prompts.memory_prompt.format(current_round, op, op, None, None)
        else:
            memory = prompts.memory_prompt.format(current_round, op, op, used_poi, used_pos)

        # if breakpoint == -1:
        #     mark = True
        # elif traj_id == breakpoint:
        mark = True

        if mark:
            cur_traj = traj[traj['seqID'] == traj_id].sort_values(by='startTime', ascending=True)
            if "_" in str(traj_id) and len(cur_traj) < 21:
                pretend_length = 21 - len(cur_traj)
                pretend_chunk_id = cur_traj['chunk_index'].unique()[0] - 1
                pretend_traj_id = f"{cur_traj['seqID_old'].unique()[0]}_{pretend_chunk_id}"
                pretend_traj = traj[traj['seqID'] == pretend_traj_id].sort_values(by='startTime',
                                                                                  ascending=True).reset_index(drop=True)
                pretend_traj = pretend_traj.tail(pretend_length)
                cur_traj = pd.concat([pretend_traj, cur_traj]).reset_index(drop=True)

            cur_traj = cur_traj['poiID'].values.tolist()

            og_traj_desc = []
            for count, i in enumerate(cur_traj):
                cur_poi_info = poi_info[poi_info['poiID'] == i]
                poplevel = traj[traj['poiID'] == i]['popLevel'].unique()
                visit_behavior = [cur_poi_info['poiName'].item(), cur_poi_info[cat_mark].item(),
                                  cur_poi_info['long'].item(),
                                  cur_poi_info['lat'].item(), poplevel[0]]
                og_traj_desc.append(visit_behavior)

            if op != 'DELETE':
                # print(poi_info)
                candidate_desc = candidate_generation(traj, poi_info, cur_traj, cat_mark=cat_mark)
                if city_name != 'Toro':
                    candidate_desc = random.sample(candidate_desc, 50)

                input_prompt = {'Original Itinerary': og_traj_desc,
                                'Target Intent Count': target_intent_count[current_round],
                                'Memory': memory,
                                'Candidate Intents': ["Spatial distance disruption", "Popularity disruption",
                                                      "Categorical diversity disruption"],
                                'Candidate POIs': candidate_desc}
            else:
                input_prompt = {'Original Itinerary': og_traj_desc,
                                'Target Intent Count': target_intent_count[current_round],
                                'Memory': memory,
                                'Candidate Intents': ["Spatial distance disruption", "Popularity disruption",
                                                      "Categorical diversity disruption"]
                                }

            if op == 'REPLACE':
                system_prompt = prompts.replace_prompt_fm
            elif op == 'ADD':
                system_prompt = prompts.add_prompt_fm
            elif op == 'DELETE':
                system_prompt = prompts.delete_prompt_fm
            messages = [{'role': 'system', 'content': str(system_prompt)},
                        {'role': 'user', 'content': str(input_prompt)}]

            print(f'TRAJ ID: {traj_id}')
            print(f"Target Intent Count: {target_intent_count[current_round]}")
            print(og_traj_desc)
            # print(input_prompt)
            start_time = time.time()
            response, one_token_usage = run_with_tools(model='deepseek-chat',
                                                       messages=messages,
                                                       tools=tools, stop_tokens=["</final_json>"], temperature=0)

            end_time = time.time()
            infer_time[traj_id] = end_time - start_time
            token_usage[traj_id] = one_token_usage
            print(f"Infer Time Cost:{end_time - start_time}")
            print(f"Token Cost:{one_token_usage}")
            print(response)
            json_response = extract_json(response)
            change = position_POI_extraction.extract_change(original=og_traj_desc,
                                                            perturbed=json_response['Perturbed Itinerary'],
                                                            operation=op)
            # update MEMORY
            failure_traj = []
            if change['ok']:
                if len(used_poi) < 50:
                    used_poi.append(change[used_mark][0])
                    used_pos.append(change['index'])
                else:
                    used_poi = deque(used_poi)
                    used_pos = deque(used_pos)

                    used_poi.append(change[used_mark][0])
                    used_pos.append(change['index'])

                    used_poi.popleft()
                    used_pos.popleft()

                    used_poi = list(used_poi)
                    used_pos = list(used_pos)

                with open(memory_path, 'w', encoding='utf-8') as f:
                    memory_buffer = {'POI': used_poi,
                                     'POSITION': used_pos}
                    json.dump(memory_buffer, f, indent=4, ensure_ascii=False)

                with open(infer_time_path, 'w', encoding='utf-8') as f:
                    json.dump(infer_time, f, indent=4, ensure_ascii=False)

                with open(token_usage_path, 'w', encoding='utf-8') as f:
                    json.dump(token_usage, f, indent=4, ensure_ascii=False)

            else:
                failure_traj.append(traj_id)
                print(f"{'~' * 20}Failure:{failure_traj}{'~' * 20}")
                print(traj_id, change)

            # print(memory)
            print(json_response)

            result[str(traj_id)] = {'seqID': str(traj_id),
                                    'original itinerary': og_traj_desc,
                                    'response': json_response}

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    # print(f"Infer Time: {infer_time}")
    # print(f"Token Usage: {token_usage}")


if __name__ == '__main__':
    city_set = ['Toro',
                'Melb',
                'Florence']
    op_set = ['ADD', 'DELETE', 'REPLACE']


    # Toronto + ADD perturbation
    iTIMO_perturbation(city_set[0], op=op_set[0], checkpoint=-1)

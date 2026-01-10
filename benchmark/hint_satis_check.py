# -*- coding: utf-8 -*-
"""
Hint satisfaction checker (multi-op: ADD / DELETE / REPLACE).

核心入口函数:
    check_one(example_input, example_output, response_obj)

其中:
- example_input: examples.json 里的一条 record["example_input"]
- example_output: 对应的 record["example_output"]
- response_obj: 预测结果文件里这一条的 rec["response"]
    * 可以是 dict / str / list([... , {...}]) 等, 函数内部会做 json-repair

支持 3 种 repair 操作:
- ADD:    在 need_to_modify itinerary 里 INSERT 一个 POI
- DELETE: 从 need_to_modify itinerary 里 DELETE 一个 POI
- REPLACE:把 need_to_modify itinerary 里的某个 POI 替换成另一个

特别注意(对应你的说明):
- *_ADD_examples.json: 轨迹是被 "ADD" 扰动的, gold 修复操作通常是 DELETE (example_output 里有 removed_index).
- *_DELETE_examples.json: 轨迹被删除了一个点, gold 修复操作通常是 ADD (example_output 里有 insert_index + selected_poi).
- *_REPLACE_examples.json: 轨迹被替换了一个点, gold 修复操作通常是 REPLACE (replaced_index + selected_poi).

这里不从文件名推断, 而是直接从 example_output / response 里的字段推断:
    insert_index / removed_index / replaced_index 等。

check_one 返回结构:
{
  "ok": bool,            # 所有轴都满足 hint 的总体判定
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


# ---------- 基础小工具 ----------

def json_repair_load(x):
    """把 response 安全转成 dict:
       - 如果已是 dict，原样返回；
       - 如果是 list（如 [analysis, final]），取最后一个 dict 或可修复的 JSON 字符串；
       - 如果是字符串，先 json-repair 再 loads；
       - 否则返回 None。
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        # 优先找最后一个 dict
        for obj in reversed(x):
            if isinstance(obj, dict):
                return obj
        # 再尝试把字符串修复成 json
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
    """支持 '0.3km' / '0.3' / 0.3 / None"""
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
    """把相同 count 的类放在同一个 bucket，按 count 降序。用于“排序是否变化”的判定（忽略并列内部交换）。"""
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
    """返回 (popularity_hinted, category_hinted, spatial_hinted)"""
    s = (hint or "").lower()
    pop = ("popularity" in s)
    # cat: category / categories / diversity 都视为“提到了类别/多样性”
    cat = ("category" in s) or ("categories" in s) or ("diversity" in s)
    # spa: spatial / distance
    spa = ("spatial" in s) or ("distance" in s)
    return pop, cat, spa


def sign_with_eps(x: float, eps: float = 1e-6) -> int:
    """带容忍区间的符号：>eps -> +1, <-eps -> -1, 其他 -> 0"""
    if x is None:
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def to_int_safe(x):
    """安全转 int，失败返回 None"""
    try:
        return int(x)
    except Exception:
        return None


# ---------- 统计 BEFORE / AFTER ----------

def pop_counts(traj: List[List[Any]]) -> Dict[str, int]:
    """统计 high/medium/low 的个数"""
    keys = ["high", "medium", "low"]
    c = Counter()
    for poi in traj:
        if len(poi) >= 5:
            c[normalize_pop(poi[4])] += 1
    return {k: c.get(k, 0) for k in keys}


def cat_diversity(traj: List[List[Any]]) -> float:
    """简单类别多样性：#unique(cat) / len(traj)"""
    cats = []
    for poi in traj:
        if len(poi) >= 2:
            cats.append(str(poi[1]).strip().lower())
    uniq = len(set(cats))
    if uniq <= 1:
        return 0.0
    return uniq / max(len(traj), 1)


def spatial_counts(traj: List[List[Any]], low_km: float, high_km: float) -> Dict[str, int]:
    """统计相邻 legs 的距离区间 low/medium/high 的个数"""
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


# ---------- 从 label / resp 中解析操作类型 ----------

def get_op_and_index_from_label(label: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """
    从 example_output 中解析 gold 修复操作类型及 index:
      - insert_index      -> op = "ADD"
      - removed_index     -> op = "DELETE"
      - replaced_index    -> op = "REPLACE"
      - 兼容 index_label 等备用字段
    """
    if not isinstance(label, dict):
        return None, None

    # 明确字段优先
    if "insert_index" in label:
        return "ADD", to_int_safe(label.get("insert_index"))
    if "removed_index" in label:
        return "DELETE", to_int_safe(label.get("removed_index"))
    if "replaced_index" in label:
        return "REPLACE", to_int_safe(label.get("replaced_index"))

    # 兜底: index_label
    if "index_label" in label:
        return None, to_int_safe(label.get("index_label"))

    return None, None


def get_poi_from_label(label: Dict[str, Any]):
    """从 label 中取 gold 的 POI（对于 ADD / REPLACE 有意义）"""
    if not isinstance(label, dict):
        return None
    if "selected_poi" in label:
        return label["selected_poi"]
    if "poi_label" in label:
        return label["poi_label"]
    return None


def get_op_and_index_from_resp(resp: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """
    从模型 response 里解析预测的修复操作类型及 index.
    如果无法确定操作类型, 返回 (None, None).
    """
    if not isinstance(resp, dict):
        return None, None

    # 先看最明确的:
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

    # 兜底：如果只有 index / position 之类，op 不好判断，返回 None
    for k in ("index", "position", "target_index"):
        if k in resp:
            return None, to_int_safe(resp.get(k))

    return None, None


def get_poi_from_resp(resp: Dict[str, Any]):
    """从 response 里取需要 INSERT/REPLACE 的 POI"""
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
    根据 op / idx / sel_poi 构建修改后的轨迹:
      - op="ADD": 在 idx 处插入 sel_poi
      - op="DELETE": 删除 idx 处的一个点
      - op="REPLACE": 替换 idx 处的一个点为 sel_poi
    返回新的轨迹; 如果参数不合法, 返回 None。
    """
    if not isinstance(traj, list) or len(traj) == 0:
        return None
    if idx is None:
        return None

    n = len(traj)
    i = max(0, min(idx, n))   # clamp

    after = list(traj)

    if op == "ADD":
        # 必须有有效的 sel_poi
        if not isinstance(sel_poi, list) or len(sel_poi) < 5:
            return None
        after = after[:i] + [sel_poi] + after[i:]
        return after

    if op == "DELETE":
        if n == 0:
            return None
        # clamp 后，如果 i==n, 删除最后一个
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

    # op 未知
    return None


# ---------- 单样本核验 ----------

def check_one(example_input: Dict[str, Any],
              example_output: Optional[Dict[str, Any]],
              response_obj: Any) -> Dict[str, Any]:
    """
    对单条样本做 Hint 约束检查。

    输入:
      - example_input: examples.json 里的 example_input
      - example_output: 对应的 gold label（可为 None）
      - response_obj: 预测结果（dict / str / list 等）

    返回:
      {
        "ok": bool,            # 三个轴都满足 hint + 未 mention 轴 invariant
        "axes": { ... },      # popularity / category / spatial 详细结果
        "meta": { ... },      # 一些便于 debug 的元信息
      }
    """

    # 1) 取起始轨迹、hint、距离阈值
    traj = example_input.get("need_to_modify itinerary") or \
           example_input.get("need_to_modify Itinerary") or []
    hint = example_input.get("hint") or ""
    low_txt = example_input.get("threshold_low") or example_input.get("threshold_low_km")
    high_txt = example_input.get("threshold_high") or example_input.get("threshold_high_km")
    low_km = parse_threshold_km(low_txt, fallback=None)
    high_km = parse_threshold_km(high_txt, fallback=None)

    # 2) 解析 LLM 响应
    resp = json_repair_load(response_obj)
    if resp is None:
        return {"ok": False, "reason": "response JSON parse fail", "axes": {}, "meta": {}}

    op_pred, idx_pred = get_op_and_index_from_resp(resp)
    sel_poi_pred = get_poi_from_resp(resp)

    after_pred = build_after(traj, op_pred, idx_pred, sel_poi_pred)

    # 如果完全构建不出 after_pred，就直接视为不通过
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

    # 4) 构造 gold 轨迹用于“方向判定”，如果 example_output 里有信息的话
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

    # share(High) 工具
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

    # 5) Hellinger + 排名是否变化
    H_pop = hellinger(pop_b, pop_p, ["high", "medium", "low"])
    H_spa = hellinger(spa_b, spa_p, ["low", "medium", "high"])
    rk_pop_changed = ranking_changed(pop_b, pop_p)
    rk_spa_changed = ranking_changed(spa_b, spa_p)

    # 6) 从 hint 中解析涉及哪些轴
    pop_hint, cat_hint, spa_hint = parse_axes_from_hint(hint)

    # ---------- popularity 轴 ----------

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
            # 没有 gold，只要求“分布确实改变”
            pop_ok = changed_pop
    else:
        # 未被 hint 的轴：保持不变
        pop_ok = (H_pop <= 0.1) and (not rk_pop_changed)

    # ---------- category 轴 ----------

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

    # ---------- spatial 轴 ----------

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

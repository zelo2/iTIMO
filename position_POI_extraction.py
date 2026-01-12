from typing import List, Tuple, Dict, Any, Optional
import math

Row = List[Any]  # [name, category, lon, lat, popularity]
Itinerary = List[Row]

def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _pop_canon(x: Any) -> str:
    s = str(x).strip().lower()
    if s == "high": return "High"
    if s == "medium": return "Medium"
    if s == "low": return "Low"
    return str(x).strip().title()

def _rows_equal(
    a: Row,
    b: Row,
    *,
    float_tol: float = 1e-6,
    compare_cols: Tuple[int, ...] = (0, 1, 2, 3, 4)  # includes lon/lat
) -> bool:
    """
    Compare two rows to judge “same POI”:
    - 0: name, 1: category, 2: lon, 3: lat, 4: popularity
    - popularity normalized by case
    - lon/lat compared as float with float_tol
    """
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    for idx in compare_cols:
        if idx in (2, 3):  # lon/lat
            fa, fb = _to_float(a[idx]), _to_float(b[idx])
            if fa is None or fb is None:
        if str(a[idx]) != str(b[idx]):  # if not numeric, compare as strings
                    return False
            else:
                if not math.isfinite(fa) or not math.isfinite(fb):
                    if str(a[idx]) != str(b[idx]):
                        return False
                elif abs(fa - fb) > float_tol:
                    return False
        elif idx == 4:  # popularity
            if _pop_canon(a[4]) != _pop_canon(b[4]):
                return False
        else:  # name / category
            if str(a[idx]) != str(b[idx]):
                return False
    return True

def extract_change(
    original: Itinerary,
    perturbed: Itinerary,
    operation: str,
    *,
    float_tol: float = 1e-6,
    compare_cols: Tuple[int, ...] = (0, 1, 2, 3, 4)  # include lon/lat by default
) -> Dict[str, Any]:
    """
    Returns:
      ADD:     {"ok":True,"operation":"ADD","index":i,"inserted": row_after}
      DELETE:  {"ok":True,"operation":"DELETE","index":i,"deleted":  row_before}
      REPLACE: {"ok":True,"operation":"REPLACE","index":i,"before": row_before,"after": row_after}
      Failure: {"ok":False,"error":"..."}
    """
    op = str(operation).strip().upper()
    n, m = len(original), len(perturbed)

    if op == "ADD":
        if m != n + 1:
            return {"ok": False, "error": f"ADD expects len(after)=len(before)+1, got {m} vs {n}."}
    elif op == "DELETE":
        if m != n - 1:
            return {"ok": False, "error": f"DELETE expects len(after)=len(before)-1, got {m} vs {n}."}
    elif op == "REPLACE":
        if m != n:
            return {"ok": False, "error": f"REPLACE expects equal lengths, got {m} vs {n}."}
    else:
        return {"ok": False, "error": f"Unknown operation '{operation}'."}

    if op == "REPLACE":
        diffs = [i for i in range(n)
                 if not _rows_equal(original[i], perturbed[i],
                                    float_tol=float_tol, compare_cols=compare_cols)]
        if len(diffs) == 0:
            return {"ok": False, "error": "REPLACE detected no difference between rows."}
        if len(diffs) > 1:
            return {"ok": False, "error": f"REPLACE detected multiple differing indices: {diffs}."}
        i = diffs[0]
        return {"ok": True, "operation": "REPLACE", "index": i,
                "before": original[i], "after": perturbed[i]}

    # ADD / DELETE: two-pointer alignment
    i = j = 0
    while i < n and j < m and _rows_equal(original[i], perturbed[j],
                                          float_tol=float_tol, compare_cols=compare_cols):
        i += 1
        j += 1

    if op == "ADD":
        insert_idx = j
        k1, k2 = i, j + 1
        while k1 < n and k2 < m and _rows_equal(original[k1], perturbed[k2],
                                                float_tol=float_tol, compare_cols=compare_cols):
            k1 += 1
            k2 += 1
        if k1 != n or k2 != m:
            return {"ok": False, "error": "ADD alignment failed (duplicate/ambiguous rows). "
                                          "Try lowering float_tol or inspect data."}
        return {"ok": True, "operation": "ADD", "index": insert_idx,
                "inserted": perturbed[insert_idx]}

    if op == "DELETE":
        delete_idx = i
        k1, k2 = i + 1, j
        while k1 < n and k2 < m and _rows_equal(original[k1], perturbed[k2],
                                                float_tol=float_tol, compare_cols=compare_cols):
            k1 += 1
            k2 += 1
        if k1 != n or k2 != m:
            return {"ok": False, "error": "DELETE alignment failed (duplicate/ambiguous rows). "
                                          "Try lowering float_tol or inspect data."}
        return {"ok": True, "operation": "DELETE", "index": delete_idx,
                "deleted": original[delete_idx]}

# -*- coding: utf-8 -*-
"""
Build embedding-based RAG neighbors and write them into SFT JSON files.

Directory conventions (auto-detected):
  repo_root/
    benchmark/
      RAG_emd/ (or RAG_Emd/)
        Melb_ADD_examples_qwen3_embeddings.npz
        Melb_ADD_examples_azure_embeddings.npz
        Melb_ADD_examples_KaLM_gemma3_embeddings.npz
        ...
      iTIMO_dataset/
        iTIMO-Florence/
          Florence_ADD_train.json
          Florence_ADD_val.json
          Florence_ADD_test.json
        iTIMO-Melbourne/
        iTIMO-Toronto/

Legacy Dataset/ and SFT_data/ layouts are still supported if present.

Goal:
  For each <City>_<OP> (e.g., Melb_ADD, Florence_DELETE, Toro_REPLACE):
    1) Use train.json samples as the candidate pool (rec_examples must come from train only).
    2) For train/val/test splits, write/overwrite rec_examples_* fields for each embedding backend:
         - Qwen3-Embedding        -> rec_examples_qwen3_8b
         - Azure text-emb-3-large -> rec_examples_gpt_text_large
         - KaLM-gemma3            -> rec_examples_kalm_gemma3
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# City directory mapping (lower-case city token -> folder name)
CITY_DIR_MAP = {
    "melb": "iTIMO-Melbourne",
    "melbourne": "iTIMO-Melbourne",
    "toro": "iTIMO-Toronto",
    "toronto": "iTIMO-Toronto",
    "florence": "iTIMO-Florence",
}


def resolve_rag_dir(root: Path) -> Path:
    """
    Locate the directory containing embedding npz files.
    Checks RAG_emd/ or RAG_Emd/ under <root> and <root>/benchmark.
    """
    candidates = [
        root / "RAG_emd",
        root / "RAG_Emd",
        root / "benchmark" / "RAG_emd",
        root / "benchmark" / "RAG_Emd",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError("Missing RAG_emd directory. Tried: " + ", ".join(str(c) for c in candidates))


def resolve_dataset_root(root: Path) -> Path:
    """
    Locate dataset root (iTIMO_dataset). Tries:
      - <root>/benchmark/iTIMO_dataset
      - <root>/iTIMO_dataset
      - <root>/Dataset (legacy)
    """
    candidates = [
        root / "benchmark" / "iTIMO_dataset",
        root / "iTIMO_dataset",
        root / "Dataset",  # legacy
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError("Missing dataset root. Tried: " + ", ".join(str(c) for c in candidates))


# -------------------------
# JSON helpers
# -------------------------

def load_examples(json_path: Path) -> Dict[str, dict]:
    """Load a JSON dict and coerce all keys to strings."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def save_examples(json_path: Path, data: Dict[str, dict], inplace: bool) -> None:
    """Save JSON back to the same path (inplace) or to a suffixed file."""
    if inplace:
        out_path = json_path
    else:
        out_path = json_path.with_name(json_path.stem + "_with_emd_rec_examples.json")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path}")


def sort_ids(ids: List[str]) -> List[str]:
    """Stable sorting for ids that are often numeric strings."""
    return sorted(ids, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))


# -------------------------
# Embedding loading
# -------------------------

def load_embeddings_from_npz(npz_path: Path, example_ids: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Support two .npz formats:

    A) Arrays:
        ids:        (N,)
        embeddings: (N, D)

    B) Key-per-id:
        "113": (D,), "115": (D,), ...

    Returns:
        id_list: ids that have vectors (preserving the input order of example_ids)
        emd_mat: (n, d) float32 matrix
    """
    npz = np.load(npz_path)
    files = list(npz.files)

    id2vec: Dict[str, np.ndarray] = {}

    if "ids" in files and "embeddings" in files:
        ids_arr = npz["ids"]
        embs = npz["embeddings"]
        if len(ids_arr) != len(embs):
            raise ValueError(f"[{npz_path.name}] 'ids' and 'embeddings' lengths mismatch")

        for sid, vec in zip(ids_arr, embs):
            id2vec[str(sid)] = np.array(vec, dtype="float32").reshape(-1)
    else:
        for k in files:
            vec = npz[k]
            id2vec[str(k)] = np.array(vec, dtype="float32").reshape(-1)

    id_list: List[str] = []
    emd_list: List[np.ndarray] = []
    missing: List[str] = []

    for sid in example_ids:
        if sid in id2vec:
            id_list.append(sid)
            emd_list.append(id2vec[sid])
        else:
            missing.append(sid)

    if missing:
        print(f"[WARN] {npz_path.name}: {len(missing)} ids missing (e.g., {missing[:5]})")

    if not emd_list:
        raise ValueError(f"[{npz_path.name}] No matching vectors for the current dataset. Check ids.")

    emd_mat = np.vstack(emd_list).astype("float32")  # (n, d)
    return id_list, emd_mat


# -------------------------
# Nearest neighbor search (cosine)
# -------------------------

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return mat / norms


def compute_topk_neighbors_within(id_list: List[str], emd_mat: np.ndarray, k: int = 5) -> Dict[str, List[str]]:
    """
    Top-k nearest neighbors within the same set (train -> train).
    The item itself is excluded from its own neighbor list.
    """
    n, _ = emd_mat.shape
    if n <= 1:
        raise ValueError("Not enough vectors for within-set neighbors (n <= 1).")

    emd_norm = _l2_normalize(emd_mat)
    sims = emd_norm @ emd_norm.T  # (n, n)
    np.fill_diagonal(sims, -np.inf)

    k = min(k, n - 1)
    neighbor_dict: Dict[str, List[str]] = {}

    for i in range(n):
        row = sims[i]
        idx_part = np.argpartition(row, -k)[-k:]
        idx_sorted = idx_part[np.argsort(row[idx_part])[::-1]]
        neighbor_dict[id_list[i]] = [id_list[j] for j in idx_sorted]

    return neighbor_dict


def compute_topk_neighbors_cross(
    query_ids: List[str],
    query_emb: np.ndarray,
    index_ids: List[str],
    index_emb: np.ndarray,
    k: int = 5,
) -> Dict[str, List[str]]:
    """
    Cross-set neighbors: for each query item, retrieve top-k from the index set.
    All candidates come only from index_ids.
    If query_id is also in index_ids, self is excluded explicitly.
    """
    nq, d = query_emb.shape
    ni, d2 = index_emb.shape
    if ni == 0:
        raise ValueError("Index set is empty.")
    if d != d2:
        raise ValueError("query_emb and index_emb dimension mismatch.")

    qn = _l2_normalize(query_emb)
    inorm = _l2_normalize(index_emb)

    sims = qn @ inorm.T  # (nq, ni)

    id2pos = {sid: j for j, sid in enumerate(index_ids)}
    for qi, qid in enumerate(query_ids):
        j = id2pos.get(qid)
        if j is not None:
            sims[qi, j] = -np.inf

    k_eff = min(k, ni)
    neighbor_dict: Dict[str, List[str]] = {}

    for qi in range(nq):
        row = sims[qi]
        # If all candidates are -inf, return an empty list
        if np.all(~np.isfinite(row)):
            neighbor_dict[query_ids[qi]] = []
            continue

        idx_part = np.argpartition(row, -k_eff)[-k_eff:]
        idx_sorted = idx_part[np.argsort(row[idx_part])[::-1]]
        neighbor_dict[query_ids[qi]] = [index_ids[j] for j in idx_sorted if np.isfinite(row[j])]

    return neighbor_dict


# -------------------------
# Scan RAG_Emd for npz files
# -------------------------

def scan_rag_dir(rag_dir: Path) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Return:
      base_key -> List[(npz_path, field_name)]
    base_key is like "Melb_ADD", "Florence_DELETE", "Toro_REPLACE",
    used to match JSON filenames: <base_key>_train/val/test.json
    """
    mapping: Dict[str, List[Tuple[Path, str]]] = {}

    for npz_path in sorted(rag_dir.glob("*.npz")):
        stem = npz_path.stem  # e.g., Melb_ADD_examples_qwen3_embeddings

        if stem.endswith("_embeddings"):
            stem_base = stem[: -len("_embeddings")]
        else:
            stem_base = stem

        lower_base = stem_base.lower()
        field_name: Optional[str] = None
        strip_patterns: List[str] = []

        # Qwen3 backend
        if ("qwen3_8b" in lower_base) or ("qwen3-8b" in lower_base) or ("qwen3" in lower_base):
            field_name = "rec_examples_qwen3_8b"
            strip_patterns = ["_qwen3_8b", "_qwen3-8b", "_qwen3"]

        # Azure backend
        elif "azure" in lower_base:
            field_name = "rec_examples_gpt_text_large"
            strip_patterns = ["_azure"]

        # KaLM-gemma3 backend
        elif ("kalm" in lower_base) and ("gemma3" in lower_base):
            field_name = "rec_examples_kalm_gemma3"
            strip_patterns = ["_kalm_gemma3", "_kalm-gemma3", "_kalmgemma3"]

        else:
            print(f"[SKIP] Unrecognized backend (not qwen3/azure/kalm+gemma3): {npz_path.name}")
            continue

        base_stem = stem_base
        for pat in strip_patterns:
            base_stem = base_stem.replace(pat, "")

        if base_stem.endswith("_examples"):
            base_stem = base_stem[: -len("_examples")]

        parts = base_stem.split("_")
        if len(parts) < 2:
            print(f"[WARN] Cannot parse city/op from '{stem}', skipping.")
            continue

        base_key = f"{parts[0]}_{parts[1]}"  # e.g., "Melb_ADD"
        mapping.setdefault(base_key, []).append((npz_path, field_name))

    return mapping


# -------------------------
# Resolve Dataset/iTIMO-xxx path for a given base_key
# -------------------------

def resolve_city_sft_dir(dataset_root: Path, base_key: str) -> Path:
    """
    Resolve the directory that contains <base_key>_train.json.

    Candidate patterns:
      iTIMO_dataset/iTIMO-*/<base_key>_train.json
      iTIMO_dataset/iTIMO-*/SFT_data/<base_key>_train.json
      iTIMO_dataset/iTIMO-*/sft_data/<base_key>_train.json
      iTIMO_dataset/<base_key>_train.json (legacy)

    First try common city mappings, then fall back to scanning all Dataset/iTIMO-* folders.
    """
    train_name = f"{base_key}_train.json"

    # Fast path using common mappings (case-insensitive).
    city = base_key.split("_", 1)[0].lower()
    preferred = []
    mapped = CITY_DIR_MAP.get(city)
    if mapped:
        preferred.append(dataset_root / mapped)

    def candidates_for_city_dir(city_dir: Path) -> List[Path]:
        return [
            city_dir,
            city_dir / "SFT_data",
            city_dir / "sft_data",
        ]

    search_dirs: List[Path] = []
    for city_dir in preferred:
        if city_dir.exists():
            search_dirs.extend(candidates_for_city_dir(city_dir))

    # Fallback: scan all iTIMO-* folders under dataset_root
    for city_dir in sorted(dataset_root.glob("iTIMO-*")):
        if not city_dir.is_dir():
            continue
        search_dirs.extend(candidates_for_city_dir(city_dir))

    # Legacy: allow train files directly under dataset_root or its SFT_data subfolder
    search_dirs.extend(candidates_for_city_dir(dataset_root))

    for cand in search_dirs:
        if (cand / train_name).exists():
            return cand

    raise FileNotFoundError(
        f"Cannot find '{train_name}' under iTIMO_dataset/ (checked folders and SFT_data subfolders)."
    )


# -------------------------
# Process one base_key across train/val/test
# -------------------------

def process_sft_splits_for_base(
    base_key: str,
    npz_list: List[Tuple[Path, str]],
    sft_dir: Path,
    topk: int,
    inplace: bool,
) -> None:
    train_path = sft_dir / f"{base_key}_train.json"
    val_path = sft_dir / f"{base_key}_val.json"
    test_path = sft_dir / f"{base_key}_test.json"

    if not train_path.exists():
        print(f"[SKIP] {base_key}: missing {train_path}")
        return

    print(f"\n=== Processing {base_key} ===")
    print(f"  SFT dir: {sft_dir}")

    train_data = load_examples(train_path)
    val_data = load_examples(val_path) if val_path.exists() else None
    test_data = load_examples(test_path) if test_path.exists() else None

    train_ids = sort_ids(list(train_data.keys()))
    print(f"  train size: {len(train_ids)}")
    if val_data is not None:
        print(f"  val   size: {len(val_data)}")
    if test_data is not None:
        print(f"  test  size: {len(test_data)}")

    for npz_path, field_name in npz_list:
        print(f"\n  -> Backend: {npz_path.name}")
        print(f"     Field:   {field_name}")

        # 1) Build train index
        train_id_in_npz, train_emb = load_embeddings_from_npz(npz_path, train_ids)
        print(f"     train vectors: {len(train_id_in_npz)}  dim={train_emb.shape[1]}")

        if len(train_id_in_npz) <= 1:
            print("     [WARN] Not enough train vectors to build neighbors; skipping this backend.")
            continue

        # train -> train (neighbors are chosen only from train)
        neighbors_train = compute_topk_neighbors_within(train_id_in_npz, train_emb, k=topk)
        updated = 0
        for sid, neigh in neighbors_train.items():
            if sid in train_data:
                train_data[sid][field_name] = [str(x) for x in neigh]
                updated += 1
        print(f"     Wrote train rec_examples for {updated} items.")

        # 2) val -> train
        if val_data is not None and len(val_data) > 0:
            val_ids = sort_ids(list(val_data.keys()))
            val_id_in_npz, val_emb = load_embeddings_from_npz(npz_path, val_ids)
            print(f"     val vectors: {len(val_id_in_npz)}")
            if len(val_id_in_npz) > 0:
                neighbors_val = compute_topk_neighbors_cross(
                    val_id_in_npz, val_emb, train_id_in_npz, train_emb, k=topk
                )
                updated = 0
                for sid, neigh in neighbors_val.items():
                    if sid in val_data:
                        val_data[sid][field_name] = [str(x) for x in neigh]
                        updated += 1
                print(f"     Wrote val rec_examples for {updated} items (candidates from train only).")

        # 3) test -> train
        if test_data is not None and len(test_data) > 0:
            test_ids = sort_ids(list(test_data.keys()))
            test_id_in_npz, test_emb = load_embeddings_from_npz(npz_path, test_ids)
            print(f"     test vectors: {len(test_id_in_npz)}")
            if len(test_id_in_npz) > 0:
                neighbors_test = compute_topk_neighbors_cross(
                    test_id_in_npz, test_emb, train_id_in_npz, train_emb, k=topk
                )
                updated = 0
                for sid, neigh in neighbors_test.items():
                    if sid in test_data:
                        test_data[sid][field_name] = [str(x) for x in neigh]
                        updated += 1
                print(f"     Wrote test rec_examples for {updated} items (candidates from train only).")

    # Save after all backends have been processed
    print("\n  -> Writing JSON outputs ...")
    save_examples(train_path, train_data, inplace)
    if val_data is not None and val_path.exists():
        save_examples(val_path, val_data, inplace)
    if test_data is not None and test_path.exists():
        save_examples(test_path, test_data, inplace)


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root (will search for RAG_emd/ and benchmark/iTIMO_dataset/ relative to it).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve for each item (default: 5).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="If set, overwrite original JSON files in place. Otherwise write to *_with_emd_rec_examples.json.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rag_dir = resolve_rag_dir(root)
    dataset_root = resolve_dataset_root(root)

    print(f"[RAG dir] {rag_dir}")
    print(f"[Dataset root] {dataset_root}")

    mapping = scan_rag_dir(rag_dir)
    if not mapping:
        print(f"No usable npz files found under: {rag_dir}")
        return

    print("Detected city/op combinations:")
    for base_key, pairs in mapping.items():
        print(f"  - {base_key}: {[p.name for p, _ in pairs]}")

    for base_key, npz_list in mapping.items():
        sft_dir = resolve_city_sft_dir(dataset_root, base_key)
        process_sft_splits_for_base(
            base_key=base_key,
            npz_list=npz_list,
            sft_dir=sft_dir,
            topk=args.topk,
            inplace=args.inplace,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

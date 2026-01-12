# -*- coding: utf-8 -*-
"""
Build rec_examples for embedding-based RAG in batch.

Expected layout:
    benchmark/
      Melb_ADD_examples.json
      Melb_ADD_examples_sample1000.json
      Melb_DELETE_examples.json
      ...
      RAG_Emd/
        Melb_ADD_examples_qwen3_embeddings.npz
        Melb_ADD_examples_azure_embeddings.npz
        Melb_ADD_examples_KaLM_gemma3_embeddings.npz
        ...

NPZ naming (any of these patterns):
    <City>_<OP>_examples_qwen3_embeddings.npz
    <City>_<OP>_examples_qwen3_8b_embeddings.npz
    <City>_<OP>_examples_qwen3-8b_embeddings.npz
    <City>_<OP>_examples_azure_embeddings.npz
    <City>_<OP>_examples_KaLM_gemma3_embeddings.npz
    <City>_<OP>_examples_kalm_gemma3_embeddings.npz

For each <City>_<OP>_examples.json:
  1) Find all matching npz (qwen / azure / kalm-gemma3)
  2) Compute rec_examples_* on the full examples.json
  3) If *_examples_sample1000.json exists, also compute within the 1000-subset only

Field names:
    Qwen3-Embedding            -> rec_examples_qwen3_8b
    Azure text-embedding-3-large -> rec_examples_gpt_text_large
    KaLM-gemma3                -> rec_examples_kalm_gemma3
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------- Utilities ----------

def load_examples(examples_path: Path) -> Dict[str, dict]:
    """Load JSON and coerce all keys to strings."""
    with examples_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def load_embeddings_from_npz(npz_path: Path, example_ids: List[str]):
    """
    Support two NPZ formats:
    A) Arrays:
        ids:        (N,)
        embeddings: (N, D)
    B) Key-per-id:
        "113": (D,), "115": (D,), ...

    Returns:
        id_list: ids with vectors (preserving input order of example_ids)
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
    missing = []

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

    emd_mat = np.vstack(emd_list)  # (n, d)
    return id_list, emd_mat


def compute_topk_neighbors(id_list: List[str], emd_mat: np.ndarray, k: int = 5):
    """
    Cosine top-k neighbors (excluding self):
        id -> [neighbor_id1, ..., neighbor_idk]

    id_list / emd_mat can be full set or sample1000 subset.
    """
    n, d = emd_mat.shape
    if n <= 1:
        raise ValueError("Not enough vectors for neighbors (n <= 1).")

    # Normalize
    norms = np.linalg.norm(emd_mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emd_norm = emd_mat / norms

    sims = emd_norm @ emd_norm.T  # (n, n)
    np.fill_diagonal(sims, -np.inf)  # exclude self

    k = min(k, n - 1)
    neighbor_dict: Dict[str, List[str]] = {}

    for i in range(n):
        row = sims[i]
        idx_part = np.argpartition(row, -k)[-k:]
        idx_sorted = idx_part[np.argsort(row[idx_part])[::-1]]
        neighbor_ids = [id_list[j] for j in idx_sorted]
        neighbor_dict[id_list[i]] = neighbor_ids

    return neighbor_dict


# ---------- Scan RAG_Emd to map base examples.json -> multiple npz ----------

def scan_rag_dir(rag_dir: Path):
    """
    Returns mapping:
        base_examples_path -> List[(npz_path, field_name)]

    base_examples_path example:
        <root>/Melb_ADD_examples.json
        <root>/Florence_DELETE_examples.json
    """
    mapping: Dict[Path, List[Tuple[Path, str]]] = {}

    for npz_path in sorted(rag_dir.glob("*.npz")):
        stem = npz_path.stem  # e.g. Melb_ADD_examples_qwen3_embeddings
        if stem.endswith("_embeddings"):
            stem_base = stem[:-len("_embeddings")]
        else:
            stem_base = stem

        lower_base = stem_base.lower()

        backend_tag = None
        field_name = None
        strip_patterns: List[str] = []

        # --- Qwen3-Embedding ---
        if ("qwen3_8b" in lower_base) or ("qwen3-8b" in lower_base) or ("qwen3" in lower_base):
            backend_tag = "qwen3_8b"
            field_name = "rec_examples_qwen3_8b"
            strip_patterns = ["_qwen3_8b", "_qwen3-8b", "_qwen3"]

        # --- Azure text-embedding-3-large ---
        elif "azure" in lower_base:
            backend_tag = "azure"
            field_name = "rec_examples_gpt_text_large"
            strip_patterns = ["_azure"]

        # --- KaLM-gemma3 ---
        elif ("kalm" in lower_base) and ("gemma3" in lower_base):
            backend_tag = "kalm_gemma3"
            field_name = "rec_examples_kalm_gemma3"
            strip_patterns = [
                "_kalm_gemma3",
                "_kalm-gemma3",
                "_KaLM_gemma3",
                "_KaLM-gemma3",
                "_kalmgemma3",
                "_KaLMgemma3",
            ]
        else:
            print(f"[SKIP] cannot detect backend (not qwen3*/azure/kalm-gemma3): {npz_path.name}")
            continue

        base_stem = stem_base
        for pat in strip_patterns:
            base_stem = base_stem.replace(pat, "")

        # Base examples filename without sample1000 suffix
        examples_path = rag_dir.parent / f"{base_stem}.json"
        mapping.setdefault(examples_path, []).append((npz_path, field_name))

    return mapping


# ---------- Apply multiple npz to one json (full or sample1000) ----------

def process_one_json(json_path: Path,
                     npz_list: List[Tuple[Path, str]],
                     topk: int,
                     inplace: bool):
    """
    For a given json (full or sample1000 subset), use multiple embedding backends
    to generate rec_examples_* fields.
    """
    if not json_path.exists():
        print(f"[WARN] {json_path} not found, skip.")
        return

    print(f"\n=== Processing {json_path.name} ===")
    data = load_examples(json_path)

    # Sort to keep ordering stable and avoid int/str mix
    example_ids = sorted(
        data.keys(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )
    print(f"  #samples: {len(example_ids)}")

    for npz_path, field_name in npz_list:
        print(f"  -> using {npz_path.name} to build field {field_name} ...")
        id_list, emd_mat = load_embeddings_from_npz(npz_path, example_ids)
        print(f"     samples with vectors: {len(id_list)}, dim: {emd_mat.shape[1]}")

        neighbors = compute_topk_neighbors(id_list, emd_mat, k=topk)

        updated = 0
        for sid, neigh_ids in neighbors.items():
            if sid not in data:
                continue
            data[sid][field_name] = [str(x) for x in neigh_ids]
            updated += 1

        print(f"     wrote {updated} rec_examples to field {field_name}")

    # Output path
    inplace = True
    if inplace:
        out_path = json_path
    else:
        out_path = json_path.with_name(json_path.stem + "_with_emd_rec_examples.json")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Saved to: {out_path}")


# ---------- Main: process full, then sample1000 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root containing RAG_Emd and *_examples.json (default: cwd)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k neighbors per itinerary (default: 5)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="If set, overwrite original json; otherwise write *_with_emd_rec_examples.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rag_dir = root / "RAG_Emd"

    if not rag_dir.exists():
        raise FileNotFoundError(f"Missing RAG_Emd directory: {rag_dir}")

    mapping = scan_rag_dir(rag_dir)
    if not mapping:
        print(f"No usable npz files found under: {rag_dir}")
        return

    print(f"Found {len(mapping)} base examples.json to process:")
    for ex_path, pairs in mapping.items():
        print(f"  - {ex_path.name}: {[p.name for p, _ in pairs]}")

    for base_examples_path, npz_list in mapping.items():
        # 1) process full examples.json
        if base_examples_path.exists():
            process_one_json(base_examples_path, npz_list, args.topk, args.inplace)
        else:
            print(f"[WARN] base examples not found: {base_examples_path}")

        # 2) if sample1000 exists, process it as its own subset
        sample_path = base_examples_path.with_name(base_examples_path.stem + "_sample1000.json")
        if sample_path.exists():
            print(f"\n  -> detected subset file: {sample_path.name}, running neighbors within 1000 subset")
            process_one_json(sample_path, npz_list, args.topk, args.inplace)
        else:
            # no sample1000 -> skip
            pass

    print("\n✅ All done.")


if __name__ == "__main__":
    main()

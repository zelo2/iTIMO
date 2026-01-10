# -*- coding: utf-8 -*-
"""
批量构建 emd_rag 的 rec_examples：

假定目录结构类似：
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

npz 命名约定（任一匹配即可）：
    <City>_<OP>_examples_qwen3_embeddings.npz
    <City>_<OP>_examples_qwen3_8b_embeddings.npz
    <City>_<OP>_examples_qwen3-8b_embeddings.npz
    <City>_<OP>_examples_azure_embeddings.npz
    <City>_<OP>_examples_KaLM_gemma3_embeddings.npz
    <City>_<OP>_examples_kalm_gemma3_embeddings.npz

对每个 <City>_<OP>_examples.json：
  1. 找到所有对应的 npz（qwen / azure / kalm-gemma3）
  2. 在“全量 examples.json”上算一次 rec_examples_*
  3. 如果存在对应的  *_examples_sample1000.json ，
     再在这个 1000 子集上单独算一遍（只在 1000 条内部找近邻）

字段：
    Qwen3-Embedding  ->  rec_examples_qwen3_8b
    Azure text-embedding-3-large -> rec_examples_gpt_text_large
    KaLM-gemma3      ->  rec_examples_kalm_gemma3
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------- 通用工具 ----------

def load_examples(examples_path: Path) -> Dict[str, dict]:
    """读取 json，并把 key 全部转成字符串。"""
    with examples_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def load_embeddings_from_npz(npz_path: Path, example_ids: List[str]):
    """
    兼容两种 npz 格式：

    A) 之前那种：
        ids:        (N,)
        embeddings: (N, D)
    B) 每个 id 一个 key：
        "113": (D,), "115": (D,), ...

    返回：
        id_list:  实际有向量的 id（按传入 example_ids 的顺序）
        emd_mat:  (n, d) 的 float32 矩阵
    """
    npz = np.load(npz_path)
    files = list(npz.files)

    id2vec: Dict[str, np.ndarray] = {}

    if "ids" in files and "embeddings" in files:
        ids_arr = npz["ids"]
        embs = npz["embeddings"]
        if len(ids_arr) != len(embs):
            raise ValueError(f"[{npz_path.name}] ids 和 embeddings 长度不一致")
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
        print(f"[WARN] {npz_path.name}: {len(missing)} ids 在 npz 里没找到，例如 {missing[:5]} ...")

    if not emd_list:
        raise ValueError(f"[{npz_path.name}] 对当前数据集没有任何匹配向量，请检查 ids。")

    emd_mat = np.vstack(emd_list)  # (n, d)
    return id_list, emd_mat


def compute_topk_neighbors(id_list: List[str], emd_mat: np.ndarray, k: int = 5):
    """
    用余弦相似度为每个 id 找 top-k 近邻（排除自己），返回 dict：
        id -> [neighbor_id1, ..., neighbor_idk]

    这里的 id_list / emd_mat 可以是全量，也可以是 sample1000 的子集。
    """
    n, d = emd_mat.shape
    if n <= 1:
        raise ValueError("向量数量不足，无法做最近邻（n <= 1）")

    # 归一化
    norms = np.linalg.norm(emd_mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emd_norm = emd_mat / norms

    sims = emd_norm @ emd_norm.T  # (n, n)
    np.fill_diagonal(sims, -np.inf)  # 排除自己

    k = min(k, n - 1)
    neighbor_dict: Dict[str, List[str]] = {}

    for i in range(n):
        row = sims[i]
        idx_part = np.argpartition(row, -k)[-k:]
        idx_sorted = idx_part[np.argsort(row[idx_part])[::-1]]
        neighbor_ids = [id_list[j] for j in idx_sorted]
        neighbor_dict[id_list[i]] = neighbor_ids

    return neighbor_dict


# ---------- 扫描 RAG_Emd，建立 “base examples.json -> 多个 npz” 的映射 ----------

def scan_rag_dir(rag_dir: Path):
    """
    返回 mapping:
        base_examples_path -> List[(npz_path, field_name)]

    base_examples_path 形如：
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
            print(f"[SKIP] 无法识别后端（既不含 qwen3*、azure，也不含 kalm/gemma3）：{npz_path.name}")
            continue

        base_stem = stem_base
        for pat in strip_patterns:
            base_stem = base_stem.replace(pat, "")

        # 这里得到的 examples 文件名是不带 sample1000 的“全量版”
        examples_path = rag_dir.parent / f"{base_stem}.json"
        mapping.setdefault(examples_path, []).append((npz_path, field_name))

    return mapping


# ---------- 对某一个 json（可以是 full，也可以是 sample1000）应用若干 npz ----------

def process_one_json(json_path: Path,
                     npz_list: List[Tuple[Path, str]],
                     topk: int,
                     inplace: bool):
    """
    在给定的 json 上（可以是全量，也可以是 sample1000 子集）
    利用 npz_list 里的多个 embedding 后端，生成对应的 rec_examples_* 字段。
    """
    if not json_path.exists():
        print(f"[WARN] {json_path} 不存在，跳过。")
        return

    print(f"\n=== 处理 {json_path.name} ===")
    data = load_examples(json_path)

    # 排序：统一返回 tuple，避免 int/str 混比较
    example_ids = sorted(
        data.keys(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )
    print(f"  样本数：{len(example_ids)}")

    for npz_path, field_name in npz_list:
        print(f"  -> 使用 {npz_path.name} 生成字段 {field_name} ...")
        id_list, emd_mat = load_embeddings_from_npz(npz_path, example_ids)
        print(f"     有向量的样本数：{len(id_list)}，维度：{emd_mat.shape[1]}")

        neighbors = compute_topk_neighbors(id_list, emd_mat, k=topk)

        updated = 0
        for sid, neigh_ids in neighbors.items():
            if sid not in data:
                continue
            data[sid][field_name] = [str(x) for x in neigh_ids]
            updated += 1

        print(f"     已写入 {updated} 条 rec_examples 到字段 {field_name}")

    # 输出路径
    inplace = True
    if inplace:
        out_path = json_path
    else:
        out_path = json_path.with_name(json_path.stem + "_with_emd_rec_examples.json")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  保存结果到：{out_path}")


# ---------- 主逻辑：先处理 full，再处理对应的 sample1000 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="工程根目录（里面要有 RAG_Emd 和 *_examples.json），默认当前目录",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="每条 itinerary 取多少个相似样本，默认 5",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="若指定，则直接覆盖原始 json；否则写到 *_with_emd_rec_examples.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rag_dir = root / "RAG_Emd"

    if not rag_dir.exists():
        raise FileNotFoundError(f"RAG_Emd 目录不存在：{rag_dir}")

    mapping = scan_rag_dir(rag_dir)
    if not mapping:
        print(f"在 {rag_dir} 下没有发现任何可用的 npz 文件。")
        return

    print(f"找到 {len(mapping)} 个 base examples.json 需要处理：")
    for ex_path, pairs in mapping.items():
        print(f"  - {ex_path.name}: {[p.name for p, _ in pairs]}")

    for base_examples_path, npz_list in mapping.items():
        # 1) 先处理全量 examples.json
        if base_examples_path.exists():
            process_one_json(base_examples_path, npz_list, args.topk, args.inplace)
        else:
            print(f"[WARN] base examples 不存在：{base_examples_path}")

        # 2) 如果有对应的 sample1000，就把它当成“新的数据集”再处理一次
        sample_path = base_examples_path.with_name(base_examples_path.stem + "_sample1000.json")
        if sample_path.exists():
            print(f"\n  -> 检测到子集文件：{sample_path.name}，将单独在 1000 条内部做最近邻")
            process_one_json(sample_path, npz_list, args.topk, args.inplace)
        else:
            # 没有 sample1000 就略过
            pass

    print("\n✅ 全部处理完毕。")


if __name__ == "__main__":
    main()

import argparse
import json
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CITY_DIR_MAP = {
    "Melb": "iTIMO-Melbourne",
    "Toro": "iTIMO-Toronto",
    "Florence": "iTIMO-Florence",
}


def split_ids(seq_ids, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    ids = list(seq_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids, (n_train, n_val, n_test)


def load_examples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Examples JSON must be a dict: {path}")
    return {str(k): v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser(description="Split trajectories and build benchmark JSON splits.")
    parser.add_argument(
        "--city",
        choices=["Florence", "Melb", "Toro"],
        help="City name to auto-resolve data4perturb/<City>/ paths.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a full trajectories CSV (overrides --city auto-detect).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory to write train/val/test CSVs (default: data4perturb/<City>).",
    )
    parser.add_argument(
        "--op",
        choices=["ADD", "DELETE", "REPLACE"],
        help="If set, split <City>_<OP>_examples.json into benchmark train/val/test JSON.",
    )
    parser.add_argument(
        "--examples_path",
        default=None,
        help="Path to <City>_<OP>_examples.json (default: repo root).",
    )
    parser.add_argument(
        "--examples_out_dir",
        default=None,
        help="Output dir for benchmark JSON splits (default: benchmark/iTIMO_dataset/iTIMO-<City>).",
    )
    parser.add_argument(
        "--force_resplit",
        action="store_true",
        help="Force re-splitting even if train/val/test CSVs already exist.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling seqIDs.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio.")
    args = parser.parse_args()

    city = args.city or "Florence"
    out_dir = Path(args.out_dir or f"data4perturb/{city}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    if not args.force_resplit and args.input is None and train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Using existing split CSVs.")
    else:
        if args.input:
            input_path = Path(args.input)
            df = pd.read_csv(input_path)
        else:
            traj_path = out_dir / f"Trajectories-{city.upper()}.csv"
            if traj_path.exists():
                df = pd.read_csv(traj_path)
            else:
                split_paths = [train_path, val_path, test_path]
                missing = [p for p in split_paths if not p.exists()]
                if missing:
                    raise FileNotFoundError(
                        "Missing trajectories CSV and split CSVs. Provide --input or --city with existing data."
                    )
                parts = [pd.read_csv(p) for p in split_paths]
                df = pd.concat(parts, ignore_index=True)
                print("[WARN] No Trajectories-<CITY>.csv found; merged existing splits and re-splitting.")

        if "seqID" not in df.columns:
            raise ValueError("Missing seqID column in input CSV.")

        seq_ids = df["seqID"].dropna().unique().tolist()
        train_ids, val_ids, test_ids, counts = split_ids(
            seq_ids, args.train_ratio, args.val_ratio, args.seed
        )

        train_df = df[df["seqID"].astype(str).isin({str(x) for x in train_ids})]
        val_df = df[df["seqID"].astype(str).isin({str(x) for x in val_ids})]
        test_df = df[df["seqID"].astype(str).isin({str(x) for x in test_ids})]

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")

        n_train, n_val, n_test = counts
        print("Split seqIDs:", f"train={n_train}", f"val={n_val}", f"test={n_test}")
        print("Rows:", f"train={len(train_df)}", f"val={len(val_df)}", f"test={len(test_df)}")
        print(f"Wrote: {train_path}, {val_path}, {test_path}")

    if args.op:
        examples_path = Path(
            args.examples_path or (REPO_ROOT / f"{city}_{args.op}_examples.json")
        )
        if not examples_path.exists():
            raise FileNotFoundError(f"Missing examples file: {examples_path}")

        examples = load_examples(examples_path)
        split_ids_map = {
            "train": set(str(x) for x in train_df["seqID"].dropna().unique()),
            "val": set(str(x) for x in val_df["seqID"].dropna().unique()),
            "test": set(str(x) for x in test_df["seqID"].dropna().unique()),
        }

        out_examples_dir = Path(
            args.examples_out_dir
            or (REPO_ROOT / "benchmark" / "iTIMO_dataset" / CITY_DIR_MAP[city])
        )
        out_examples_dir.mkdir(parents=True, exist_ok=True)

        for split, ids in split_ids_map.items():
            data = {k: v for k, v in examples.items() if str(k) in ids}
            out_path = out_examples_dir / f"{city}_{args.op}_{split}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote benchmark splits to: {out_examples_dir}")


if __name__ == "__main__":
    main()

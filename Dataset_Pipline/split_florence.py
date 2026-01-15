import argparse
import random
from pathlib import Path

import pandas as pd


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


def main():
    parser = argparse.ArgumentParser(description="Split trajectories into train/val/test CSVs.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling seqIDs.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio.")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        out_dir = Path(args.out_dir or input_path.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(input_path)
    else:
        city = args.city or "Florence"
        out_dir = Path(args.out_dir or f"data4perturb/{city}")
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = out_dir / f"Trajectories-{city.upper()}.csv"
        if traj_path.exists():
            df = pd.read_csv(traj_path)
        else:
            # Fall back to merging existing split CSVs, then re-splitting.
            split_paths = [out_dir / "train.csv", out_dir / "val.csv", out_dir / "test.csv"]
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

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    n_train, n_val, n_test = counts
    print("Split seqIDs:", f"train={n_train}", f"val={n_val}", f"test={n_test}")
    print("Rows:", f"train={len(train_df)}", f"val={len(val_df)}", f"test={len(test_df)}")
    print(f"Wrote: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()

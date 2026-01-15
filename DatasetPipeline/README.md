# 🧪 DatasetPipeline: Perturbation & Dataset Construction

This folder contains the data construction pipeline for iTIMO, including perturbation
generation and building the example JSON files used by the benchmark.

## 📥 Inputs

- `data4perturb/<City>/{train,val,test}.csv` for split definitions (Melb/Toro provided; Florence can be generated).
- `data4perturb/Florence/Trajectories-FLORENCE.csv`
- `data4perturb/Florence/PoIs-FLORENCE.csv`
- `data4perturb/Florence/Categories-Florence.csv` (optional in scripts, used for reference).
- `og_dataset/data-ijcai15/poiList-ijcai15/POI-*.csv` and `og_dataset/data-cikm16/POI-*.csv`
  for POI metadata.

## 🧬 1) Perturbation generation

Scripts:
- `DatasetPipeline/V31FM_perturbation.py` (LLM + tool-calling)
- `DatasetPipeline/baseline_perturbation.py` (baseline)

Before running, set API keys in the scripts (and/or `Benchmark/api_key/api_key.py` if used).

```bash
python DatasetPipeline/V31FM_perturbation.py
```

Notes:
- City / operation are configured in each script’s `__main__` block.
- Outputs are written under:
  - `data-cikm16/perturbation_data/` (Melb)
  - `data-ijcai15/Toro/perturbation_data/` (Toro)
  - `LearNext-DATASET/Florence/perturbation_data/` (Florence)

## 🧱 2) Build examples (`*_examples.json`)

`DatasetPipeline/data_cons.py` converts perturbation outputs into example JSON files
that include candidate POIs, thresholds, and gold labels.

```bash
python DatasetPipeline/data_cons.py
```

Notes:
- City / operation are configured in `__main__`.
- The output file is written to the current working directory as
  `<City>_<OP>_examples.json`.

## 🧩 3) Build benchmark splits (train/val/test)

`*_examples.json` are keyed by `seqID`. To build the benchmark dataset, split the
examples into train/val/test (7:1:2) and write them to:

`Benchmark/iTIMO_dataset/iTIMO-<City>/<City>_<OP>_{train,val,test}.json`

Use the split CSVs under `data4perturb/<City>/` and filter by the `seqID` column
(already 7:1:2). Example:

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

city = "Toro"   # Melb / Toro
op = "ADD"
city_dir = {"Melb": "iTIMO-Melbourne", "Toro": "iTIMO-Toronto"}

with open(f"{city}_{op}_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

splits = {}
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data4perturb/{city}/{split}.csv")
    ids = set(str(x) for x in df["seqID"].unique())
    splits[split] = {k: v for k, v in examples.items() if str(k) in ids}

out_dir = Path("Benchmark/iTIMO_dataset") / city_dir[city]
out_dir.mkdir(parents=True, exist_ok=True)
for split, data in splits.items():
    out_path = out_dir / f"{city}_{op}_{split}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
PY
```

If you do not have split CSVs yet, generate them first:

```bash
python DatasetPipeline/data_split.py
```

This uses a fixed random seed of 42 by default. You can override it via `--seed`.
You can also auto-generate splits for Melb/Toro by specifying `--city`:

```bash
python DatasetPipeline/data_split.py --city Melb
python DatasetPipeline/data_split.py --city Toro
```

To build benchmark JSON splits from `data_cons.py` outputs, pass `--op`:

```bash
python DatasetPipeline/data_split.py --city Florence --op ADD
python DatasetPipeline/data_split.py --city Melb --op DELETE
```

## 🔎 4) (Optional) RAG neighbor construction

These scripts use embedding files under `RAG_emd/` to add `rec_examples_*` fields.

```bash
python DatasetPipeline/RAG_build_emd.py --root . --inplace
python DatasetPipeline/RAG_build_hint.py
```

Notes:
- `RAG_build_emd.py` writes `rec_examples_*` into
  `Benchmark/iTIMO_dataset/<City>_<OP>_{train,val,test}.json`.
- `RAG_build_hint.py` builds `rec_examples` from train-only pools.

### 🔗 Relationship between RAG scripts

- `RAG_build_emd.py`: add `rec_examples_*` to train/val/test splits, with
  retrieval candidates drawn from train only.
- `RAG_build_hint.py`: add `rec_examples` using hint similarity (no embeddings).

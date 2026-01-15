# Dataset_Pipline: Perturbation & Dataset Construction

This folder contains the data construction pipeline for iTIMO, including perturbation
generation and building the example JSON files used by the benchmark.

## Inputs

- `data4perturb/Melb|Toro/{train,val,test}.csv` for Melb/Toro splits.
- `data4perturb/Florence/Trajectories-FLORENCE-final2.csv`
- `data4perturb/Florence/PoIs-FLORENCE-final.csv`
- `data4perturb/Florence/Categories-Florence.csv` (optional in scripts, used for reference).
- `og_dataset/data-ijcai15/poiList-ijcai15/POI-*.csv` and `og_dataset/data-cikm16/POI-*.csv`
  for POI metadata.

## 1) Perturbation generation

Scripts:
- `Dataset_Pipline/V31FM_perturbation.py` (LLM + tool-calling)
- `Dataset_Pipline/baseline_perturbation.py` (baseline)

Before running, set API keys in the scripts (and/or `benchmark/api_key/api_key.py` if used).

```bash
python Dataset_Pipline/V31FM_perturbation.py
```

Notes:
- City / operation are configured in each script’s `__main__` block.
- Outputs are written under:
  - `data-cikm16/perturbation_data/` (Melb)
  - `data-ijcai15/Toro/perturbation_data/` (Toro)
  - `LearNext-DATASET/Florence/perturbation_data/` (Florence)

## 2) Build examples (`*_examples.json`)

`Dataset_Pipline/data_cons.py` converts perturbation outputs into example JSON files
that include candidate POIs, thresholds, and gold labels.

```bash
python Dataset_Pipline/data_cons.py
```

Notes:
- City / operation are configured in `__main__`.
- The output file is written to the current working directory as
  `<City>_<OP>_examples.json`. Move or symlink it to `benchmark/iTIMO_dataset/`
  (or a city subfolder) if you want it picked up by benchmark/RAG tools.

## 3) (Optional) RAG neighbor construction

These scripts use embedding files under `RAG_emd/` to add `rec_examples_*` fields.

```bash
python Dataset_Pipline/RAG_emd_search.py --root .
python Dataset_Pipline/RAG_enhanced_data_cons.py --root . --inplace
python Dataset_Pipline/RAG_hint_based.py
```

Notes:
- `RAG_emd_search.py` updates `*_examples.json` files directly.
- `RAG_enhanced_data_cons.py` writes `rec_examples_*` into
  `benchmark/iTIMO_dataset/<City>_<OP>_{train,val,test}.json`.
- `RAG_hint_based.py` builds `rec_exmaples` (legacy field name) from train-only pools.

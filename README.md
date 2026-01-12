<p align="center">
  <img src="figures/iTIMO.png" width="420" alt="iTIMO" />
</p>

# 🧭 iTIMO: An LLM-Empowered Synthesis Dataset for Travel Itinerary Modification

This repository provides the dataset and code for *iTIMO: An LLM-Empowered Synthesis Dataset for Travel Itinerary Modification*.

## 📦 Dataset

The released benchmark dataset is under `benchmark/Dataset/`:
- `benchmark/Dataset/iTIMO-Florence/`
- `benchmark/Dataset/iTIMO-Melbourne/`
- `benchmark/Dataset/iTIMO-Toronto/`

### 🔁 Perturbation vs. Modification (Important)

In filenames like `benchmark/Dataset/iTIMO-Florence/Florence_ADD_test.json`, the `ADD/DELETE/REPLACE` token refers to the **perturbation** operation used to create the need-to-modify itinerary. The **modification/repair** operation is the *inverse*:
- `*_ADD_*.json` → repair with **DELETE** (gold label field: `removed_index`)
- `*_DELETE_*.json` → repair with **ADD** (gold label fields: `insert_index`, `selected_poi`, `selected_cand_id`)
- `*_REPLACE_*.json` → repair with **REPLACE** (gold label fields: `replaced_index`, `selected_poi`, `selected_cand_id`)

### 🧾 File Naming and Format

- Naming: `<City>_<PerturbOp>_<split>.json` (e.g., `Florence_ADD_test.json`)
- Each file is a JSON dict: `{ "<sid>": sample, ... }`
- `sample["example_input"]` includes:
  - `need_to_modify itinerary`: `[[name, category, lon, lat, popularity], ...]`
  - `hint`: natural-language constraints for axes (popularity / category / spatial)
  - `threshold_low`, `threshold_high`: spatial thresholds (km)
  - `Candidate POIs`: present in `*_DELETE_*.json` and `*_REPLACE_*.json` (needed for ADD/REPLACE repair); typically absent in `*_ADD_*.json`

### 📊 Dataset Size (#samples)

The dataset statistics are provided in the paper (Table 2):

<p align="center">
  <img src="figures/dataset_stats_table2.png" width="900" alt="iTIMO dataset statistics (Table 2)" />
</p>

## 🧪 Perturbation (Generate Need-to-Modify Itineraries)

Use these scripts to generate perturbed (need-to-modify) itineraries from raw trajectories:
- `uni_perturbation.py`: perturbation generator with tool-calling + optional memory
- `baseline_perturbation.py`: baseline perturbation generator

Before running, set API keys in `benchmark/api_key/api_key.py` (and/or in the scripts if required).

```bash
python uni_perturbation.py
```

Notes:
- City / operation are currently configured in each script’s `__main__` block.
- Outputs are written under the corresponding data folders (e.g., `data-cikm16/`, `data-ijcai15/`, `LearNext-DATASET/`), depending on the selected city/operation.

## 🛠️ Installation

Recommended Python `>=3.10`.

```bash
pip install -r requirements.txt
```

Note: running `uni_perturbation.py` / `baseline_perturbation.py` / `benchmark/Prompt_LLM_Eval_*.py` requires access to the corresponding APIs (DeepSeek / Azure OpenAI / OpenAI, etc.).

## 📈 Benchmark: Itinerary Modification Evaluation (Different LLMs)

This benchmark evaluates *itinerary modification (repair)*: given a need-to-modify itinerary, the LLM must output the **repair operation** (the inverse of the perturbation in the filename).

### 0) Prepare dataset paths (required by the benchmark scripts)

Some benchmark scripts expect files under `benchmark/Dataset/<City>_<PerturbOp>_<split>.json`, while the released data is stored under `benchmark/Dataset/iTIMO-*/`. Run once:

```bash
cd benchmark
python - <<'PY'
from pathlib import Path

city_dir = {
    "Melb": "iTIMO-Melbourne",
    "Toro": "iTIMO-Toronto",
    "Florence": "iTIMO-Florence",
}
ops = ["ADD", "DELETE", "REPLACE"]
splits = ["train", "val", "test"]

dataset_root = Path("Dataset")
for city, sub in city_dir.items():
    for op in ops:
        for sp in splits:
            src = dataset_root / sub / f"{city}_{op}_{sp}.json"
            dst = dataset_root / f"{city}_{op}_{sp}.json"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

# eval.py also looks for {City}_{Op}_examples.json at benchmark/ root.
for city in city_dir:
    for op in ops:
        src = dataset_root / f"{city}_{op}_test.json"
        dst = Path(f"{city}_{op}_examples.json")
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

print("Symlinks ready.")
PY
```

### 1) Configure API keys / endpoints

- Azure OpenAI: set `azure_endpoint` and API key in `benchmark/Prompt_LLM_Eval_Azure.py` and `benchmark/api_key/api_key.py`
- DeepSeek: set key in `benchmark/api_key/api_key.py` (used by `benchmark/Prompt_LLM_Eval_DS.py`)
- LM Studio: ensure a local OpenAI-compatible endpoint is running (used by `benchmark/Prompt_LLM_Eval_Lmstudio.py`)

### 2) Run inference (choose one runner)

Each runner writes raw predictions to `benchmark/SFT_results/`.

```bash
cd benchmark
python Prompt_LLM_Eval_Azure.py
# or: python Prompt_LLM_Eval_DS.py
# or: python Prompt_LLM_Eval_Lmstudio.py
```

To switch LLMs / settings, edit the runner’s `__main__` block (e.g., `model_name`, `city_set`, `perturb_op_set`, `rag_settings`, `icl_num`, `temperature`).

### 3) Parse model outputs to JSON

```bash
cd benchmark
python process_pred.py
```

This writes parsed results to `benchmark/results_parsed/`.

### 4) Compute metrics

```bash
cd benchmark
python eval.py
```

The summary is saved to `benchmark/results_parsed/accuracy_hint_summary.json`.

## 🏋️‍♀️ SFT Fine-tuning Runners (Single Setting per Run)

Both SFT runners load data from `benchmark/Dataset/<City>/<City>_<OP>_<split>.json`. You can override base model paths via env (e.g., `ITIMO_FFT_MODEL_QWEN3`, `ITIMO_LORA_MODEL_GEMMA3`).

### Full-parameter FT (Unsloth FFT)

Runs one (city, op) with chosen train/infer RAG + ICL:

```bash
python benchmark/fine_tune_full.py \
  --city Melb \
  --op ADD \
  --model_key qwen3 \
  --train_rag_mode none --train_icl_num 3 \
  --infer_rag_mode none --infer_icl_num 3 \
  --batch_size 8 --max_new_tokens 256
```

Key flags:
- `--city {Melb,Toro,Florence}` and `--op {ADD,DELETE,REPLACE}`
- `--model_key {qwen3,gemma3,llama3}`
- Train setting: `--train_rag_mode`, `--train_icl_num`
- Inference setting: `--infer_rag_mode`, `--infer_icl_num`
- `--batch_size`, `--max_new_tokens`, `--resume/--no-resume`, `--force_rerun`

Outputs: `benchmark/SFT_predictions_fullft/{model}_{city}_{op}_...json`

### LoRA / QLoRA

Runs one (city, op) with chosen train/infer RAG + ICL:

```bash
python benchmark/fine_tune_lora.py \
  --city Melb \
  --op ADD \
  --model_key gemma3 \
  --train_rag_mode none --train_icl_num 3 \
  --infer_rag_mode none --infer_icl_num 3 \
  --batch_size 1 --max_new_tokens 256
```

Key flags mirror the full-FT script (`--city`, `--op`, `--model_key`, train/infer RAG+ICL, batch, tokens, resume/force).

Outputs: `benchmark/SFT_predictions_lora/{model}_{city}_{op}_...json`

## 🗂️ Repository Layout (What Each Part Does)

### 🧩 Top-level scripts

- `uni_perturbation.py`: main perturbation generator (LLM + tool-calling + optional memory).
- `baseline_perturbation.py`: baseline perturbation generator.
- `position_POI_extraction.py`: detects the edit (ADD/DELETE/REPLACE) between an original itinerary and a perturbed itinerary.

### 🧰 Templates

- `template/prompts.py`: prompts used by `uni_perturbation.py`.
- `template/baseline_prompts.py`: prompts used by `baseline_perturbation.py`.
- `template/functions.py`: tool JSON schemas used for tool-calling.
- `template/CaseStudy.py`: case-study/demo utilities (if used).

### 🧪 Benchmark (Repair Task Inference + Evaluation)

- `benchmark/Dataset/`: released benchmark data (see “Dataset” above).
- `benchmark/Prompt_LLM_Eval_Azure.py`: inference via Azure OpenAI → `benchmark/SFT_results/`.
- `benchmark/Prompt_LLM_Eval_DS.py`: inference via DeepSeek API → `benchmark/SFT_results/`.
- `benchmark/Prompt_LLM_Eval_Lmstudio.py`: inference via LM Studio endpoint → `benchmark/SFT_results/`.
- `benchmark/process_pred.py`: parse/repair model outputs → `benchmark/results_parsed/`.
- `benchmark/eval.py`: compute accuracy + hint-pass metrics.
- `benchmark/hint_satis_check.py`: per-sample hint satisfaction checker.
- `benchmark/benchmark_prompts.py`: benchmark prompts.
- `benchmark/RAG_emd_search.py`, `benchmark/RAG_enhanced_data_cons.py`: embedding-based RAG neighbor construction.
- `benchmark/api_key/api_key.py`: API key placeholders.

### 🗃️ Raw data folders (used for perturbation generation)

- `data-cikm16/`: Melbourne raw data and POI lists. Reference: Xiaoting Wang et al., “Improving Personalized Trip Recommendation to Avoid Crowds Using Pedestrian Sensor Data”, CIKM 2016 (see `data-cikm16/README.txt`).
- `data-ijcai15/`: Toronto raw data and POI lists. References: Kwan Hui Lim et al., “Personalized Tour Recommendation based on User Interests and Points of Interest Visit Durations”, IJCAI 2015; and “Towards Next Generation Touring: Personalized Group Tours”, ICAPS 2016 (see `data-ijcai15/poiList-ijcai15/README.txt`).
- `LearNext-DATASET/`: Florence trajectories/POIs/categories (LearNext). Reference: Baraglia, Muntean, Nardini, Silvestri, “LearNext: Learning to Predict Tourists Movements”, CIKM 2013 (see `LearNext-DATASET/ReadMe.txt`).

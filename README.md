<p align="center">
  <img src="logo/iTIMO.png" width="420" alt="iTIMO" />
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

- `data-cikm16/`: Melbourne raw data and POI lists.
- `data-ijcai15/`: Toronto raw data and POI lists.
- `LearNext-DATASET/`: Florence trajectories/POIs/categories.

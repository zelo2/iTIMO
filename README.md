# iTIMO: Dataset and Code for Travel Itinerary Modification

This repository contains code and datasets for the paper *iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification*.

## Task and Dataset Overview

The paper formalizes “itinerary modification” into two tasks:

- **Itinerary Perturbation (synthesizing need-to-modify itineraries)**: given a real itinerary `i`, apply exactly one atomic edit to generate `i*` (a need-to-modify itinerary). The edit operation is `O={ADD, DELETE, REPLACE}` and is constrained by an intent set `Z={popularity, spatial distance, category diversity}` (the perturbation must disrupt the requested attributes).
- **Itinerary Modification (repairing back to the original)**: given a need-to-modify itinerary `i*`, apply exactly one atomic operation to repair it back to the original itinerary `i`.

This repository provides:
- **Pre-built iTIMO SFT/eval data**: `benchmark/Dataset/`
- **Perturbation scripts to generate need-to-modify itineraries**: `uni_perturbation.py` (V3.2 FM: Function Calling + Memory) and `baseline_perturbation.py` (DeepSeek V3.2 / R3.2 baselines)
- **Benchmark scripts (zero-shot / few-shot / RAG / SFT evaluation)**: `benchmark/`

## Project Structure (Paper-to-Code Mapping)

- `uni_perturbation.py`: perturbation generation with V3.2 FM (toolbox + memory; outputs need-to-modify itineraries)
- `baseline_perturbation.py`: perturbation baselines (DeepSeek V3.2 / R3.2)
- `template/prompts.py`: prompt templates for V3.2 FM (tool-calling contract + memory template)
- `template/baseline_prompts.py`: baseline prompt templates
- `template/functions.py`: tool schema (geo distance, stats, category diversity, etc.)
- `position_POI_extraction.py`: extract the edit position and POI by comparing original vs perturbed itineraries
- `benchmark/Dataset/`: iTIMO JSON data for 3 cities (Toronto/Melbourne/Florence) × 3 perturbations (ADD/DELETE/REPLACE) × train/val/test
- `benchmark/`: evaluation + data processing (RAG construction, LLM calls, prediction parsing, metrics)
- `dataset/`, `data-cikm16/`, `data-ijcai15/`, `LearNext-DATASET/`: real itineraries and POI metadata used for perturbation generation (from public datasets used in the paper)

## Environment and Installation

Recommended Python `>=3.10`.

```bash
pip install openai pandas numpy tenacity httpx json-repair tqdm
```

Note: running `uni_perturbation.py` / `baseline_perturbation.py` / `benchmark/Prompt_LLM_Eval_*.py` requires access to the corresponding APIs (DeepSeek / Azure OpenAI / OpenAI, etc.).

## iTIMO (SFT/Eval) Data Format

Data lives in `benchmark/Dataset/iTIMO-*/`, with filenames:
- `<City>_<PerturbOp>_<split>.json`, e.g. `benchmark/Dataset/iTIMO-Melbourne/Melb_ADD_test.json`

Each JSON file is a dict: `{ "<sid>": sample, ... }`, where:
- `sample["example_input"]`:
  - `need_to_modify itinerary`: `[[name, category, lon, lat, popularity], ...]`
  - `hint`: which axes should shift vs remain invariant (popularity / category diversity / spatial)
  - `threshold_low` / `threshold_high`: distance thresholds (km) for spatial class segmentation
- `sample["example_output"]`: gold repair action (**important: `PerturbOp` in the filename indicates how the itinerary was perturbed; the gold action is the inverse edit used to repair it**)
  - `*_ADD_*.json`: itinerary was perturbed by **ADD**, so the gold repair is usually **DELETE**, with field `removed_index`
  - `*_DELETE_*.json`: itinerary was perturbed by **DELETE**, so the gold repair is usually **ADD**, with fields `insert_index`, `selected_poi`, `selected_cand_id`
  - `*_REPLACE_*.json`: itinerary was perturbed by **REPLACE**, so the gold repair is **REPLACE**, with fields `replaced_index`, `selected_poi`, `selected_cand_id`

## Benchmark: Reproducing Experiments (LLM Repair + Evaluation)

The main workflow is under `benchmark/` (corresponding to the paper’s experiments).

### 0) Setup: Make Paths Compatible (Optional but Recommended)

Some evaluation scripts expect data at `benchmark/Dataset/<City>_<Op>_<split>.json`, while this repo stores them under `benchmark/Dataset/iTIMO-*/`. You can create symlinks under `benchmark/` to match the expected paths:

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

# eval.py expects {City}_{Op}_examples.json; here we use the test split as examples
for city in city_dir:
    for op in ops:
        src = dataset_root / f"{city}_{op}_test.json"
        dst = Path(f"{city}_{op}_examples.json")
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

print("Symlinks ready.")
PY
```

### 1) Configure API Keys

- `benchmark/api_key/api_key.py`: fill in `azure_api_key` / `deepseek_api_key` etc.
- Azure: set `azure_endpoint` in `benchmark/Prompt_LLM_Eval_Azure.py`

### 2) Run Evaluation (Generate Model Outputs)

Example using Azure OpenAI (outputs to `benchmark/SFT_results/`):

```bash
cd benchmark
python Prompt_LLM_Eval_Azure.py
```

The run configuration (cities, perturbation types, RAG/ICL settings, temperature, etc.) is at the bottom `__main__` block.

### 3) Parse Outputs into Structured JSON

```bash
cd benchmark
python process_pred.py
```

This cleans/repairs `response` fields in `benchmark/SFT_results/**/_example.json` and writes parsed files to `benchmark/results_parsed/`.

### 4) Compute Metrics (Accuracy + Hint Pass)

```bash
cd benchmark
python eval.py
```

Summary output: `benchmark/results_parsed/accuracy_hint_summary.json`.

## Constructing Need-to-Modify Itineraries (Paper Pipeline)

### V3.2 FM (Function Calling + Memory)

- Entry script: `uni_perturbation.py`
- Prompts and toolbox: `template/prompts.py`, `template/functions.py` (tool implementations are in `uni_perturbation.py` as `tool_*` functions)

Notes:
- This is research-style code; the city/op selection is currently hard-coded in `__main__` (e.g., Toronto + ADD).
- Before running, set the DeepSeek API key (see `benchmark/api_key/api_key.py`, and the `deepseek_api_key` config near the top of the script).
- Outputs are written under `data-cikm16/`, `data-ijcai15/`, `LearNext-DATASET/` in `perturbation_data/` (created if missing).

### Baselines (DeepSeek V3.2 / R3.2)

- Entry script: `baseline_perturbation.py`
- Prompts: `template/baseline_prompts.py`

Also configured in `__main__` (city/op and `think` mode: `think="R"` for reasoning mode, `think="V"` for non-reasoning mode).

## RAG Neighbor Construction (Optional)

If you already have embeddings (`.npz`) under `benchmark/RAG_Emd/`, you can write top-k neighbors back into the SFT JSON files (fields like `rec_examples_qwen3_8b` / `rec_examples_gpt_text_large` / `rec_examples_kalm_gemma3`):

```bash
cd benchmark
python RAG_enhanced_data_cons.py --root . --topk 5 --inplace
```

## Citation

Please cite the iTIMO paper if you use this code or dataset.

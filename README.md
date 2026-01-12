<p align="center">
  <img src="logo/iTIMO.png" width="420" alt="iTIMO" />
</p>

# iTIMO: An LLM-Empowered Synthesis Dataset for Travel Itinerary Modification <img src="logo/iTIMO_roundlogo.png" width="36" alt="iTIMO logo" />

This README documents what each major file/folder in this repository is used for.

## Repository Contents (File/Folder Roles)

### Top-level scripts

- `uni_perturbation.py`: main perturbation generator (LLM + tool-calling + optional memory) that produces perturbed itineraries for different operations.
- `baseline_perturbation.py`: baseline perturbation generator (simpler prompting / baseline settings).
- `position_POI_extraction.py`: utility to detect which edit happened between an original itinerary and a perturbed itinerary (ADD/DELETE/REPLACE + index and POI).

### Prompt and tool templates

- `template/prompts.py`: prompt templates used by `uni_perturbation.py` (including memory prompt and tool-calling constraints).
- `template/baseline_prompts.py`: baseline prompt templates used by `baseline_perturbation.py`.
- `template/functions.py`: JSON-schema definitions of callable tools (used when invoking LLM tool-calling).
- `template/CaseStudy.py`: case-study/demo utilities (if used).

### Benchmark (model inference + evaluation)

- `benchmark/Dataset/`: pre-built JSON datasets for benchmarking (organized by city/operation/split).
- `benchmark/Prompt_LLM_Eval_Azure.py`: runs repair/inference with Azure OpenAI and writes predictions to `benchmark/SFT_results/`.
- `benchmark/Prompt_LLM_Eval_DS.py`: runs repair/inference with DeepSeek-compatible API and writes predictions to `benchmark/SFT_results/`.
- `benchmark/Prompt_LLM_Eval_Lmstudio.py`: runs repair/inference via LM Studio (local OpenAI-compatible endpoint).
- `benchmark/process_pred.py`: parses/repairs model outputs into valid JSON and writes to `benchmark/results_parsed/`.
- `benchmark/hint_satis_check.py`: checks whether a model prediction satisfies the hint constraints (popularity/category/spatial axes).
- `benchmark/eval.py`: computes accuracy metrics and hint-pass metrics from parsed results.
- `benchmark/benchmark_prompts.py`: prompt templates for benchmarking (repair task prompts).
- `benchmark/data_cons.py`: dataset construction/processing helpers used in benchmarking.
- `benchmark/RAG_emd_search.py`: builds RAG neighbors from embeddings and writes `rec_examples_*` fields.
- `benchmark/RAG_enhanced_data_cons.py`: writes embedding-based neighbor ids back into dataset JSONs (top-k retrieval).
- `benchmark/RAG_hint_based.py`: hint-based retrieval / RAG helper logic (if used).
- `benchmark/api_key/api_key.py`: placeholder API key file for different providers (fill with your keys).

### Raw data and derived data folders

- `dataset/`: processed itinerary splits used by perturbation scripts (e.g., `dataset/Melb/*.csv`, `dataset/Toro/*.csv`).
- `data-cikm16/`: Melbourne-related raw data and POI lists.
- `data-ijcai15/`: Toronto-related raw data and POI lists.
- `LearNext-DATASET/`: Florence-related data (trajectories, POIs, categories).

### Assets

- `logo/iTIMO.png`: README cover image.
- `logo/iTIMO_roundlogo.png`: README round logo (used in the title).

## Environment and Installation

Recommended Python `>=3.10`.

```bash
pip install openai pandas numpy tenacity httpx json-repair tqdm
```

Note: running `uni_perturbation.py` / `baseline_perturbation.py` / `benchmark/Prompt_LLM_Eval_*.py` requires access to the corresponding APIs (DeepSeek / Azure OpenAI / OpenAI, etc.).

## Quick Run Pointers (Optional)

- Perturbation generation: edit the `__main__` blocks in `uni_perturbation.py` / `baseline_perturbation.py`, then run `python uni_perturbation.py` or `python baseline_perturbation.py`.
- Benchmark inference/evaluation: see scripts under `benchmark/` (runners, parsers, metrics).

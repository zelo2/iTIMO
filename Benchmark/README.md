# 📈 Benchmark: Evaluation & Fine-tuning

This folder contains evaluation and training code that operates on the released
iTIMO benchmark dataset under `Benchmark/iTIMO_dataset/`.

If you are building the dataset yourself (perturbation + examples + splits),
see [DatasetPipeline/README.md](../DatasetPipeline/README.md) first.

## 🧩 0) Prepare dataset paths (required once)

Some scripts expect flat files under `Benchmark/iTIMO_dataset/` while the released data
is organized under `Benchmark/iTIMO_dataset/iTIMO-*/`. Run once to create symlinks:

```bash
cd Benchmark
python - <<'PY'
from pathlib import Path

city_dir = {
    "Melb": "iTIMO-Melbourne",
    "Toro": "iTIMO-Toronto",
    "Florence": "iTIMO-Florence",
}
ops = ["ADD", "DELETE", "REPLACE"]
splits = ["train", "val", "test"]

dataset_root = Path("iTIMO_dataset")
for city, sub in city_dir.items():
    for op in ops:
        for sp in splits:
            src = dataset_root / sub / f"{city}_{op}_{sp}.json"
            dst = dataset_root / f"{city}_{op}_{sp}.json"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

# eval.py also looks for {City}_{Op}_examples.json at Benchmark/ root.
for city in city_dir:
    for op in ops:
        src = dataset_root / f"{city}_{op}_test.json"
        dst = Path(f"{city}_{op}_examples.json")
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

print("Symlinks ready.")
PY
```

## 🔐 1) Configure API keys / endpoints

- Azure OpenAI: pass `--azure_endpoint` and `--api_key` (or env `AZURE_API_KEY`) to
  `Benchmark/Prompting_LLM.py` with `--provider azure`.
- DeepSeek or other OpenAI-compatible endpoints: pass `--base_url` and `--api_key`
  (or env `OPENAI_API_KEY`) to `Benchmark/Prompting_LLM.py`.
- LM Studio: ensure a local OpenAI-compatible endpoint is running, then call
  `Benchmark/Prompting_LLM.py` with `--base_url http://localhost:1234/v1` and an `--api_key`.

## ▶️ 2) Run inference

```bash
cd Benchmark
python Prompting_LLM.py \
  --city Melb --op ADD --split test \
  --provider openai \
  --model "deepseek-chat" \
  --api_key "$DEEPSEEK_KEY" \
  --rag_mode none --icl_num 3 \
  --temperature 0.1 --max_new_tokens 256
```

Output goes to `Benchmark/prompt_results/` by default.

## 🧾 3) Parse model outputs

```bash
cd Benchmark
python process_pred.py
```

Parsed results are written to `Benchmark/results_parsed/`.

## 📊 4) Compute metrics

```bash
cd Benchmark
python eval.py
```

The summary is saved to `Benchmark/results_parsed/accuracy_hint_summary.json`.

## 🧪 5) (Optional) SFT fine-tuning

Full-parameter FT:

```bash
python Benchmark/fine_tune_full.py \
  --city Melb \
  --op ADD \
  --model_key qwen3 \
  --train_rag_mode none --train_icl_num 3 \
  --infer_rag_mode none --infer_icl_num 3 \
  --batch_size 8 --max_new_tokens 256
```

LoRA / QLoRA:

```bash
python Benchmark/fine_tune_lora.py \
  --city Melb \
  --op ADD \
  --model_key gemma3 \
  --train_rag_mode none --train_icl_num 3 \
  --infer_rag_mode none --infer_icl_num 3 \
  --batch_size 1 --max_new_tokens 256
```

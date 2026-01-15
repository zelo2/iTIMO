# -*- coding: utf-8 -*-
"""
iTIMO Full Fine-tuning (Unsloth FFT) + Immediate Inference (NO model saving)
- A800 80G single GPU
- ✅ full_finetuning=True (full-parameter finetuning)
- ✅ train_on_responses_only: compute loss only on assistant(JSON)
- ✅ Run test inference right after training (batch + LEFT padding)
- ✅ Resume: skip if prediction JSON already complete
- ✅ Lazy-build token cache: only build for needed (city, op, rag, icl)
- ✅ No ckpt/model saving (only prediction JSON)
- ✅ gemma3 / llama3 hyperparameters auto-mapped via MODEL_TRIAL_MAP
- ✅ Notebook-safe: works without __file__ (auto-locates benchmark/)
"""

import os
import re
import json
import random
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from json import JSONDecodeError

import unsloth
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from benchmark_prompts import prompt_add, prompt_delete, prompt_replace


# ======================================================================================
# 1) Notebook-safe benchmark/ discovery
# ======================================================================================
def find_benchmark_dir(start: Path) -> Path:
    """
    Locate benchmark directory that contains benchmark_prompts.py and iTIMO_dataset/.
    Supports:
      - cwd == benchmark/
      - cwd == project root with benchmark in ./benchmark
      - deeper cwd: walk parents until found
    """
    start = start.resolve()

    def ok(p: Path) -> bool:
        return (p / "iTIMO_dataset").exists() and (p / "benchmark_prompts.py").exists()

    # start is already benchmark/
    if ok(start):
        return start

    # start/benchmark
    if ok(start / "benchmark"):
        return (start / "benchmark").resolve()

    # parents
    for p in [start] + list(start.parents):
        if ok(p):
            return p.resolve()
        if ok(p / "benchmark"):
            return (p / "benchmark").resolve()

    raise FileNotFoundError(f"Cannot locate benchmark/ with iTIMO_dataset + benchmark_prompts.py from start={start}")

BENCHMARK_DIR = find_benchmark_dir(Path.cwd())
print("[BENCHMARK_DIR]", BENCHMARK_DIR, flush=True)


# ======================================================================================
# 2) Model trial hyperparameter mapping (gemma3 / llama3 / qwen3)
# ======================================================================================
MODEL_TRIAL_MAP = {
    "gemma3": {
        "seed": 42,
        "max_seq_length": 8192,
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "warmup_steps": 30,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "optim": "adamw_torch",
    },
    "llama3": {
        "seed": 42,
        "max_seq_length": 8192,
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 30,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 0.5,
        "optim": "adamw_torch",
    },
    "qwen3": {
        "seed": 42,
        "max_seq_length": 8192,
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 30,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 0.5,
        "optim": "adamw_torch",
    },
}

def get_trial(model_key: str) -> dict:
    if model_key not in MODEL_TRIAL_MAP:
        raise ValueError(f"Unknown model_key={model_key}, available={list(MODEL_TRIAL_MAP.keys())}")
    return MODEL_TRIAL_MAP[model_key]

def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================================
# 3) Global hardware/training/inference settings (outside trial map)
# ======================================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Inference settings
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
USE_CACHE = True
BATCH_SIZE = 8

PRINT_EVERY_BATCH = 20
FLUSH_EVERY_BATCH = 20

# Checkpoint/resume
RESUME = True
FORCE_RERUN = False

# =========================
# 4) Data/Prompt/RAG settings
# =========================
RAG_FIELD_MAP: Dict[str, Optional[str]] = {
    "none": None,
    "hint": "rec_examples",
    "emd_qwen3_8b": "rec_examples_qwen3_8b",
    "emd_azure": "rec_examples_gpt_text_large",
    "emd_kalm_gemma3": "rec_examples_kalm_gemma3",
}

PROMPT_MAP: Dict[str, str] = {
    "ADD":     prompt_delete,
    "DELETE":  prompt_add,
    "REPLACE": prompt_replace,
}

CHAT_TEMPLATE_MAP = {
    "qwen3": "qwen-3",
    "gemma3": "gemma-3",
    "llama3": "llama-3.1",
}

INSTR_RESP_MARKERS = {
    "qwen3": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "gemma3": {
        "instruction": "<start_of_turn>user\n",
        "response": "<start_of_turn>model\n",
    },
    "llama3": {
        "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}


# ======================================================================================
# 5) Paths and model names
# ======================================================================================
DATA_ROOT = BENCHMARK_DIR / "iTIMO_dataset"
CITY_DIR_MAP: Dict[str, str] = {
    "Melb": "iTIMO-Melbourne",
    "Toro": "iTIMO-Toronto",
    "Florence": "iTIMO-Florence",
}

def dataset_dir(city: str) -> Path:
    try:
        sub = CITY_DIR_MAP[city]
    except KeyError:
        raise ValueError(f"Unknown city key {city}, expected one of {list(CITY_DIR_MAP.keys())}")
    p = DATA_ROOT / sub
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset folder for {city}: {p}")
    return p

# Model names or paths (can be repo ids or local paths)
MODEL_NAME_MAP = {
    "qwen3": os.getenv("ITIMO_FFT_MODEL_QWEN3", "Qwen3-8B"),
    "gemma3": os.getenv("ITIMO_FFT_MODEL_GEMMA3", "gemma-3-4b-it"),
    "llama3": os.getenv("ITIMO_FFT_MODEL_LLAMA3", "llama-3-8b-Instruct"),
}

# Prediction output path
PRED_ROOT = BENCHMARK_DIR / "SFT_predictions_fullft"
PRED_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================================================
# 6) JSON load/save with resume support
# ======================================================================================
def safe_load_json(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (JSONDecodeError, UnicodeDecodeError):
        broken = path.with_suffix(path.suffix + ".broken")
        try:
            os.replace(path, broken)
        except Exception:
            pass
        return {}

def atomic_save_json(path: Path, data: Dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ======================================================================================
# 7) Stable seed (for few-shot sampling)
# ======================================================================================
def stable_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)


# ======================================================================================
# 8) Gemma3 message compatibility
# ======================================================================================
def maybe_fix_messages_for_gemma3(messages: List[dict], model_key: str) -> List[dict]:
    if model_key != "gemma3":
        return messages
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]


# ======================================================================================
# 9) Build few-shot examples (ICL only from train)
# ======================================================================================
def build_example_desc(
    city_name: str,
    perturb_op: str,
    sid: str,
    examples_for_icl: Dict[str, dict],
    valid_icl_pool: List[str],
    rag_mode: str,
    icl_num: int,
    rec_ids: Optional[List[str]],
) -> str:
    if icl_num <= 0:
        return ""

    rng = random.Random(stable_seed(city_name, perturb_op, sid, rag_mode))
    field_name = RAG_FIELD_MAP.get(rag_mode, None)

    if field_name is None or not rec_ids:
        cand_pool = [k for k in valid_icl_pool if k != sid]
    else:
        cand_pool = [str(x) for x in rec_ids if str(x) in examples_for_icl and str(x) != sid]
        if not cand_pool:
            cand_pool = [k for k in valid_icl_pool if k != sid]

    icl_pool = cand_pool[:] if len(cand_pool) <= icl_num else rng.sample(cand_pool, k=icl_num)

    parts: List[str] = []
    for j, ex_sid in enumerate(icl_pool, 1):
        ex_item = examples_for_icl.get(str(ex_sid))
        if ex_item is None:
            continue
        parts.append(
            f"Example #{j} Input:\n{json.dumps(ex_item['example_input'], ensure_ascii=False)}\n"
            f"Example #{j} Output:\n{json.dumps(ex_item['example_output'], ensure_ascii=False)}\n"
        )
    return "".join(parts)


# ======================================================================================
# 10) Build train/val Dataset (conversations -> text)
# ======================================================================================
def build_sft_dataset_conversations(
    city: str,
    op: str,
    rag_mode: str,
    icl_num: int,
    split: str,
) -> Dataset:
    base_dir = dataset_dir(city)
    file_path = base_dir / f"{city}_{op}_{split}.json"
    if not file_path.exists():
        print(f"[WARN] missing {split} file: {file_path}", flush=True)
        return Dataset.from_dict({"conversations": []})

    examples: Dict[str, dict] = json.loads(file_path.read_text(encoding="utf-8"))

    train_path = base_dir / f"{city}_{op}_train.json"
    if train_path.exists():
        icl_examples: Dict[str, dict] = json.loads(train_path.read_text(encoding="utf-8"))
    else:
        icl_examples = examples

    valid_icl_pool = list(icl_examples.keys())
    base_prompt = PROMPT_MAP[op]
    field_name = RAG_FIELD_MAP.get(rag_mode, None)

    records: List[dict] = []
    for sid, item in examples.items():
        sid_str = str(sid)
        ex_in = item["example_input"]
        ex_out = item["example_output"]

        rec_ids = item.get(field_name) if field_name is not None else None
        example_desc = build_example_desc(city, op, sid_str, icl_examples, valid_icl_pool, rag_mode, icl_num, rec_ids)

        system_prompt = base_prompt
        if example_desc:
            system_prompt = system_prompt + "\n" + example_desc + "\n[End of Examples]"

        input_prompt = {k: v for k, v in ex_in.items() if k != "original itinerary"}
        user_text = json.dumps(input_prompt, ensure_ascii=False, indent=2)
        assistant_text = json.dumps(ex_out, ensure_ascii=False)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        records.append({"conversations": messages})

    return Dataset.from_list(records) if records else Dataset.from_dict({"conversations": []})

def map_conversations_to_text(ds: Dataset, tokenizer) -> Dataset:
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}
    return ds.map(formatting_prompts_func, batched=True)


# ======================================================================================
# 11) test sids / cached test entries
# ======================================================================================
def load_test_sids(city: str, op: str) -> List[str]:
    test_path = dataset_dir(city) / f"{city}_{op}_test.json"
    if not test_path.exists():
        return []
    obj = json.loads(test_path.read_text(encoding="utf-8"))
    return [str(k) for k in obj.keys()] if isinstance(obj, dict) else []

def build_cached_test_entries(
    *,
    city: str,
    op: str,
    rag_mode: str,
    icl_num: int,
    model_key: str,
    tokenizer,
) -> List[dict]:
    base_dir = dataset_dir(city)
    test_path = base_dir / f"{city}_{op}_test.json"
    if not test_path.exists():
        print(f"[WARN] missing test file: {test_path}", flush=True)
        return []
    test_examples: Dict[str, dict] = json.loads(test_path.read_text(encoding="utf-8"))

    train_path = base_dir / f"{city}_{op}_train.json"
    if train_path.exists():
        icl_examples: Dict[str, dict] = json.loads(train_path.read_text(encoding="utf-8"))
    else:
        icl_examples = test_examples

    valid_icl_pool = list(icl_examples.keys())
    base_prompt = PROMPT_MAP[op]
    field_name = RAG_FIELD_MAP.get(rag_mode, None)

    entries: List[dict] = []
    for sid, item in test_examples.items():
        sid_str = str(sid)
        example_input = item["example_input"]
        example_output = item.get("example_output", {})

        rec_ids = item.get(field_name) if field_name is not None else None
        example_desc = build_example_desc(city, op, sid_str, icl_examples, valid_icl_pool, rag_mode, icl_num, rec_ids)

        system_prompt = base_prompt
        if example_desc:
            system_prompt = system_prompt + "\n" + example_desc + "\n[End of Examples]"

        input_prompt = {k: v for k, v in example_input.items() if k != "original itinerary"}
        user_text = json.dumps(input_prompt, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        messages = maybe_fix_messages_for_gemma3(messages, model_key)

        enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = enc["input_ids"][0].cpu()
        attention_mask = enc.get("attention_mask", None)
        attention_mask = attention_mask[0].cpu() if attention_mask is not None else torch.ones_like(input_ids)

        entries.append({
            "sid": sid_str,
            "label": json.dumps(example_output, ensure_ascii=False),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

    return entries


# ======================================================================================
# 12) Inference: LEFT padding + batching + resume
# ======================================================================================
def left_pad_batch(ids_list: List[torch.Tensor], msk_list: List[torch.Tensor], pad_id: int):
    maxlen = max(int(x.numel()) for x in ids_list)
    bsz = len(ids_list)

    input_ids = torch.full((bsz, maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, maxlen), dtype=torch.long)

    for i, (ids, msk) in enumerate(zip(ids_list, msk_list)):
        l = int(ids.numel())
        input_ids[i, maxlen - l:] = ids
        attention_mask[i, maxlen - l:] = msk

    return input_ids, attention_mask, maxlen

def get_infer_device(model) -> torch.device:
    for p in model.parameters():
        if hasattr(p, "device") and p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference_cached(
    *,
    model,
    tokenizer,
    entries: List[dict],
    out_path: Path,
    batch_size: int,
    flush_every_batch: int,
    resume: bool,
):
    results = {} if FORCE_RERUN else (safe_load_json(out_path) if resume else {})
    done = set(results.keys())

    infer_device = get_infer_device(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    remaining = [e for e in entries if (not resume) or (e["sid"] not in done)]
    total_rem = len(remaining)
    if total_rem == 0:
        print(f"[SKIP] already complete: {out_path.name}", flush=True)
        return results

    print(f"[RUN] {out_path.name} remaining={total_rem} batch_size={batch_size} left-pad", flush=True)

    batch_cnt = 0
    for s in range(0, total_rem, batch_size):
        batch = remaining[s:s + batch_size]
        ids_list = [e["input_ids"] for e in batch]
        msk_list = [e["attention_mask"] for e in batch]
        sids = [e["sid"] for e in batch]
        labels = [e.get("label", "") for e in batch]

        input_ids, attention_mask, in_len = left_pad_batch(ids_list, msk_list, pad_id)
        input_ids = input_ids.to(infer_device, non_blocking=True)
        attention_mask = attention_mask.to(infer_device, non_blocking=True)

        with torch.inference_mode():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                use_cache=USE_CACHE,
                pad_token_id=pad_id,
            )

        gen_new = gen[:, in_len:]
        decoded = tokenizer.batch_decode(gen_new, skip_special_tokens=True)

        for i, text in enumerate(decoded):
            results[sids[i]] = {"response": text, "label": labels[i]}

        batch_cnt += 1
        if flush_every_batch > 0 and (batch_cnt % flush_every_batch == 0):
            atomic_save_json(out_path, results)
            print(f"[CKPT] {out_path.name} saved={len(results)}", flush=True)

        if PRINT_EVERY_BATCH > 0 and (batch_cnt % PRINT_EVERY_BATCH == 0):
            print(f"[PROG] {out_path.name} batches={batch_cnt} saved={len(results)}", flush=True)

    atomic_save_json(out_path, results)
    print(f"[DONE] {out_path.name} saved={len(results)}", flush=True)
    return results


# ======================================================================================
# 13) Load base model and enable FFT (full finetuning)
# ======================================================================================
def load_base_fft(model_key: str, max_seq_length: int):
    model_name = MODEL_NAME_MAP[model_key]
    dtype_try = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    dtype_fallback = torch.float16

    def _load(dtype):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_name),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=True,
            device_map={"": 0},
            local_files_only=False,
        )
        return model, tokenizer

    try:
        model, tokenizer = _load(dtype_try)
    except Exception as e:
        print(f"[WARN] FFT load dtype={dtype_try} failed: {repr(e)} -> fallback fp16", flush=True)
        model, tokenizer = _load(dtype_fallback)

    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_MAP[model_key])

    # Disable cache during training
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return model, tokenizer


# ======================================================================================
# 14) Single (city, op): FFT train -> release trainer -> immediate inference (no model save)
# ======================================================================================
def train_then_infer_single(
    *,
    city: str,
    op: str,
    model_key: str,
    train_rag_mode: str,
    train_icl_num: int,
    infer_rag_mode: str,
    infer_icl_num: int,
):
    trial = get_trial(model_key)
    set_global_seed(int(trial["seed"]))
    max_seq_length = int(trial["max_seq_length"])

    print("\n" + "=" * 120, flush=True)
    print(f"[PIPE] FULL-FT(no-save) -> INFER | model={model_key} | {city}/{op} | train(rag={train_rag_mode}, icl={train_icl_num}) | infer(rag={infer_rag_mode}, icl={infer_icl_num})", flush=True)
    print(f"[TRIAL] {trial}", flush=True)
    print("=" * 120, flush=True)

    # 1) load base
    model, tokenizer = load_base_fft(model_key, max_seq_length=max_seq_length)

    # 2) build train/val
    train_conv = build_sft_dataset_conversations(city, op, train_rag_mode, train_icl_num, split="train")
    val_conv   = build_sft_dataset_conversations(city, op, train_rag_mode, train_icl_num, split="val")

    if len(train_conv) == 0:
        print(f"[SKIP] empty train set: {city}-{op}", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()
        return

    train_ds = map_conversations_to_text(train_conv, tokenizer)
    val_ds   = map_conversations_to_text(val_conv, tokenizer) if len(val_conv) > 0 else None

    dummy_out = str(PRED_ROOT / "_trainer_tmp")
    bfloat16 = is_bfloat16_supported()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        packing=False,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=dummy_out,

            per_device_train_batch_size=int(trial["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(trial["gradient_accumulation_steps"]),
            num_train_epochs=float(trial["num_train_epochs"]),
            warmup_steps=int(trial["warmup_steps"]),
            learning_rate=float(trial["learning_rate"]),
            lr_scheduler_type=str(trial["lr_scheduler_type"]),

            logging_steps=20,
            report_to="none",
            seed=int(trial["seed"]),

            bf16=bfloat16,
            fp16=(not bfloat16),

            gradient_checkpointing=True,
            optim=str(trial["optim"]),
            weight_decay=float(trial["weight_decay"]),
            max_grad_norm=float(trial["max_grad_norm"]),

            save_strategy="no",
            eval_strategy="no",
        ),
    )

    # ✅ Compute loss only on assistant segment
    markers = INSTR_RESP_MARKERS[model_key]
    trainer = train_on_responses_only(
        trainer,
        instruction_part=markers["instruction"],
        response_part=markers["response"],
    )

    # 3) train
    trainer.train()

    # 4) Release trainer/optimizer memory but keep model for inference
    trained_model = trainer.model
    trained_tokenizer = trainer.tokenizer
    try:
        trainer.optimizer = None
        trainer.lr_scheduler = None
    except Exception:
        pass
    del trainer
    torch.cuda.empty_cache()

    # 5) inference prep: left padding
    trained_model.eval()
    trained_model.config.use_cache = True

    trained_tokenizer.padding_side = "left"
    if trained_tokenizer.pad_token is None:
        trained_tokenizer.pad_token = trained_tokenizer.eos_token
        trained_tokenizer.pad_token_id = trained_tokenizer.eos_token_id

    test_sids = load_test_sids(city, op)
    print(f"[TEST SIDS] {(city, op)} n={len(test_sids)}", flush=True)

    tag = f"{model_key}_fullft_{city}_{op}_trainrag-{train_rag_mode}_trainicl-{train_icl_num}"
    out_path = PRED_ROOT / f"{tag}__prompt-rag-{infer_rag_mode}_icl-{infer_icl_num}_test_predictions.json"

    if (not FORCE_RERUN) and RESUME and out_path.exists():
        existing = safe_load_json(out_path)
        need_sids = set(test_sids)
        done_sids = set(existing.keys())
        if need_sids and need_sids.issubset(done_sids):
            print(f"[SKIP] complete: {out_path.name} ({len(done_sids)}/{len(need_sids)})", flush=True)
            del trained_model, trained_tokenizer, model, tokenizer
            torch.cuda.empty_cache()
            return
        else:
            print(f"[RESUME] {out_path.name} done={len(done_sids)}/{len(need_sids)} rem={len(need_sids-done_sids)}", flush=True)

    print(f"[CACHE BUILD] {(city, op, infer_rag_mode, infer_icl_num)}", flush=True)
    entries = build_cached_test_entries(
        city=city,
        op=op,
        rag_mode=infer_rag_mode,
        icl_num=infer_icl_num,
        model_key=model_key,
        tokenizer=trained_tokenizer,
    )
    print(f"[CACHE DONE] entries={len(entries)}", flush=True)

    if entries:
        run_inference_cached(
            model=trained_model,
            tokenizer=trained_tokenizer,
            entries=entries,
            out_path=out_path,
            batch_size=BATCH_SIZE,
            flush_every_batch=FLUSH_EVERY_BATCH,
            resume=(RESUME and not FORCE_RERUN),
        )
    else:
        print(f"[WARN] empty entries => skip inference for {(city, op, infer_rag_mode, infer_icl_num)}", flush=True)

    del trained_model, trained_tokenizer, model, tokenizer
    torch.cuda.empty_cache()


# ======================================================================================
# 15) main
# ======================================================================================
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run a single full-FT SFT experiment.")
    parser.add_argument("--city", required=True, choices=list(CITY_DIR_MAP.keys()))
    parser.add_argument("--op", required=True, choices=["ADD", "DELETE", "REPLACE"])
    parser.add_argument("--model_key", default="qwen3", choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--train_rag_mode", default="none")
    parser.add_argument("--train_icl_num", type=int, default=3)
    parser.add_argument("--infer_rag_mode", default="none")
    parser.add_argument("--infer_icl_num", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--resume", action="store_true", default=RESUME)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force_rerun", action="store_true", default=FORCE_RERUN)
    return parser.parse_args()


def main():
    global BATCH_SIZE, MAX_NEW_TOKENS, RESUME, FORCE_RERUN
    args = parse_args()
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens
    RESUME = args.resume
    FORCE_RERUN = args.force_rerun

    print(f"[DATA_ROOT] {DATA_ROOT}", flush=True)
    print(f"[PRED_ROOT] {PRED_ROOT}", flush=True)
    print(f"[MODEL_NAME_MAP] {MODEL_NAME_MAP}", flush=True)
    print(f"[RUN] city={args.city} op={args.op} model={args.model_key} train_rag={args.train_rag_mode} train_icl={args.train_icl_num} infer_rag={args.infer_rag_mode} infer_icl={args.infer_icl_num}", flush=True)

    train_then_infer_single(
        city=args.city,
        op=args.op,
        model_key=args.model_key,
        train_rag_mode=args.train_rag_mode,
        train_icl_num=args.train_icl_num,
        infer_rag_mode=args.infer_rag_mode,
        infer_icl_num=args.infer_icl_num,
    )

if __name__ == "__main__":
    main()

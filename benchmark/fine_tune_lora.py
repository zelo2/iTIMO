# -*- coding: utf-8 -*-
"""
iTIMO LoRA Fine-tuning (Unsloth SFT) -> Immediate Inference (NO adapter/ckpt saving)
- ✅ QLoRA / LoRA (train per city + op)
- ✅ train_on_responses_only: compute loss only on assistant(JSON)
- ✅ Run test inference right after training (batch + LEFT padding)
- ✅ Do not save LoRA adapter / ckpt (only prediction JSON with resume)
- ✅ Token cache: lazy-build per (city, op, rag, icl)
"""

import hashlib
import json
import os
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from json import JSONDecodeError

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from benchmark_prompts import prompt_add, prompt_delete, prompt_replace

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ======================================================================================
# 1) Benchmark discovery and paths
# ======================================================================================
def find_benchmark_dir(start: Path) -> Path:
    start = start.resolve()
    def ok(p: Path) -> bool:
        return (p / "iTIMO_dataset").exists() and (p / "benchmark_prompts.py").exists()
    if ok(start):
        return start
    if ok(start / "benchmark"):
        return (start / "benchmark").resolve()
    for p in [start] + list(start.parents):
        if ok(p):
            return p.resolve()
        if ok(p / "benchmark"):
            return (p / "benchmark").resolve()
    raise FileNotFoundError(f"Cannot locate benchmark/ with iTIMO_dataset + benchmark_prompts.py from start={start}")

BENCHMARK_DIR = find_benchmark_dir(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())

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

# ✅ Only save prediction JSON (similar to FullFT)
PRED_ROOT = BENCHMARK_DIR / "SFT_predictions_lora"
PRED_ROOT.mkdir(parents=True, exist_ok=True)

# ======================================================================================
# 2) Model names/paths and prompt markers
# ======================================================================================
BASE_MODEL = "gemma3"  # "llama3" | "gemma3" | "qwen3"

MODEL_NAME_MAP = {
    "llama3": os.getenv("ITIMO_LORA_MODEL_LLAMA3", "unsloth/llama-3-8b-Instruct-bnb-4bit"),
    "gemma3": os.getenv("ITIMO_LORA_MODEL_GEMMA3", "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"),
    "qwen3": os.getenv("ITIMO_LORA_MODEL_QWEN3", "unsloth/Qwen3-8B-unsloth-bnb-4bit"),
}
MODEL_NAME: str = MODEL_NAME_MAP[BASE_MODEL]

CHAT_TEMPLATE_MAP = {
    "qwen3": "qwen-3",
    "gemma3": "gemma-3",
    "llama3": "llama-3.1",
}

INSTR_RESP_MARKERS = {
    "qwen3": {"instruction": "<|im_start|>user\n", "response": "<|im_start|>assistant\n"},
    "gemma3": {"instruction": "<start_of_turn>user\n", "response": "<start_of_turn>model\n"},
    "llama3": {
        "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}

# ======================================================================================
# 3) Data/Prompt/RAG settings
# ======================================================================================
CITY_SET: List[str] = ["Melb", "Toro", "Florence"]
PERTURB_OP_SET: List[str] = ["ADD", "DELETE", "REPLACE"]

TRAIN_RAG_MODE: str = "none"
TRAIN_ICL_NUM: int = 3

# Inference stage: run full set (aligned with FullFT settings)
RAG_SETTINGS: List[Tuple[str, int]] = [
    ("none", 0),
    ("none", 3),
    ("hint", 3),
    ("emd_qwen3_8b", 3),
    ("emd_azure", 3),
    ("emd_kalm_gemma3", 3),
]

RAG_FIELD_MAP: Dict[str, Optional[str]] = {
    "none": None,
    "hint": "rec_exmaples",
    "emd_qwen3_8b": "rec_examples_qwen3_8b",
    "emd_azure": "rec_examples_gpt_text_large",
    "emd_kalm_gemma3": "rec_examples_kalm_gemma3",
}

# ======================================================================================
# 4) Training hyperparameters (aligned with FullFT trial style; adjust if needed)
# ======================================================================================
MODEL_TRIAL_MAP = {
    "gemma3": {
        "seed": 42,
        "max_seq_length": 8192,
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
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
        "learning_rate": 5e-5,
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
        "learning_rate": 5e-5,
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
# 5) LoRA config
# ======================================================================================
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# ======================================================================================
# 6) Inference parameters
# ======================================================================================
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
USE_CACHE = True

# ✅ Key: start conservatively for long prompts + 8k context; increase batch after it works
BATCH_SIZE = 1

# ✅ Optional: truncate prompt length during inference (None = no truncation)
# If OOM persists, try 6144 or 4096
MAX_PROMPT_TOKENS: Optional[int] = None

PRINT_EVERY_BATCH = 20
FLUSH_EVERY_BATCH = 20

# JSON resume
RESUME = True
FORCE_RERUN = False

# ======================================================================================
# 7) PROMPT (kept as requested)
# ======================================================================================
PROMPT_MAP: Dict[str, str] = {
    "ADD":     prompt_delete,
    "DELETE":  prompt_add,
    "REPLACE": prompt_replace,
}

# ======================================================================================
# 8) Utilities: JSON load/save with resume
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
# 9) Few-shot & dataset construction (same logic as before)
# ======================================================================================
def stable_seed(*parts: str) -> int:
    h = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

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

def build_sft_dataset_conversations(
    city_name: str,
    perturb_op: str,
    rag_mode: str,
    icl_num: int,
    split: str = "train",
) -> Dataset:
    base_dir = dataset_dir(city_name)
    file_path = base_dir / f"{city_name}_{perturb_op}_{split}.json"
    if not file_path.exists():
        print(f"[WARN] {city_name}-{perturb_op} {split} file not found: {file_path} -> empty dataset.", flush=True)
        return Dataset.from_dict({"conversations": []})

    examples: Dict[str, dict] = json.loads(file_path.read_text(encoding="utf-8"))

    train_file_path = base_dir / f"{city_name}_{perturb_op}_train.json"
    if train_file_path.exists():
        icl_examples: Dict[str, dict] = json.loads(train_file_path.read_text(encoding="utf-8"))
    else:
        print(f"[WARN] train split not found for {city_name}-{perturb_op}, ICL pool falls back to {split}.json", flush=True)
        icl_examples = examples

    valid_icl_pool = list(icl_examples.keys())
    base_prompt: str = PROMPT_MAP[perturb_op]
    field_name = RAG_FIELD_MAP.get(rag_mode, None)

    print(
        f"[LOAD] {city_name}-{perturb_op} split={split} samples={len(examples)} rag_mode={rag_mode} icl_num={icl_num} "
        f"ICL(train_pool)={len(icl_examples)}",
        flush=True,
    )

    records: List[dict] = []
    for sid, item in examples.items():
        sid_str = str(sid)
        example_input = item["example_input"]
        example_output = item["example_output"]

        rec_ids = item.get(field_name) if field_name is not None else None
        example_desc = build_example_desc(
            city_name, perturb_op, sid_str,
            icl_examples, valid_icl_pool,
            rag_mode, icl_num,
            rec_ids,
        )

        system_prompt = base_prompt
        if example_desc:
            system_prompt = system_prompt + "\n" + example_desc + "\n[End of Examples]"

        input_prompt = {k: v for k, v in example_input.items() if k != "original itinerary"}
        user_text = json.dumps(input_prompt, ensure_ascii=False, indent=2)

        assistant_text = json.dumps(example_output, ensure_ascii=False)

        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        records.append({"conversations": messages})

    return Dataset.from_list(records) if records else Dataset.from_dict({"conversations": []})

def map_conversations_to_text(ds: Dataset, tokenizer) -> Dataset:
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}
    return ds.map(formatting_prompts_func, batched=True)

# ======================================================================================
# 10) Marker check (robust logic as before)
# ======================================================================================
def pick_markers_or_raise(sample_text: str) -> Dict[str, str]:
    mk = INSTR_RESP_MARKERS[BASE_MODEL]
    ins, rsp = mk["instruction"], mk["response"]
    if ins in sample_text and rsp in sample_text:
        return mk

    candidates = []
    if BASE_MODEL == "gemma3":
        candidates = [
            {"instruction": "<start_of_turn>user\n", "response": "<start_of_turn>model\n"},
            {"instruction": "<start_of_turn>user\n\n", "response": "<start_of_turn>model\n"},
            {"instruction": "<start_of_turn>user\n", "response": "<start_of_turn>assistant\n"},
        ]
    elif BASE_MODEL == "qwen3":
        candidates = [{"instruction": "<|im_start|>user\n", "response": "<|im_start|>assistant\n"}]
    elif BASE_MODEL == "llama3":
        candidates = [{
            "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response": "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }]

    for c in candidates:
        if c["instruction"] in sample_text and c["response"] in sample_text:
            print(f"[WARN] Using auto-picked markers: {c}", flush=True)
            return c

    head = sample_text[:800]
    raise RuntimeError(
        f"[MASK] Cannot find instruction/response markers in rendered sample_text.\n"
        f"BASE_MODEL={BASE_MODEL}\n"
        f"Tried default={mk}\n"
        f"sample_head={repr(head)}\n"
        f"Tip: print tokenizer.chat_template and inspect the rendered text markers."
    )

# ======================================================================================
# 11) Inference: token cache + LEFT pad + batching (aligned with FullFT)
# ======================================================================================
def maybe_fix_messages_for_gemma3(messages: List[dict], model_key: str) -> List[dict]:
    if model_key != "gemma3":
        return messages
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]

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

        # ✅ Optional: truncate prompt during inference (reduces 8k-sequence OOM risk)
        if MAX_PROMPT_TOKENS is not None and int(input_ids.numel()) > int(MAX_PROMPT_TOKENS):
            input_ids = input_ids[-int(MAX_PROMPT_TOKENS):]
            attention_mask = attention_mask[-int(MAX_PROMPT_TOKENS):]

        entries.append({
            "sid": sid_str,
            "label": json.dumps(example_output, ensure_ascii=False),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "length": int(input_ids.numel()),  # ✅ Used for length-sorting to reduce padding
        })

    return entries

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

def _sdp_ctx():
    """
    ✅ SDPA backend selection (new API + fallback)
    - Prefer FLASH / EFFICIENT
    - Allow MATH fallback to avoid RuntimeError: No available kernel
    - Use torch.nn.attention.sdpa_kernel() to silence torch.backends.cuda.sdp_kernel FutureWarning
    """
    if not torch.cuda.is_available():
        return nullcontext()

    # ✅ New API (common in PyTorch 2.1+)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        # Do not disable MATH; some paths (fp32 SDPA) otherwise lack kernels
        return sdpa_kernel([
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ])
    except Exception:
        pass

    # ✅ Legacy fallback (still works but warns)
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=True,
            enable_math=True,  # ✅ Must remain True
        )

    return nullcontext()

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
    results = {} if FORCE_RERUN else (safe_load_json(out_path) if (resume and out_path.exists()) else {})
    done = set(results.keys())

    infer_device = get_infer_device(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    remaining = [e for e in entries if (not resume) or (e["sid"] not in done)]
    # ✅ Sort by length to reduce padding in each batch (cuts peak memory)
    remaining.sort(key=lambda e: e.get("length", 0), reverse=True)

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

        with _sdp_ctx(), torch.inference_mode():
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
            # ✅ Lightly release fragmentation (avoid empty_cache every batch to keep speed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if PRINT_EVERY_BATCH > 0 and (batch_cnt % PRINT_EVERY_BATCH == 0):
            print(f"[PROG] {out_path.name} batches={batch_cnt} saved={len(results)}", flush=True)

    atomic_save_json(out_path, results)
    print(f"[DONE] {out_path.name} saved={len(results)}", flush=True)
    return results

# ======================================================================================
# 12) Single (city, op): LoRA train -> release trainer -> immediate inference (no adapter save)
# ======================================================================================
def train_then_infer_one(city_name: str, perturb_op: str):
    trial = get_trial(BASE_MODEL)
    set_global_seed(int(trial["seed"]))
    max_seq_length = int(trial["max_seq_length"])

    print("\n" + "=" * 120, flush=True)
    print(f"[PIPE] LoRA(no-save) -> INFER | model={BASE_MODEL} | {city_name}/{perturb_op}", flush=True)
    print(f"[TRIAL] {trial}", flush=True)
    print("=" * 120, flush=True)

    # 1) load base model + tokenizer（QLoRA）
    dtype = "bfloat16"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_MAP[BASE_MODEL])

    # Training: right padding fits better (original behavior)
    tokenizer.padding_side = "right"
    model.config.use_cache = False

    # 2) LoRA attach
    FastLanguageModel.for_training(model)
    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=LORA_TARGET_MODULES,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(trial["seed"]),
        use_rslora=False,
        loftq_config=None,
    )

    # 3) build train/val
    train_conv = build_sft_dataset_conversations(
        city_name=city_name,
        perturb_op=perturb_op,
        rag_mode=TRAIN_RAG_MODE,
        icl_num=TRAIN_ICL_NUM,
        split="train",
    )
    val_conv = build_sft_dataset_conversations(
        city_name=city_name,
        perturb_op=perturb_op,
        rag_mode=TRAIN_RAG_MODE,
        icl_num=TRAIN_ICL_NUM,
        split="val",
    )

    if len(train_conv) == 0:
        print(f"[SKIP] empty train set: {city_name}-{perturb_op}", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()
        return

    train_ds = map_conversations_to_text(train_conv, tokenizer)
    val_ds = map_conversations_to_text(val_conv, tokenizer) if len(val_conv) > 0 else None

    # 4) trainer (✅ no ckpt saving: save_strategy="no")
    bfloat16 = is_bfloat16_supported()
    dummy_out = str(PRED_ROOT / "_trainer_tmp")  # placeholder for required field

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

    # 5) Compute loss only on assistant segment (markers must match template)
    sample_text = train_ds[0]["text"]
    markers = pick_markers_or_raise(sample_text)
    print(f"[MARKERS] instruction={repr(markers['instruction'])}  response={repr(markers['response'])}", flush=True)
    trainer = train_on_responses_only(
        trainer,
        instruction_part=markers["instruction"],
        response_part=markers["response"],
    )

    # 6) train
    trainer.train()

    # 7) Release trainer/optimizer memory but keep model for inference
    trained_model = trainer.model

    # ✅ Avoid deprecated access: keep external tokenizer instead of trainer.tokenizer
    trained_tokenizer = tokenizer

    try:
        trainer.optimizer = None
        trainer.lr_scheduler = None
    except Exception:
        pass
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 8) Inference prep: left padding + cache
    FastLanguageModel.for_inference(trained_model)
    trained_model.eval()
    trained_model.config.use_cache = True

    # ✅ Disable gradient checkpointing (prevent it from running during inference)
    try:
        trained_model.gradient_checkpointing_disable()
    except Exception:
        pass
    try:
        trained_model.config.gradient_checkpointing = False
    except Exception:
        pass

    trained_tokenizer.padding_side = "left"
    if trained_tokenizer.pad_token is None:
        trained_tokenizer.pad_token = trained_tokenizer.eos_token
        trained_tokenizer.pad_token_id = trained_tokenizer.eos_token_id

    token_cache: Dict[Tuple[str, str, str, int], List[dict]] = {}
    test_sids = load_test_sids(city_name, perturb_op)
    print(f"[TEST SIDS] {(city_name, perturb_op)} n={len(test_sids)}", flush=True)

    tag = f"{BASE_MODEL}_lora_{city_name}_{perturb_op}_trainrag-{TRAIN_RAG_MODE}_trainicl-{TRAIN_ICL_NUM}"

    for rag_mode, icl_num in RAG_SETTINGS:
        out_path = PRED_ROOT / f"{tag}__prompt-rag-{rag_mode}_icl-{icl_num}_test_predictions.json"

        # Skip if output already covers all test sids
        if (not FORCE_RERUN) and RESUME and out_path.exists():
            existing = safe_load_json(out_path)
            need_sids = set(test_sids)
            done_sids = set(existing.keys())
            if need_sids and need_sids.issubset(done_sids):
                print(f"[SKIP] complete: {out_path.name} ({len(done_sids)}/{len(need_sids)})", flush=True)
                continue
            else:
                print(
                    f"[RESUME] {out_path.name} done={len(done_sids)}/{len(need_sids)} rem={len(need_sids-done_sids)}",
                    flush=True
                )

        k = (city_name, perturb_op, rag_mode, icl_num)
        if k not in token_cache:
            print(f"[CACHE BUILD] {k}", flush=True)
            token_cache[k] = build_cached_test_entries(
                city=city_name,
                op=perturb_op,
                rag_mode=rag_mode,
                icl_num=icl_num,
                model_key=BASE_MODEL,
                tokenizer=trained_tokenizer,
            )
            print(f"[CACHE DONE] {k} entries={len(token_cache[k])}", flush=True)

        entries = token_cache[k]
        if not entries:
            print(f"[WARN] empty entries => skip {k}", flush=True)
            continue

        run_inference_cached(
            model=trained_model,
            tokenizer=trained_tokenizer,
            entries=entries,
            out_path=out_path,
            batch_size=BATCH_SIZE,
            flush_every_batch=FLUSH_EVERY_BATCH,
            resume=(RESUME and not FORCE_RERUN),
        )

    # 9) ✅ Do not save LoRA adapter: release directly
    del trained_model, trained_tokenizer, model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ======================================================================================
# 13) main: iterate city/op for training and inference
# ======================================================================================
def main():
    print(f"[DATA_ROOT] {DATA_ROOT}", flush=True)
    print(f"[PRED   ] {PRED_ROOT}", flush=True)
    print(f"[MODEL  ] {BASE_MODEL} -> {MODEL_NAME}", flush=True)
    print(f"[TRAIN  ] rag={TRAIN_RAG_MODE} icl={TRAIN_ICL_NUM}", flush=True)
    print(f"[INFER  ] settings={RAG_SETTINGS}", flush=True)
    print(f"[INFER  ] BATCH_SIZE={BATCH_SIZE} MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS}", flush=True)

    for city_name in CITY_SET:
        for perturb_op in PERTURB_OP_SET:
            train_then_infer_one(city_name, perturb_op)

if __name__ == "__main__":
    main()

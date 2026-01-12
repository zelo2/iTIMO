#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-setting prompt-based itinerary modification evaluation.
Supports Azure OpenAI, OpenAI-compatible endpoints (DeepSeek, LM Studio, etc.).

Examples:
- Azure GPT-4o-mini (temperature=0, max_new_tokens=2048):
    python Prompting_LLM.py --city Melb --op ADD --split test \
      --provider azure --model gpt-4o-mini \
      --azure_endpoint https://<your-endpoint>.openai.azure.com \
      --api_key "$AZURE_KEY" --azure_api_version 2024-12-01-preview \
      --rag_mode none --icl_num 3 --temperature 0 --max_new_tokens 2048

- DeepSeek (reasoning: temp=1, max_new_tokens=2048):
    python Prompting_LLM.py --city Melb --op ADD --split test \
      --provider openai --model deepseek-chat --base_url https://api.deepseek.com/v1 \
      --api_key "$DEEPSEEK_KEY" \
      --rag_mode none --icl_num 3 --temperature 1 --max_new_tokens 2048

- LM Studio (Qwen3-32B local, temp=0, max_new_tokens=2048):
    python Prompting_LLM.py --city Melb --op ADD --split test \
      --provider openai --model "Qwen3-32B" --base_url http://localhost:1234/v1 \
      --api_key "lm-studio" \
      --rag_mode none --icl_num 3 --temperature 0 --max_new_tokens 2048
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from openai import AzureOpenAI

from benchmark import benchmark_prompts

DATA_ROOT = Path(__file__).resolve().parent / "iTIMO"
CITY_DIR_MAP: Dict[str, str] = {
    "Melb": "iTIMO-Melbourne",
    "Toro": "iTIMO-Toronto",
    "Florence": "iTIMO-Florence",
}

# RAG modes -> example.json field names
RAG_FIELD_MAP: Dict[str, Optional[str]] = {
    "none": None,
    "hint": "rec_exmaples",  # keep spelling as in data
    "emd_qwen3_8b": "rec_examples_qwen3_8b",
    "emd_azure": "rec_examples_gpt_text_large",
    "emd_kalm_gemma3": "rec_examples_kalm_gemma3",
}

PROMPT_MAP = {
    "ADD": benchmark_prompts.prompt_delete,
    "DELETE": benchmark_prompts.prompt_add,
    "REPLACE": benchmark_prompts.prompt_replace,
}


def load_split(city: str, op: str, split: str) -> Dict[str, dict]:
    sub = CITY_DIR_MAP[city]
    path = DATA_ROOT / sub / f"{city}_{op}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected JSON format in {path}")
    return obj


def build_example_desc(
    *,
    sid: str,
    eval_examples: Dict[str, dict],
    icl_examples: Dict[str, dict],
    valid_icl_pool: List[str],
    rag_mode: str,
    icl_num: int,
) -> str:
    if icl_num <= 0:
        return ""

    field_name = RAG_FIELD_MAP.get(rag_mode, None)
    rng = random.Random(hash((sid, rag_mode)) & 0xffffffff)

    if field_name is None:
        cand_pool = [k for k in valid_icl_pool if k != sid]
    else:
        rec_ids = eval_examples[sid].get(field_name) or []
        cand_pool = [str(x) for x in rec_ids if str(x) in icl_examples and str(x) != sid]
        if not cand_pool:
            cand_pool = [k for k in valid_icl_pool if k != sid]

    if len(cand_pool) <= icl_num:
        icl_pool = cand_pool[:]
    else:
        icl_pool = rng.sample(cand_pool, k=icl_num)

    parts: List[str] = []
    for j, ex_sid in enumerate(icl_pool, 1):
        ex_item = icl_examples.get(str(ex_sid))
        if ex_item is None:
            continue
        parts.append(
            f"Example #{j} Input:\n{json.dumps(ex_item['example_input'], ensure_ascii=False)}\n"
            f"Example #{j} Output:\n{json.dumps(ex_item['example_output'], ensure_ascii=False)}\n"
        )
    return "".join(parts)


def build_messages(
    *,
    sid: str,
    eval_examples: Dict[str, dict],
    icl_examples: Dict[str, dict],
    valid_icl_pool: List[str],
    op: str,
    rag_mode: str,
    icl_num: int,
) -> tuple:
    ex = eval_examples[sid]
    example_input = ex["example_input"]
    example_output = ex["example_output"]

    example_desc = build_example_desc(
        sid=sid,
        eval_examples=eval_examples,
        icl_examples=icl_examples,
        valid_icl_pool=valid_icl_pool,
        rag_mode=rag_mode,
        icl_num=icl_num,
    )

    system_prompt = PROMPT_MAP[op]
    if example_desc:
        system_prompt = system_prompt + "\n" + example_desc + "\n[End of Examples]"

    user_text = json.dumps(example_input, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return messages, example_output


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def create_client(args):
    if args.provider == "azure":
        return AzureOpenAI(
            azure_endpoint=args.azure_endpoint,
            api_key=args.api_key,
            api_version=args.azure_api_version,
        )
    # openai-compatible (DeepSeek, LM Studio, etc.)
    return OpenAI(api_key=args.api_key, base_url=args.base_url or None)


def run_once(args):
    eval_examples = load_split(args.city, args.op, args.split)
    icl_examples = load_split(args.city, args.op, "train")
    valid_icl_pool = list(icl_examples.keys())

    out_path = Path(args.output or f"benchmark/prompt_results/prompt_eval_{args.provider}_{args.model}_{args.city}_{args.op}_rag-{args.rag_mode}_icl-{args.icl_num}_{args.split}.json")
    existing = {}
    if args.resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}

    client = create_client(args)
    results = dict(existing)
    sids = [s for s in eval_examples.keys() if s not in results]

    print(f"[RUN] city={args.city} op={args.op} provider={args.provider} model={args.model} split={args.split} remaining={len(sids)}")
    for sid in sids:
        messages, label = build_messages(
            sid=sid,
            eval_examples=eval_examples,
            icl_examples=icl_examples,
            valid_icl_pool=valid_icl_pool,
            op=args.op,
            rag_mode=args.rag_mode,
            icl_num=args.icl_num,
        )
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        text = resp.choices[0].message.content
        results[sid] = {"response": text, "label": label}
        if args.flush_every > 0 and (len(results) % args.flush_every == 0):
            save_json(out_path, results)
            print(f"[CKPT] saved={len(results)}")

    save_json(out_path, results)
    print(f"[DONE] saved={len(results)} -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-setting prompt evaluation runner.")
    parser.add_argument("--city", required=True, choices=list(CITY_DIR_MAP.keys()))
    parser.add_argument("--op", required=True, choices=["ADD", "DELETE", "REPLACE"])
    parser.add_argument("--split", default="test", choices=["test", "val", "full", "train"])
    parser.add_argument("--rag_mode", default="none", choices=list(RAG_FIELD_MAP.keys()))
    parser.add_argument("--icl_num", type=int, default=3)
    parser.add_argument("--provider", default="openai", choices=["azure", "openai"])
    parser.add_argument("--model", required=True, help="Model name or deployment name.")
    parser.add_argument("--api_key", default=None, help="API key (env fallback: OPENAI_API_KEY / AZURE_API_KEY).")
    parser.add_argument("--azure_endpoint", default=None, help="Azure endpoint (required when provider=azure).")
    parser.add_argument("--azure_api_version", default="2024-12-01-preview")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible base URL (for DeepSeek/LM Studio, etc.).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--flush_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--output", default=None, help="Custom output JSON path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.provider == "azure":
        args.api_key = args.api_key or (Path("api_key/api_key.py").exists() and None)
        if args.azure_endpoint is None:
            raise ValueError("azure_endpoint is required when provider=azure")
    else:
        if args.api_key is None:
            raise ValueError("api_key is required for OpenAI-compatible providers")
    run_once(args)


if __name__ == "__main__":
    main()

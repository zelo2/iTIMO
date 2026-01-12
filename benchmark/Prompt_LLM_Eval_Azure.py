# -*- coding: utf-8 -*-
import os, json, re, random, time, threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from benchmark import benchmark_prompts
from pathlib import Path

# ======================================================
# 0) Initialize Azure OpenAI client
# ======================================================
from openai import AzureOpenAI
from api_key import api_key

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="YOUR AZURE ENDPOINT",
    api_key=api_key.azure_api_key,
)

# ================== RAG modes -> example.json field names ==================
RAG_FIELD_MAP = {
    "none": None,
    "hint": "rec_exmaples",              # keep spelling as in data
    "emd_qwen3_8b": "rec_examples_qwen3_8b",
    "emd_azure": "rec_examples_gpt_text_large",
    "emd_kalm_gemma3": "rec_examples_kalm_gemma3",
}


def atomic_dump_json(path: str, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_run_one_sid(*args, **kwargs):
    """
    args:
        0: sid
        1: eval_examples   (current split: test/val/full)
        2: icl_examples    (ICL pool: usually train)
    """
    sid = args[0]
    eval_examples = args[1]
    try:
        return run_one_sid(*args, **kwargs)
    except Exception as e:
        return sid, {
            "response": "",
            "label": eval_examples[sid]["example_output"],
            "error": repr(e),
        }


# ======================================================
# 1) Global TPM rate limiter
# ======================================================
TPM_LIMIT = 100000
WINDOW_SEC = 120

_budget_lock = threading.Lock()
_budget_events = deque()  # (timestamp, tokens)


def _cleanup_events(now):
    while _budget_events and now - _budget_events[0][0] > WINDOW_SEC:
        _budget_events.popleft()


def acquire_budget(estimate_tokens: int):
    while True:
        now = time.time()
        with _budget_lock:
            _cleanup_events(now)
            used = sum(t for _, t in _budget_events)
            if used + estimate_tokens <= TPM_LIMIT:
                _budget_events.append((now, estimate_tokens))
                return
            earliest_t, _ = _budget_events[0]
            wait_s = max(0.2, WINDOW_SEC - (now - earliest_t))
        time.sleep(wait_s)


def update_budget_actual(estimate_tokens: int, actual_tokens: int):
    delta = actual_tokens - estimate_tokens
    if delta == 0:
        return
    now = time.time()
    with _budget_lock:
        _cleanup_events(now)
        _budget_events.append((now, delta))


def rough_token_estimate(text: str) -> int:
    return max(200, int(len(text) / 4))


def parse_retry_after_seconds(err: Exception) -> float:
    msg = str(err)
    m = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", msg, flags=re.I)
    if m:
        return float(m.group(1))
    return 5.0


# ======================================================
# 2) Retry API call
# ======================================================
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(8),
    retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
)
def openaiAPIcall(**kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        time.sleep(parse_retry_after_seconds(e))
        raise


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text,
                  flags=re.DOTALL | re.IGNORECASE)


def chunked(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# ======================================================
# 3) Build messages for one sid (supports multiple RAG modes)
# ======================================================
def build_messages_for_sid(
    sid: str,
    eval_examples: dict,
    icl_examples: dict,
    valid_icl_pool,
    base_prompt: str,
    icl_num: int = 0,
    is_think: bool = False,
    rag_mode: str = "none",
):
    """
    eval_examples: split to evaluate (test/val/full)
    icl_examples : ICL pool (usually train; in full mode may equal eval_examples)
    """
    input_ex = eval_examples[sid]["example_input"]
    label_ex = eval_examples[sid]["example_output"]

    # ----- ICL examples -----
    example_desc = ""
    if icl_num > 0:
        field_name = RAG_FIELD_MAP.get(rag_mode, None)

        rng_seed = hash((sid, rag_mode)) & 0xffffffff
        rng = random.Random(rng_seed)

        if field_name is None:
            # No RAG: random few-shots from ICL pool
            cand_pool = [k for k in valid_icl_pool if k != sid]
        else:
            # RAG: take candidate IDs from eval_examples[sid] (these IDs point to train)
            rec_ids = eval_examples[sid].get(field_name) or []
            cand_pool = [str(x) for x in rec_ids if str(x) != sid]

            if not cand_pool:
                print(f"[WARN] sid={sid} rag_mode={rag_mode} missing {field_name}, fallback to random few-shots")
                cand_pool = [k for k in valid_icl_pool if k != sid]

        if len(cand_pool) <= icl_num:
            icl_pool = cand_pool[:]
        else:
            icl_pool = rng.sample(cand_pool, k=icl_num)

        # Always fetch content from icl_examples (train pool), not eval_examples, to avoid missing keys.
        for j, ex_sid in enumerate(icl_pool, 1):
            ex_item = icl_examples[str(ex_sid)]
            ex_in = ex_item["example_input"]
            ex_out = ex_item["example_output"]
            example_desc += (
                f"Example #{j} Input:\n{json.dumps(ex_in, ensure_ascii=False)}\n"
                f"Example #{j} Output:\n{json.dumps(ex_out, ensure_ascii=False)}\n"
            )

    # ----- system prompt -----
    if is_think:
        system_prompt = base_prompt + "\n" + example_desc + "\n" + "[End of Examples]"
    else:
        system_prompt = (
            "/no_think\n"
            "Only output the final JSON; never print <think> tags or intermediate reasoning.\n"
            + base_prompt + "\n" + example_desc + "\n" + "[End of Examples]"
        )

    # ----- user content -----
    input_prompt = {k: v for k, v in input_ex.items() if k not in {"original itinerary"}}
    user_payload = json.dumps(input_prompt, ensure_ascii=False)
    user_content = "[Input]\n" + user_payload + "\n[End of Input]"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages, label_ex, system_prompt, user_content


# ======================================================
# 4) Single sid invocation
# ======================================================
def run_one_sid(
    sid: str,
    eval_examples: dict,
    icl_examples: dict,
    valid_icl_pool,
    base_prompt: str,
    model_name: str,
    icl_num: int,
    is_think: bool,
    rag_mode: str,
    temperature: float,
):
    messages, label, sys_prompt, user_content = build_messages_for_sid(
        sid, eval_examples, icl_examples, valid_icl_pool,
        base_prompt,
        icl_num=icl_num,
        is_think=is_think,
        rag_mode=rag_mode,
    )

    est_tokens = rough_token_estimate(sys_prompt + user_content)
    acquire_budget(est_tokens)

    resp = openaiAPIcall(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    content = resp.choices[0].message.content
    if is_think:
        content = strip_think(content)

    try:
        actual_tokens = resp.usage.total_tokens
        update_budget_actual(est_tokens, actual_tokens)
    except Exception:
        pass

    return sid, {"response": content, "label": label}


# ======================================================
# 5) Main entry: batch parallel requests
# ======================================================
def batch_parallel_run(
    city_name="Melb",
    perturb_op="DELETE",
    icl_num=3,
    model_name="gpt-4.1",
    is_think=True,
    is_full=False,                # False: eval uses SFT_data/test, ICL uses SFT_data/train
    rag_mode: str = "none",
    batch_size=16,
    max_workers=8,
    temperature: float = 1.0,
):
    """
    When is_full is False:
        eval_examples = SFT_data/{city}_{op}_test.json
        icl_examples  = SFT_data/{city}_{op}_train.json (fallback to eval_examples if missing)

    When is_full is True:
        eval_examples = {city}_{op}_test.json
        icl_examples  = eval_examples
    """
    # ---- load eval_examples ----
    if is_full:
        example_path = f"{city_name}_{perturb_op}_test.json"
    else:
        example_path = os.path.join("Dataset", f"{city_name}_{perturb_op}_test.json")

    with open(example_path, "r", encoding="utf-8") as f:
        eval_examples = json.load(f)

    # ---- load icl_examples (ICL pool, prefer *_train.json) ----
    if is_full:
        icl_examples = eval_examples
    else:
        icl_path = os.path.join("Dataset", f"{city_name}_{perturb_op}_train.json")
        if os.path.exists(icl_path):
            with open(icl_path, "r", encoding="utf-8") as f:
                icl_examples = json.load(f)
        else:
            print(f"[WARN] ICL train file not found: {icl_path}, fallback to eval_examples")
            icl_examples = eval_examples

    valid_icl_pool = list(icl_examples.keys())
    traj_set = sorted(eval_examples.keys(), key=lambda x: int(x))

    # ---- base prompt ----
    if perturb_op == "DELETE":
        base_prompt = benchmark_prompts.prompt_add
    elif perturb_op == "REPLACE":
        base_prompt = benchmark_prompts.prompt_replace
    elif perturb_op == "ADD":
        base_prompt = benchmark_prompts.prompt_delete
    else:
        raise ValueError(f"Unknown perturb_op={perturb_op}")

    temp_tag = str(temperature).replace(".", "p")
    temp_tag = f"t{temp_tag}"

    tag_parts = []
    if is_think:
        tag_parts.append("think")
    if rag_mode != "none":
        tag_parts.append(rag_mode)
    tag_parts.append(temp_tag)
    tag_parts.append(str(icl_num))
    tag = "_".join(tag_parts)

    if is_full:
        out_path = (
            f"results/{city_name}/{perturb_op}/{model_name}/"
            f"{model_name}_{tag}_example.json"
        )
    else:
        out_path = (
            f"SFT_results/{city_name}/{perturb_op}/{model_name}/"
            f"{model_name}_{tag}_example.json"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- resume ----
    output_records = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_records = json.load(f)
            print(f"[Resume] loaded {len(output_records)} finished samples from {out_path}")
        except Exception as e:
            print(f"[Resume] cannot load existing out_path, start fresh. err={e}")
            output_records = {}

    done_sids = set(output_records.keys())
    traj_set = [sid for sid in traj_set if sid not in done_sids]
    print(f"[To Run] {city_name}-{perturb_op}, rag_mode={rag_mode}, icl={icl_num}, "
          f"temp={temperature}, remaining {len(traj_set)} samples")

    if not traj_set:
        print("Nothing to run. Exit.")
        return out_path

    # ---- batch ----
    for b_idx, batch in enumerate(chunked(traj_set, batch_size), 1):
        print(f"\n[Batch {b_idx}] size={len(batch)}")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    safe_run_one_sid,
                    sid, eval_examples, icl_examples, valid_icl_pool, base_prompt,
                    model_name, icl_num, is_think, rag_mode,
                    temperature,
                )
                for sid in batch
            ]

            for fut in as_completed(futs):
                sid, rec = fut.result()
                output_records[sid] = rec
                atomic_dump_json(out_path, output_records)

        atomic_dump_json(out_path, output_records)

    print(f"\nDone. Saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    perturb_op_set = ["ADD", "DELETE", "REPLACE"]
    model_name = "o4-mini"
    is_think = True
    is_full = False
    temperature = 1

    rag_settings = [
        ("none", 0),
        ("none", 3),
        # ("hint", 3),
        # ("emd_qwen3_8b", 3),
        # ("emd_azure", 3),
        # ("emd_kalm_gemma3", 3),
    ]
    city_set = ["Melb", "Toro", "Florence"]
    for city_name in city_set:
        for perturb_op in perturb_op_set:
            for rag_mode, icl_num in rag_settings:
                batch_parallel_run(
                    city_name=city_name,
                    perturb_op=perturb_op,
                    icl_num=icl_num,
                    model_name=model_name,
                    is_think=is_think,
                    is_full=is_full,
                    rag_mode=rag_mode,
                    batch_size=12,
                    max_workers=1,
                    temperature=temperature,
                )

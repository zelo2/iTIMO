# -*- coding: utf-8 -*-
import os, json, re, random, time, threading, tempfile, shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from benchmark import benchmark_prompts
from pathlib import Path
from api_key import api_key

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key.deepseek_api_key,
)

# ================== RAG 模式 -> example.json 字段名 ==================
RAG_FIELD_MAP = {
    "none": None,
    "hint": "rec_exmaples",              # 注意拼写
    "emd_qwen3_8b": "rec_examples_qwen3_8b",
    "emd_azure": "rec_examples_gpt_text_large",
    "emd_kalm_gemma3": "rec_examples_kalm_gemma3",
}

SFT_DIR = "Dataset"  # 你的 SFT json 目录

# ======================================================
# 0) Windows-safe atomic dump
# ======================================================
_write_lock = threading.Lock()

def atomic_dump_json(path: str, obj: dict, retries: int = 10, base_sleep: float = 0.3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent)
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        for k in range(retries):
            try:
                os.replace(tmp_name, path)
                return
            except PermissionError:
                time.sleep(base_sleep * (k + 1))

        try:
            if path.exists():
                path.unlink()
            shutil.move(tmp_name, path)
            return
        except Exception:
            raise
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass


def safe_run_one_sid(*args, **kwargs):
    """
    args:
        0: sid
        1: eval_examples   （当前要跑的 split：SFT test 或 full）
        2: icl_examples    （ICL 池：通常是 train）
    """
    sid = args[0]
    eval_examples = args[1]
    try:
        return run_one_sid(*args, **kwargs)
    except Exception as e:
        return sid, {
            "response": "",
            "label": eval_examples[sid]["example_output"],
            "error": repr(e)
        }

# ======================================================
# 1) 全局 TPM 限流器
# ======================================================
TPM_LIMIT = 300000
WINDOW_SEC = 60

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
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)

def chunked(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# ======================================================
# 3) 构建单条 sid messages（eval_examples + icl_examples）
# ======================================================
def build_messages_for_sid(
    sid: str,
    eval_examples: dict,      # 当前评估集（SFT test 或 full）
    icl_examples: dict,       # ICL few-shot 池（通常是 train）
    valid_icl_pool,           # list[str] -> icl_examples 的键
    base_prompt,
    icl_num: int = 0,
    is_think: bool = False,
    rag_mode: str = "none",
):
    """
    rag_mode:
        "none"            : 不用 RAG，随机 few-shots（从 valid_icl_pool 中选）
        "hint"            : 用 rec_exmaples（eval_examples[sid] 上的字段）
        "emd_qwen3_8b"    : 用 rec_examples_qwen3_8b
        "emd_azure"       : 用 rec_examples_gpt_text_large
        "emd_kalm_gemma3" : 用 rec_examples_kalm_gemma3
    """
    input_ex = eval_examples[sid]["example_input"]
    label_ex = eval_examples[sid]["example_output"]

    example_desc = ""
    if icl_num > 0:
        field_name = RAG_FIELD_MAP.get(rag_mode, None)

        # 构造局部 RNG，使同 (sid, rag_mode) 可复现
        rng_seed = hash((sid, rag_mode)) & 0xffffffff
        rng = random.Random(rng_seed)

        if field_name is None:
            # 不用 RAG：在 few-shot 库中随机
            cand_pool = [k for k in valid_icl_pool if k != sid]
        else:
            # 用某种 RAG：eval_examples[sid] 里有 rec_* 字段，值是 train 的 id
            rec_ids = eval_examples[sid].get(field_name) or []
            # 过滤掉不在 icl_examples 里的样本，避免 KeyError
            cand_pool = [
                str(x) for x in rec_ids
                if str(x) in icl_examples and str(x) != sid
            ]

            if not cand_pool:
                print(f"[WARN] sid={sid} rag_mode={rag_mode} 没有有效 {field_name} 候选，降级为随机 few-shots")
                cand_pool = [k for k in valid_icl_pool if k != sid]

        if len(cand_pool) <= icl_num:
            icl_pool = cand_pool[:]
        else:
            icl_pool = rng.sample(cand_pool, k=icl_num)

        # few-shot 内容一律从 icl_examples（train）里取
        for j, ex_sid in enumerate(icl_pool, 1):
            ex_sid_str = str(ex_sid)
            if ex_sid_str not in icl_examples:
                continue
            ex_item = icl_examples[ex_sid_str]
            ex_in = ex_item["example_input"]
            ex_out = ex_item["example_output"]
            example_desc += (
                f"Example #{j} Input:\n{json.dumps(ex_in, ensure_ascii=False)}\n"
                f"Example #{j} Output:\n{json.dumps(ex_out, ensure_ascii=False)}\n"
            )

    if is_think:
        system_prompt = base_prompt + "\n" + example_desc + "\n" + "[End of Examples]"
    else:
        system_prompt = (
            "/no_think\n"
            "Only output the final JSON; never print <think> tags or intermediate reasoning.\n"
            + base_prompt + "\n" + example_desc + "\n" + "[End of Examples]"
        )

    input_prompt = {k: v for k, v in input_ex.items() if k not in {"original itinerary"}}
    user_payload = json.dumps(input_prompt, ensure_ascii=False)
    user_content = "[Input]\n" + user_payload + "\n[End of Input]"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages, label_ex, system_prompt, user_content

# ======================================================
# 4) 单条 sid 调用
# ======================================================
def run_one_sid(
    sid: str,
    eval_examples: dict,
    icl_examples: dict,
    valid_icl_pool,
    base_prompt,
    model_name,
    icl_num,
    is_think,
    rag_mode: str = "none",
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
        temperature=0,
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
# 5) 主入口：batch 并发 + 断点续跑（仅 is_full 判定）
# ======================================================
def batch_parallel_run(
    city_name="Melb",
    perturb_op="DELETE",
    icl_num=3,
    model_name="deepseek-chat",
    is_think=False,
    is_full=False,
    rag_mode: str = "none",
    batch_size=32,
    max_workers=4,
    checkpoint_every=1,
):
    """
    is_full = False:
        eval_examples = SFT_data/{city}_{op}_test.json
        icl_examples  = SFT_data/{city}_{op}_train.json （若无则退回 eval_examples）

    is_full = True:
        eval_examples = {city}_{op}_test.json
        icl_examples  = eval_examples
    """

    # ---- load eval_examples ----
    if is_full:
        eval_path = f"{city_name}_{perturb_op}_test.json"
    else:
        eval_path = os.path.join(SFT_DIR, f"{city_name}_{perturb_op}_test.json")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_examples = json.load(f)

    # ---- load icl_examples（few-shot / RAG 池）----
    if is_full:
        icl_examples = eval_examples
    else:
        icl_path = os.path.join(SFT_DIR, f"{city_name}_{perturb_op}_train.json")
        if os.path.exists(icl_path):
            with open(icl_path, "r", encoding="utf-8") as f:
                icl_examples = json.load(f)
        else:
            print(f"[WARN] train ICL file not found: {icl_path}, fallback to eval_examples")
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

    # ---- 构造输出文件名：包含 think / rag_mode / icl_num ----
    tag_parts = []
    if is_think:
        tag_parts.append("think")
    if rag_mode != "none":
        tag_parts.append(rag_mode)
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

    # ---- Resume ----
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
    print(f"[To Run] {city_name}-{perturb_op}, rag_mode={rag_mode}, icl={icl_num}, remaining {len(traj_set)} samples")

    if len(traj_set) == 0:
        print("Nothing to run. Exit.")
        return out_path

    # ---- batch loop ----
    for b_idx, batch in enumerate(chunked(traj_set, batch_size), 1):
        print(f"\n[Batch {b_idx}] size={len(batch)}")

        batch_counter = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    safe_run_one_sid,
                    sid, eval_examples, icl_examples, valid_icl_pool, base_prompt,
                    model_name, icl_num, is_think, rag_mode
                )
                for sid in batch
            ]

            for fut in as_completed(futs):
                sid, rec = fut.result()
                output_records[sid] = rec
                batch_counter += 1

                if checkpoint_every > 0 and (batch_counter % checkpoint_every == 0):
                    with _write_lock:
                        atomic_dump_json(out_path, output_records)

        with _write_lock:
            atomic_dump_json(out_path, output_records)

    print(f"\nDone. Saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    perturb_op_set = ["ADD", "DELETE", "REPLACE"]
    model_name = "deepseek-reasoner"
    is_think = True
    is_full = False   # 和 Azure/LM Studio 脚本保持一致

    # (rag_mode, icl_num)
    rag_settings = [
        ("none", 0),              # 0-shot baseline
        ("none", 3),              # few-shot, no RAG
        # ("hint", 3),              # 原始 hint-RAG (rec_exmaples)
        # ("emd_qwen3_8b", 3),      # Qwen embedding RAG
        # ("emd_azure", 3),         # GPT embedding RAG
        # ("emd_kalm_gemma3", 3),   # KaLM-gemma3 RAG
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
                    batch_size=24,
                    max_workers=6,
                    checkpoint_every=10,
                )

# -*- coding: utf-8 -*-
import json
from json import JSONDecodeError
import os
import dataset
from openai import OpenAI
from uni_perturbation import candidate_generation, haversine_distance
import pandas as pd
import numpy as np
import csv
from urllib.parse import unquote_plus
import random
from benchmark import benchmark_prompts
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    timeout=1200,
    max_retries=0
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# ================== RAG modes -> example.json field names ==================
RAG_FIELD_MAP = {
    "none": None,
    "hint": "rec_exmaples",  # keep spelling as in the data
    "emd_qwen3_8b": "rec_examples_qwen3_8b",
    "emd_azure": "rec_examples_gpt_text_large",
    "emd_kalm_gemma3": "rec_examples_kalm_gemma3",
}


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openaiAPIcall(**kwargs):
    return client.chat.completions.create(**kwargs)


def new_data_open(city_name):
    if city_name in ['Melb', 'Toro']:
        train_data = pd.read_csv(f'../dataset/{city_name}/train.csv')
        test_data = pd.read_csv(f'../dataset/{city_name}/test.csv')
        val_data = pd.read_csv(f'../dataset/{city_name}/val.csv')

        data = pd.concat([train_data, test_data])
        data = pd.concat([data, val_data])
        data = data.reset_index(drop=True)

        return data

    if city_name in ['Florence', 'Pisa', 'Rome']:
        data = pd.read_csv(
            f"../LearNext-DATASET/{city_name}/Trajectories-{city_name.upper()}-final2.csv",
            encoding='utf-8'
        )
        poi_info = pd.read_csv(
            f"../LearNext-DATASET/{city_name}/PoIs-{city_name.upper()}-final.csv",
            encoding='utf-8'
        )
        return data, poi_info


def geo_info_collect(city_name):
    if city_name not in ['Florence', 'Pisa', 'Rome']:
        if city_name in ['Buda', 'Delh', 'Edin', 'Glas', 'Osak', 'Pert', 'Toro', 'Vien']:
            path = f'../data-ijcai15/poiList-ijcai15/POI-{city_name}.csv'
        elif city_name == 'Melb':
            path = f'../data-cikm16/POI-{city_name}.csv'
        poi_info = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            count = 0
            for row in reader:
                row = row[0].split(';')
                if count == 0:
                    columns = row
                else:
                    row[0] = eval(row[0])
                    poi_info.append(row)
                count += 1
        poi_info = pd.DataFrame(poi_info, columns=columns)
        poi_info['poiName'] = poi_info['poiName'].astype(str).apply(unquote_plus)
        poi_info['poiID'] -= 1
        dis_mat = np.zeros([len(poi_info), len(poi_info)])
        for i in range(len(poi_info)):
            for j in range(len(poi_info)):
                if i < j:
                    cur_lat = eval(poi_info.loc[i, 'lat'])
                    cur_long = eval(poi_info.loc[i, 'long'])

                    next_lat = eval(poi_info.loc[j, 'lat'])
                    next_long = eval(poi_info.loc[j, 'long'])

                    dis_mat[i, j] = haversine_distance(
                        lat1=cur_lat,
                        lat2=next_lat,
                        lon1=cur_long,
                        lon2=next_long
                    )

                    dis_mat[j, i] = dis_mat[i, j]

        return dis_mat, poi_info


def strip_think(text: str) -> str:
    # Remove <think>…</think> (case-insensitive, across lines)
    return re.sub(r"<think>.*?</think>\s*", "", text,
                  flags=re.DOTALL | re.IGNORECASE)


def prompt_test(city_name,
                dis_mark=None,
                llm_seed=-1,
                perturb_op=None,
                icl_num=-1,
                model_name='deepseek-chat',
                is_think=False,
                rag_mode: str = "none",
                is_full: bool = False,
                flush_every: int = 20):
    """
    rag_mode:
        "none"            : no RAG, random few-shots
        "hint"            : use rec_exmaples (from current eval file; ids point to train)
        "emd_qwen3_8b"    : use rec_examples_qwen3_8b
        "emd_azure"       : use rec_examples_gpt_text_large
        "emd_kalm_gemma3" : use rec_examples_kalm_gemma3
    is_full:
        True  -> eval_examples = {city}_{OP}_EXAMPLES.json; ICL also from here
        False -> eval_examples = SFT_data/{city}_{OP}_test.json; ICL from SFT_data/{city}_{OP}_train.json
    """

    print(f"Configurations:\n"
          f"Dataset: {city_name}_{perturb_op}\n"
          f"Model: {model_name}\n"
          f"Is_Thinking: {is_think}\n"
          f"RAG Mode: {rag_mode}\n"
          f"#Examples: {icl_num}\n"
          f"LLM_seed: {llm_seed}\n"
          f"is_full: {is_full}\n")

    # ========= 1. Decide output filename (includes rag_mode) =========
    tag_parts = []
    if is_think:
        tag_parts.append("think")
    if rag_mode != "none":
        tag_parts.append(rag_mode)
    tag_parts.append(str(icl_num))
    tag = "_".join(tag_parts) if tag_parts else str(icl_num)

    if is_full:
        out_root = "results"
    else:
        out_root = "SFT_results"

    out_path = f"{out_root}/{city_name}/{perturb_op}/{model_name}/{model_name}_{tag}_example.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ========= 2. Load existing results (resume) =========
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_records = json.load(f)
            print(f"[RESUME] Loaded existing results from {out_path}, "
                  f"{len(output_records)} trajectories already done.")
        except Exception as e:
            print(f"[WARN] Failed to load existing results ({e}), start from scratch.")
            output_records = {}
    else:
        output_records = {}

    done_ids = set(output_records.keys())  # JSON keys are strings

    # ========= 3. Data prep =========
    # 3.1 choose eval_examples and icl_examples
    if city_name in ['Melb', 'Toro']:
        if city_name == 'Melb':
            cat_mark = 'subTheme'
        else:
            cat_mark = 'theme'

        # eval file
        if is_full:
            eval_path = f'{city_name}_{perturb_op.upper()}_examples.json'
        else:
            eval_path = os.path.join("Dataset", f"{city_name}_{perturb_op}_test.json")

        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_examples = json.load(f)

        # icl file (few-shot / RAG pool)
        if is_full:
            icl_examples = eval_examples
        else:
            icl_path = os.path.join("Dataset", f"{city_name}_{perturb_op}_train.json")
            if os.path.exists(icl_path):
                with open(icl_path, 'r', encoding='utf-8') as f:
                    icl_examples = json.load(f)
            else:
                print(f"[WARN] ICL train file not found: {icl_path}, fallback to eval_examples")
                icl_examples = eval_examples

        # Traj/popLevel logic unchanged; does not affect RAG
        traj = new_data_open(city_name)
        traj['poiID'] -= 1
        _, poi_info = geo_info_collect(city_name)
        pop = traj['poiFreq'].unique()
        q1 = np.quantile(pop, 1 / 3)
        q2 = np.quantile(pop, 2 / 3)

        for j in range(len(traj)):
            traj.loc[j, 'startTime'] = eval(traj.loc[j, 'dateTaken'])[0]
            traj.loc[j, '#photo'] = len(eval(traj.loc[j, 'dateTaken']))
            if traj.loc[j, 'poiFreq'] <= q1:
                traj.loc[j, 'popLevel'] = 'low'
            elif traj.loc[j, 'poiFreq'] >= q2:
                traj.loc[j, 'popLevel'] = 'high'
            else:
                traj.loc[j, 'popLevel'] = 'medium'

    else:
        if city_name == 'Florence':
            cat_mark = 'theme'

        if is_full:
            eval_path = f'{city_name}_{perturb_op.upper()}_examples.json'
        else:
            eval_path = os.path.join("Dataset", f"{city_name}_{perturb_op}_test.json")

        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_examples = json.load(f)

        if is_full:
            icl_examples = eval_examples
        else:
            icl_path = os.path.join("Dataset", f"{city_name}_{perturb_op}_train.json")
            if os.path.exists(icl_path):
                with open(icl_path, 'r', encoding='utf-8') as f:
                    icl_examples = json.load(f)
            else:
                print(f"[WARN] ICL train file not found: {icl_path}, fallback to eval_examples")
                icl_examples = eval_examples

        traj, poi_info = new_data_open(city_name)
        poi_info = poi_info.rename(columns={'PoIName_Italian': 'poiName'})
        traj['poiID'] -= 1
        poi_info['poiID'] -= 1

        pop = traj['poiFreq'].unique()
        q1 = np.quantile(pop, 1 / 3)
        q2 = np.quantile(pop, 2 / 3)

        for j in range(len(traj)):
            if traj.loc[j, 'poiFreq'] <= q1:
                traj.loc[j, 'popLevel'] = 'low'
            elif traj.loc[j, 'poiFreq'] >= q2:
                traj.loc[j, 'popLevel'] = 'high'
            else:
                traj.loc[j, 'popLevel'] = 'medium'

    # Few-shot / RAG pool ids
    valid_icl_pool = list(icl_examples.keys())

    data = dataset.Prompt_Dataset(city_name=city_name, perturb_op=perturb_op)

    # Current run samples come from eval_examples (default test)
    traj_set = eval_examples.keys()

    newly_added = 0

    # ========= 4. Main loop =========
    for sid in sorted(traj_set, key=lambda x: str(x)):
        sid_str = str(sid)

        # Resume: skip already processed ids
        if sid_str in done_ids:
            continue

        ex = eval_examples[sid_str]
        input_ex = ex['example_input']
        label = ex['example_output']

        # ==============================In-context Learning==================================
        example_desc = ""
        if icl_num > 0:
            field_name = RAG_FIELD_MAP.get(rag_mode, None)

            # Local RNG to make few-shots deterministic for (city, op, seed, sid, rag_mode)
            rng = random.Random(
                hash((city_name, perturb_op, llm_seed, sid_str, rag_mode)) & 0xffffffff
            )

            if field_name is None:
                # No RAG: random few-shots from ICL pool (usually train)
                cand_pool = [k for k in valid_icl_pool if k != sid_str]
            else:
                # RAG: use rec_* ids from eval_examples[sid] (ids point to train)
                rec_ids = ex.get(field_name) or []
                cand_pool = [
                    str(x) for x in rec_ids
                    if str(x) in icl_examples and str(x) != sid_str
                ]

                # If no valid candidates, fall back to random few-shots
                if not cand_pool:
                    print(f"[WARN] {sid_str} rag_mode={rag_mode} missing valid {field_name}, fallback to random few-shots")
                    cand_pool = [k for k in valid_icl_pool if k != sid_str]

            # Actual few-shot list icl_pool
            if len(cand_pool) <= icl_num:
                icl_pool = cand_pool[:]  # use all if not enough
            else:
                icl_pool = rng.sample(cand_pool, k=icl_num)

            # Build few-shot text: always from icl_examples (usually train)
            for j, sid2 in enumerate(icl_pool, 1):
                sid2_str = str(sid2)
                if sid2_str not in icl_examples:
                    continue
                ex2 = icl_examples[sid2_str]
                example_input = ex2['example_input']
                example_output = ex2['example_output']
                example_desc += (
                    f"Example #{j} Input:\n{json.dumps(example_input, ensure_ascii=False)}\n"
                    f"Example #{j} Output:\n{json.dumps(example_output, ensure_ascii=False)}\n"
                )
        # ==============================In-context Learning==================================

        if perturb_op == 'DELETE':
            base_prompt = benchmark_prompts.prompt_add
        elif perturb_op == 'REPLACE':
            base_prompt = benchmark_prompts.prompt_replace
        elif perturb_op == 'ADD':
            base_prompt = benchmark_prompts.prompt_delete
        else:
            raise ValueError(f"Unknown perturb_op: {perturb_op}")

        if is_think:
            system_prompt = (base_prompt + "\n" + example_desc + "\n" + "[End of Examples]")
        else:
            system_prompt = (
                "/no_think\n"
                "Only output the final JSON; never print <think> tags or intermediate reasoning.\n"
                + (base_prompt + "\n" + example_desc + "\n" + "[End of Examples]")
            )

        input_prompt = {k: v for k, v in input_ex.items() if k not in {'original itinerary'}}

        user_payload = json.dumps(input_prompt, ensure_ascii=False)
        user_content = (
            "[Input]\n"
            f"{user_payload}\n"
            "[End of Input]"
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]

        response = openaiAPIcall(
            model=model_name,
            messages=messages,
            temperature=0,
            seed=llm_seed
        )

        if is_think:
            result = strip_think(response.choices[0].message.content)
        else:
            result = response.choices[0].message.content

        # Save result
        output_records[sid_str] = {'response': result, 'label': label}
        newly_added += 1

        # Periodic flush
        if newly_added % flush_every == 0:
            print(f"[FLUSH] {newly_added} new results, flushing to {out_path}")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output_records, f, indent=4, ensure_ascii=False)

    # Final write
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_records, f, indent=4, ensure_ascii=False)
    print(f"[DONE] Saved all results to {out_path}")

    return 0, 0, 0


def multi_round_test(city_name,
                     dis_mark,
                     llm_seeds,
                     icl_num_set,
                     perturb_op_set,
                     model_name,
                     is_think,
                     rag_mode_list,
                     is_full: bool = False,
                     max_workers=2):
    """
    rag_mode_list: one-to-one with icl_num_set, e.g.:
        icl_num_set = [0, 3, 3, 3]
        rag_mode_list = ["none", "none", "hint", "emd_qwen3_8b"]
    is_full: forwarded to prompt_test
    """
    assert len(icl_num_set) == len(rag_mode_list), "icl_num_set and rag_mode_list must have same length"

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for seed in llm_seeds:
            for perturb_op in perturb_op_set:
                for idx, icl in enumerate(icl_num_set):
                    rag_mode = rag_mode_list[idx]
                    fut = ex.submit(
                        prompt_test,
                        city_name,
                        dis_mark,
                        seed,
                        perturb_op,
                        icl,
                        model_name,
                        is_think,
                        rag_mode,
                        is_full
                    )
                    fut.seed, fut.icl, fut.rag_mode = seed, icl, rag_mode
                    futs.append(fut)

        for fut in as_completed(futs):
            try:
                res = fut.result()
                results[(fut.seed, fut.icl, fut.rag_mode)] = res
            except Exception as e:
                print(f"[ERROR] seed={fut.seed}, icl={fut.icl}, rag={fut.rag_mode}: {e}")
                results[(fut.seed, fut.icl, fut.rag_mode)] = (0, 0, 0)


if __name__ == '__main__':
    city_set = ['Melb', 'Toro', 'Florence']
    llm_seeds = [i for i in range(1)]

    model_name = "microsoft/phi-4-reasoning-plus"
    is_think = False
    is_full = False   # align with Azure script: False -> eval uses SFT_data/test, ICL uses SFT_data/train

    for city_name in city_set:
        if city_name == 'Melb':
            dis_mark = {'low': 0.3, 'high': 0.77}
        elif city_name == 'Toro':
            dis_mark = {'low': 0.91, 'high': 1.74}
        elif city_name == 'Florence':
            dis_mark = {'low': 0.20, 'high': 0.45}
        else:
            dis_mark = None

        perturb_op_set = ['ADD', 'DELETE', 'REPLACE']

        #  0-shot: icl=0, rag=none
        #  few-shot: icl=3, rag=none
        #  three embedding RAG settings
        icl_num_set = [0, 3, 3, 3, 3, 3]
        rag_mode_list = ["none", "none", "hint", "emd_qwen3_8b", "emd_azure", "emd_kalm_gemma3"]

        multi_round_test(city_name, dis_mark, llm_seeds,
                         icl_num_set, perturb_op_set,
                         model_name, is_think,
                         rag_mode_list, is_full=is_full, max_workers=1)

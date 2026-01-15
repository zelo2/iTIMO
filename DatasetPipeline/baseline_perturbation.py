import random
import time
import numpy as np
import json
import sys
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import pandas as pd
import csv
from itertools import groupby
import datetime
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DatasetPipeline.template import baseline_prompts
from urllib.parse import unquote_plus

deepseek_api_key = "YOUR API KEY"
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=deepseek_api_key)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openaiAPIcall(**kwargs):
    return client.chat.completions.create(**kwargs)


def unix_time_convert(unix_timestamp):
    datatime_object = datetime.datetime.fromtimestamp(unix_timestamp, tz=ZoneInfo("Australia/Melbourne"))
    formatted_time = datatime_object.strftime('%H:%M')

    return formatted_time


def new_data_open(city_name):
    if city_name in ['Melb', 'Toro']:
        base_dir = REPO_ROOT / "data4perturb" / city_name
        train_data = pd.read_csv(base_dir / "train.csv")
        test_data = pd.read_csv(base_dir / "test.csv")
        val_data = pd.read_csv(base_dir / "val.csv")

        data = pd.concat([train_data, test_data])
        data = pd.concat([data, val_data])
        data = data.reset_index(drop=True)

    if city_name == 'Florence':
        # WARNING: MUST ENCODE it via "utf-8"
        base_path = REPO_ROOT / "data4perturb" / "Florence"
        data = pd.read_csv(base_path / f"Trajectories-{city_name.upper()}.csv",
                           encoding='utf-8')
        poi_info = pd.read_csv(base_path / f"PoIs-{city_name.upper()}.csv",
                               encoding='utf-8')
        # cat_info = pd.read_csv(f"{base_path}/Categories-{city_name.upper()}.csv")

        return data, poi_info

    return data


def remove_consecutive_duplicates(lst):
    """Remove consecutive duplicate elements"""
    return [key for key, _ in groupby(lst)]


def haversine_distance(lat1, lon1, lat2, lon2):
    # lat1 = eval(lat1)
    # lon1 = eval(lon1)
    # lat2 = eval(lat2)
    # lon2 = eval(lon2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return distance


def geo_info_collect(city_name):
    if city_name != 'Florence':
        if city_name in ['Buda', 'Delh', 'Edin', 'Glas', 'Osak', 'Pert', 'Toro', 'Vien']:
            path = REPO_ROOT / "og_dataset" / "data-ijcai15" / "poiList-ijcai15" / f"POI-{city_name}.csv"
        elif city_name == 'Melb':
            path = REPO_ROOT / "og_dataset" / "data-cikm16" / f"POI-{city_name}.csv"
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

                    dis_mat[i, j] = haversine_distance(lat1=cur_lat,
                                                       lat2=next_lat,
                                                       lon1=cur_long,
                                                       lon2=next_long)

                    dis_mat[j, i] = dis_mat[i, j]

        return dis_mat, poi_info


def candidate_generation(traj, poi_info, og_traj, cat_mark=None):
    candidate_pool = list(set(traj['poiID'].unique()) - set(og_traj))
    candidate_desc = []
    for i in candidate_pool:
        cur_poi_info = poi_info[poi_info['poiID'] == i]
        poplevel = traj[traj['poiID'] == i]['popLevel'].unique()
        visit_behavior = [cur_poi_info['poiName'].item(), cur_poi_info[cat_mark].item(),
                          cur_poi_info['long'].item(),
                          cur_poi_info['lat'].item(), poplevel[0]]

        candidate_desc.append(visit_behavior)
    return candidate_desc


def iTIMO_perturbation(city_name, op=None, think="R"):
    if not op:
        print("Error")
        return "Error"

    if op == 'ADD':
        used_mark = 'inserted'
    elif op == 'REPLACE':
        used_mark = 'after'
    elif op == 'DELETE':
        used_mark = 'deleted'

    if city_name in ['Melb', 'Toro']:
        if city_name == 'Melb':
            # Path Initialization
            memory_path = f"data-cikm16/perturbation_data/{op}/memory_buffer_{op}_{think}.json"
            token_usage_path = f"data-cikm16/perturbation_data/{op}/token_usage_{op}_{think}.json"
            result_path = f"data-cikm16/perturbation_data/Melbourne_{op}_{think}.json"
            infer_time_path = f"data-cikm16/perturbation_data/{op}/infer_time_{op}_{think}.json"
            cat_mark = 'subTheme'
            dis_mark = {'low': 0.3, 'medium': 0.77}
        elif city_name == 'Toro':
            memory_path = f"data-ijcai15/Toro/baseline_samples/{op}/memory_buffer_{op}_{think}.json"
            token_usage_path = f"data-ijcai15/Toro/baseline_samples/{op}/token_usage_{op}_{think}.json"
            result_path = f"data-ijcai15/Toro/baseline_samples/{op}/Toronto_{op}_{think}.json"
            infer_time_path = f"data-ijcai15/Toro/baseline_samples/{op}/infer_time_{op}_{think}.json"
            cat_mark = 'theme'
            dis_mark = {'low': 0.91, 'medium': 1.74}
            # _spatial_class Function

        traj = new_data_open(city_name)
        traj['poiID'] -= 1
        dis_mat, poi_info = geo_info_collect(city_name)

        '''Data Preprocessing'''
        pop = traj['poiFreq'].unique()
        q1 = np.quantile(pop, 1 / 3)
        q2 = np.quantile(pop, 2 / 3)

        for j in range(len(traj)):
            # print(traj.loc[j, 'dateTaken'], type(traj.loc[j, 'dateTaken']))
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
            memory_path = f"LearNext-DATASET/Florence/baseline_samples/{op}/memory_buffer_{op}.json"
            token_usage_path = f"LearNext-DATASET/Florence/baseline_samples/{op}/token_usage_{op}.json"
            result_path = f"LearNext-DATASET/Florence/baseline_samples/{op}/Florence_{op}.json"
            infer_time_path = f"LearNext-DATASET/Florence/baseline_samples/{op}/infer_time_{op}.json"
            cat_mark = 'theme'
            dis_mark = {'low': 0.20, 'medium': 0.45}
            # _spatial_class Function

        traj, poi_info = new_data_open(city_name)
        poi_info = poi_info.rename(columns={'PoIName_Italian': 'poiName'})
        traj['poiID'] -= 1
        poi_info['poiID'] -= 1
        '''Data Preprocessing'''
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
    # d1, d2 = spatial_distance_classification(traj, dis_mat)
    # print(d1, d2)
    traj_set = traj['seqID'].unique()
    traj_set = traj_set.tolist()

    # target_intent_count = np.random.default_rng(2025).integers(1, 4, size=len(traj_set))

    # random.seed(2025)
    # traj_set = random.sample(traj_set, 50)
    '''seed 2025'''
    # For Toronto
    if city_name == 'Toro':
        '''seed 2025-Toronto'''
        traj_set = [3881, 807, 5530, 5640, 1378, 624, 58, 4132, 4198, 5073, 1938,
                    744, 4308, 4493, 852, 634, 923, 1865, 4756, 683, 1135, 453, 2686,
                    5959, 745, 1592, 1525, 380, 1934, 5214, 4451, 5324, 3898, 71, 472,
                    623, 898, 2198, 1620, 1802, 929, 2036, 904, 515, 920, 1066, 2604,
                    4004, 813, 99]
    elif city_name == 'Melb':
        '''seed 2025-Melbourne'''
        traj_set = [3454, 605, 1043, 3185, 1021, 3334, 113, 2433, 2477, 3629, 1468, 574, 2586, 2700,
                    660, 382, 757, 3184,
                    1288, 2811, 472, 3950, 298, 3447, 3256, 575, 3866, 1229, 3588, 1136, 295, 45,
                    1397, 3935, 779, 2691,
                    3046, 2321, 3282, 329, 3799, 712, 3426, 1235, 1278, 798, 1578, 3865, 336, 731]
    elif city_name == 'Florence':
        '''seed 2025-Florence'''
        traj_set = ['3332', '481', '3', '5125', '6', '5759', '997',
                    '3133', '1', '2206', '5924', '2229', '3409', '1363',
                    '395', '2267', '2377', '582', '248', '16', '4662',
                    '1229', '5974', '591', '2444', '291', '3688', '154',
                    '3309', '17', '400', '10', '5500', '1120',
                    '4674', '15', '147', '4276', '1332', '5968',
                    '4763', '5719', '4733', '14', '2350', '5619',
                    '2734', '2112', '3083', '5562']

    target_intent_count = [3, 2, 3, 3, 2, 3, 1, 3, 1, 3, 1, 3, 2, 1, 1,
                           3, 1, 3, 1, 3, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1,
                           2, 1, 1, 2, 3, 1, 2,
                           1, 3, 2, 1, 2, 3, 1, 1, 2, 2, 2, 2, 1]

    mark = False
    result = {}
    infer_time = {}
    token_usage = {}

    # with open(result_path, encoding='utf-8') as f:
    #     result = json.load(f)

    for current_round, traj_id in enumerate(traj_set):
        '''Initialize Memory'''
        if traj_id != -1:  # If problematic traj_ids (e.g., 2138/2492), regenerate (REPLACE)
            mark = True
        if mark:
            cur_traj = traj[traj['seqID'] == traj_id].sort_values(by='startTime', ascending=True)
            if "_" in str(traj_id) and len(cur_traj) < 21:
                pretend_length = 21 - len(cur_traj)
                pretend_chunk_id = cur_traj['chunk_index'].unique()[0] - 1
                pretend_traj_id = f"{cur_traj['seqID_old'].unique()[0]}_{pretend_chunk_id}"
                pretend_traj = traj[traj['seqID'] == pretend_traj_id].sort_values(by='startTime',
                                                                                  ascending=True).reset_index(drop=True)
                pretend_traj = pretend_traj.tail(pretend_length)
                cur_traj = pd.concat([pretend_traj, cur_traj]).reset_index(drop=True)
            cur_traj = cur_traj['poiID'].values.tolist()

            og_traj_desc = []
            for count, i in enumerate(cur_traj):
                cur_poi_info = poi_info[poi_info['poiID'] == i]
                poplevel = traj[traj['poiID'] == i]['popLevel'].unique()
                visit_behavior = [cur_poi_info['poiName'].item(), cur_poi_info[cat_mark].item(),
                                  cur_poi_info['long'].item(),
                                  cur_poi_info['lat'].item(), poplevel[0]]
                og_traj_desc.append(visit_behavior)

            if op == 'DELETE':
                input_prompt = {'Original Itinerary': og_traj_desc,
                                'Target Intent Count': target_intent_count[current_round],
                                'Candidate Intents': ["Spatial distance disruption", "Popularity disruption",
                                                      "Categorical diversity disruption"]
                                }

            else:
                candidate_desc = candidate_generation(traj, poi_info, cur_traj, cat_mark=cat_mark)
                if city_name != 'Toro':
                    candidate_desc = random.sample(candidate_desc, 50)
                input_prompt = {'Original Itinerary': og_traj_desc,
                                'Target Intent Count': target_intent_count[current_round],
                                'Candidate Intents': ["Spatial distance disruption", "Popularity disruption",
                                                      "Categorical diversity disruption"],
                                'Candidate POIs': candidate_desc}

            if op == 'REPLACE':
                system_prompt = baseline_prompts.replace_prompt_icl.format(dis_mark['low'] * 1000,
                                                                           dis_mark['low'] * 1000,
                                                                           dis_mark['medium'] * 1000,
                                                                           dis_mark['medium'] * 1000)
            elif op == 'ADD':
                system_prompt = baseline_prompts.add_prompt_icl.format(dis_mark['low'] * 1000,
                                                                       dis_mark['low'] * 1000,
                                                                       dis_mark['medium'] * 1000,
                                                                       dis_mark['medium'] * 1000)
            elif op == 'DELETE':
                system_prompt = baseline_prompts.delete_prompt_icl.format(dis_mark['low'] * 1000,
                                                                          dis_mark['low'] * 1000,
                                                                          dis_mark['medium'] * 1000,
                                                                          dis_mark['medium'] * 1000)

            messages = [{'role': 'system', 'content': str(system_prompt)},
                        {'role': 'user', 'content': str(input_prompt)}]

            print(f'TRAJ ID: {traj_id}')
            print(f"Target Intent Count: {target_intent_count[current_round]}")

            start_time = time.time()

            if think == 'R':
                model = 'deepseek-reasoner'
            elif think == 'V':
                model = 'deepseek-chat'
            response = openaiAPIcall(
                model=model,
                messages=messages,
                response_format={
                    'type': 'json_object'
                },
                temperature=0
            )

            end_time = time.time()
            infer_time[traj_id] = end_time - start_time
            token_usage[traj_id] = response.model_dump()["usage"]
            print(f"Infer Time Cost:{end_time - start_time}")
            print(f"Token Cost:{response.model_dump()['usage']}")
            # print(response)
            json_response = json.loads(response.choices[0].message.content)

            with open(infer_time_path, 'w', encoding='utf-8') as f:
                json.dump(infer_time, f, indent=4, ensure_ascii=False)

            with open(token_usage_path, 'w', encoding='utf-8') as f:
                json.dump(token_usage, f, indent=4, ensure_ascii=False)

            print(json_response)

            result[int(traj_id)] = {'seqID': int(traj_id),
                                    'original itinerary': og_traj_desc,
                                    'response': json_response}

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    city_set = ['Toro',
                'Melb',
                'Florence']
    target_city_index = 1
    print("Current City is", city_set[target_city_index])
    # iTIMO_perturbation(city_set[i], op='DELETE', think='R')

    jobs = [
        (city_set[target_city_index], "ADD", "V"),
        # (city_set[target_city_index], "REPLACE", "R"),
        # (city_set[target_city_index], "DELETE", "R")
    ]
    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(iTIMO_perturbation, *args) for args in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())  # add try/except if fault-tolerant collection is needed
    print(results)

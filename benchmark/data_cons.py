import json

import tqdm

import dataset
from openai import OpenAI
from uni_perturbation import candidate_generation, haversine_distance
import pandas as pd
import numpy as np
import csv
from urllib.parse import unquote_plus
import random
from benchmark import benchmark_prompts


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def new_data_open(city_name):
    max_len = -1
    if city_name in ['Melb', 'Toro']:
        train_data = pd.read_csv(f'../dataset/{city_name}/train.csv')
        test_data = pd.read_csv(f'../dataset/{city_name}/test.csv')
        val_data = pd.read_csv(f'../dataset/{city_name}/val.csv')

        data = pd.concat([train_data, test_data])
        data = pd.concat([data, val_data])
        data = data.reset_index(drop=True)

        return data

    if city_name in ['Florence', 'Pisa', 'Rome']:
        # WARNING: MUST ENCODE it via "utf-8"
        data = pd.read_csv(
            f"../LearNext-DATASET/{city_name}/Trajectories-{city_name.upper()}-final2.csv",
            encoding='utf-8'
        )

        poi_info = pd.read_csv(
            f"../LearNext-DATASET/{city_name}/PoIs-{city_name.upper()}-final.csv",
            encoding='utf-8'
        )

        max_length = 21

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

                    dis_mat[i, j] = haversine_distance(lat1=cur_lat,
                                                       lat2=next_lat,
                                                       lon1=cur_long,
                                                       lon2=next_long)

                    dis_mat[j, i] = dis_mat[i, j]

        return dis_mat, poi_info


def example_construction(city_name, traj_id, dis_mark=None, perturb_op=None):
    if city_name in ['Melb', 'Toro']:
        if city_name == 'Melb':
            cat_mark = 'subTheme'
        elif city_name == 'Toro':
            cat_mark = 'theme'

        traj = new_data_open(city_name)
        traj['poiID'] -= 1
        _, poi_info = geo_info_collect(city_name)
        # Data Preprocessing
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
            cat_mark = 'theme'

        traj, poi_info = new_data_open(city_name)
        poi_info = poi_info.rename(columns={'PoIName_Italian': 'poiName'})
        traj['poiID'] -= 1
        poi_info['poiID'] -= 1

        # Data Preprocessing
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

    data = dataset.Prompt_Dataset(city_name=city_name, perturb_op=perturb_op)
    traj_set = data.traj_set

    random.seed(int(traj_id))

    input, label = data.sample(traj_id)
    # print(input, label)

    # Ensure lat/long columns are float before the loop
    poi_info['lat'] = pd.to_numeric(poi_info['lat'], errors='coerce')
    poi_info['long'] = pd.to_numeric(poi_info['long'], errors='coerce')

    og_traj_id = []
    for x in input['need_to_modify itinerary']:
        poi_name = x[0]
        cat = x[1]  # e.g., 'palazzidifirenze'
        lon_val = float(x[2])
        lat_val = float(x[3])

        # First try name + approximate lat/long match (with tolerance)
        cand = poi_info[
            (poi_info['poiName'] == poi_name) &
            (np.isclose(poi_info['lat'], lat_val, atol=1e-8)) &
            (np.isclose(poi_info['long'], lon_val, atol=1e-8))
        ]

        # If empty, fall back to name + category (Florence uses theme; Melb/Toro use theme/subTheme)
        if cand.empty and cat_mark in poi_info.columns:
            cand = poi_info[
                (poi_info['poiName'] == poi_name) &
                (poi_info[cat_mark] == cat)
            ]

        # If still empty, raise for easier debugging of mismatched POIs
        if cand.empty:
            raise ValueError(
                f"POI not found in poi_info: {poi_name} @({lat_val},{lon_val}), cat={cat}"
            )

        # If multiple rows, try narrowing by category; otherwise take first
        if len(cand) > 1 and cat_mark in cand.columns:
            cand2 = cand[cand[cat_mark] == cat]
            if not cand2.empty:
                cand = cand2
        og_traj_id.append(int(cand['poiID'].iloc[0]))

    # Candidate generation + GT alignment starts here
    label_float = None

    if perturb_op != 'ADD':
        # Generate candidates and sample 4
        candidate_desc = candidate_generation(
            traj=traj,
            poi_info=poi_info,
            og_traj=og_traj_id,
            cat_mark=cat_mark
        )
        candidate_desc = random.sample(candidate_desc, 4)

        # Normalize lon/lat types: convert label and candidates to float
        gt = label['poi_label']
        label_float = [gt[0], gt[1], float(gt[2]), float(gt[3]), gt[4]]

        norm_cands = []
        for poi in candidate_desc:
            name, cat, lon, lat, pop = poi
            norm_cands.append([name, cat, float(lon), float(lat), pop])

        candidate_desc = norm_cands
        candidate_desc.append(label_float)

        # Sort then assign cand_id to keep deterministic ordering
        candidate_desc.sort()

        candidate_poi = []
        cand_id_label = -1
        for i, poi in enumerate(candidate_desc):
            candidate_poi.append({'cand_id': i, 'poi': poi})
            # Match using float label to avoid str-vs-float mismatches
            if label_float is not None and poi == label_float:
                cand_id_label = i

        input['Candidate POIs'] = candidate_poi
    else:
        cand_id_label = -1  # not used for ADD

    input['threshold_low'] = str(dis_mark['low']) + 'km'
    input['threshold_high'] = str(dis_mark['high']) + 'km'

    example_input = {k: v for k, v in input.items() if k not in {'original itinerary'}}
    example_output = {}
    if perturb_op == 'DELETE':
        example_output['insert_index'] = label['index_label']
    elif perturb_op == 'REPLACE':
        example_output['replaced_index'] = label['index_label']
    elif perturb_op == 'ADD':
        example_output['removed_index'] = label['index_label']

    if perturb_op != 'ADD':
        example_output['selected_cand_id'] = cand_id_label
        # Keep selected_poi format (lon/lat as str) for LLM output
        example_output['selected_poi'] = label['poi_label']

    return example_input, example_output


def example_gene(city_name, dis_mark=None, perturb_op=None):
    if city_name in ['Melb', 'Toro']:
        if city_name == 'Melb':
            cat_mark = 'subTheme'
        elif city_name == 'Toro':
            cat_mark = 'theme'

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

    data = dataset.Prompt_Dataset(city_name=city_name, perturb_op=perturb_op)
    traj_set = data.traj_set

    example_pool = {}
    error_traj = []   # track traj_id with errors (and optional messages)

    for i in tqdm.tqdm(traj_set):
        try:
            random.seed(int(i))
            input, label = data.sample(i)
            if input['hint'] == '':
                continue
            # ValueError may be raised inside example_construction
            example_input, example_output = example_construction(
                city_name, i, dis_mark, perturb_op
            )
            example_pool[i] = {'example_input': example_input,
                               'example_output': example_output}
        except ValueError as e:
            # Record ValueErrors only; skip and continue
            error_traj.append((str(i), str(e)))
            continue
        except Exception as e:
            # Optionally record unexpected errors to avoid crashing the run
            error_traj.append((str(i), f"{type(e).__name__}: {e}"))
            continue

    example_path = f'{city_name}_{perturb_op.upper()}_examples.json'
    with open(example_path, 'w', encoding='utf-8') as f:
        json.dump(example_pool, f, indent=4, ensure_ascii=False)

    # Print/save error list
    if error_traj:
        ids_only = [t[0] for t in error_traj]
        print(f"[WARN] skipped {len(error_traj)} traj_ids due to errors.")
        print(ids_only)  # if you only want id list
        # optional: write an error file for debugging
        err_path = f'{city_name}_{perturb_op.upper()}_errors.json'
        with open(err_path, 'w', encoding='utf-8') as f:
            json.dump(
                {"errors": [{"traj_id": tid, "message": msg} for tid, msg in error_traj]},
                f,
                ensure_ascii=False,
                indent=2
            )

    return example_pool, error_traj


if __name__ == '__main__':
    city_set = ['Melb', 'Toro', 'Florence']
    city_name = city_set[-1]
    llm_seeds = [i for i in range(1)]
    icl_num_set = [i for i in range(6)]

    if city_name == 'Melb':
        dis_mark = {'low': 0.3, 'high': 0.77}
    elif city_name == 'Toro':
        dis_mark = {'low': 0.91, 'high': 1.74}
    elif city_name == 'Florence':
        dis_mark = {'low': 0.20, 'high': 0.45}
    perturb_op = 'DELETE'
    example_gene(city_name, dis_mark, perturb_op)

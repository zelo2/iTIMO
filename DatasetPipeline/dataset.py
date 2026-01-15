import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DatasetPipeline.position_POI_extraction import extract_change
from Benchmark import benchmark_prompts

class Prompt_Dataset():
    def __init__(self, city_name, perturb_op=None):
        if perturb_op == 'ADD':
            self.poi_mark = 'inserted'
        elif perturb_op == 'DELETE':
            self.poi_mark = 'deleted'
        else:
            self.poi_mark = 'before'

        if city_name == 'Melb':
            path = REPO_ROOT / "data-cikm16" / "perturbation_data" / f"Melbourne_{perturb_op}.json"
        elif city_name == 'Toro':
            path = REPO_ROOT / "data-ijcai15" / "Toro" / "perturbation_data" / f"Toronto_{perturb_op}.json"
        elif city_name == 'Florence':
            path = (
                REPO_ROOT
                / "LearNext-DATASET"
                / "Florence"
                / "perturbation_data"
                / f"Florence_{perturb_op}.json"
            )

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.traj_set = []
        self.data = {}
        self.label = {}
        count = 0
        for key, value in data.items():
            self.traj_set.append(key)
            label = value['original itinerary']
            input_traj = value['response']['Perturbed Itinerary']
            intents = value['response']['Intents']
            hint = ""
            if 'Spatial distance disruption' in intents:
                hint += benchmark_prompts.hint_dis

            if 'Category diversity disruption' in intents:
                hint += benchmark_prompts.hint_div

            if 'Categorical diversity disruption' in intents:
                hint += benchmark_prompts.hint_div

            if 'Popularity disruption' in intents:
                hint += benchmark_prompts.hint_pop


            self.data[key] = {'original itinerary': label,
                              'need_to_modify itinerary': input_traj,
                              'hint': hint
                              }

            change = extract_change(original=label, perturbed=input_traj, operation=perturb_op)
            self.label[key] = {'index_label': change['index'],
                               'poi_label': change[self.poi_mark]
                               }


    def sample(self, key):
        return self.data[key], self.label[key]

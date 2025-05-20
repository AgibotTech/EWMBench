import os

from .utils import init_submodules, save_json
import EWMBench
import importlib
from itertools import chain
from pathlib import Path
import importlib.util
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json

from .distributed import get_rank, print0

from .trajectory_consistency import compute_trajectory_consistency
from .scene_consistency import compute_scene_consistency
from .diversity import compute_diversity
from .caption import caption_reference
from .semantics import compute_semantics

import csv
import re
from collections import defaultdict



class EmbodiedWorldModelBenchmark(object):
    def __init__(self, device, output_path):
        self.device = device                        # cuda or cpu
        self.output_path = output_path              # output directory to save VBench results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        return ['diversity', 'scene_consistency','trajectory_consistency','semantics']        


    def build_full_info_json(self, data_base, data_name, dimension_list, **kwargs):

        task_names = sorted(os.listdir(data_base))

        cur_full_info_list = []
        for task_id in task_names:
            task_path = os.path.join(data_base, task_id)
            for episode_id in sorted(os.listdir(task_path)):
                if episode_id.endswith(('.png', '.json')): 
                    continue
                episode_path = os.path.join(task_path, episode_id)
                for gid in sorted(os.listdir(episode_path)):
                    gid_path = os.path.join(episode_path, gid)
                    video_path = os.path.join(gid_path, "video")

                    cur_full_info_list.append({
                        "dimension": dimension_list, 
                        "video_list": [video_path]
                    })
        
        cur_full_info_path = os.path.join(self.output_path, data_name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')

        return cur_full_info_path

    def build_full_gt_info_json(self, data_base, data_name, **kwargs):
        task_names = sorted(os.listdir(data_base))

        cur_full_info_list = []
        for task_id in task_names:
            task_path = os.path.join(data_base, task_id)
            for episode_id in sorted(os.listdir(task_path)):
                if episode_id.endswith(('.png', '.json')): 
                    continue
                episode_path = os.path.join(task_path, episode_id)

                video_path = os.path.join(episode_path, "video")

                cur_full_info_list.append({
                    "video_list": [video_path]
                })
        
        cur_full_info_path = os.path.join(self.output_path, data_name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation gt data saved to {cur_full_info_path}')

        return cur_full_info_path        

    def get_evaluator(self, model_ckpt=None, model_code=None, device=None):
        from .compute_diversity import compute_diversity
        return lambda full_info_path, submodules, **kwargs: compute_diversity(full_info_path, device, submodules, **kwargs)


    def merge_all_metrics_to_csv(self, data_name, data, save_path="final_results.csv"):

        rows = []
        metrics = defaultdict(list)
        all_fields = set(["task_id", "episode_id", "trial_id"])
        scene_dict = {}
        logic_dict = {}
        diversity_data = data.get("diversity", {})
        all_triplets = set()

        if "scene_consistency" in data:
            for entry in data["scene_consistency"][1]:
                match = re.search(r'/(\d+)/(\d+)/(\d+)/video$', entry["video_path"])
                if match:
                    task_id, episode_id, trial_id = match.groups()
                    scene_dict[(task_id, episode_id, trial_id)] = entry["video_results"]
                    all_triplets.add((task_id, episode_id, trial_id))


        if "logics" in data:
            for gid, val in data["logics"].items():
                match = re.search(rf'{data_name}_dataset_(\d+)_(\d+)_(\d+)', gid)
                if match:
                    task_id, episode_id, trial_id = match.groups()
                    logic_dict[(task_id, episode_id, trial_id)] = val
                    all_triplets.add((task_id, episode_id, trial_id))

        
        if "diversity" in data:
            for task_id, episodes in data["diversity"].items():
                for episode_id in episodes:
                    all_triplets.add((str(task_id), str(episode_id), "1"))

        
        for dim in ["semantics", "trajectory_consistency"]:
            if dim not in data:
                continue
            for task_id, epis in data[dim].items():
                for episode_id, trials in epis.items():
                    for trial_id in trials.keys():
                        all_triplets.add((task_id, episode_id, trial_id))


        for task_id, episode_id, trial_id in sorted(all_triplets):
            row = {
                "task_id": int(task_id),
                "episode_id": int(episode_id),
                "trial_id": int(trial_id)
            }

            if "semantics" in data:
                sem = data["semantics"].get(task_id, {}).get(episode_id, {}).get(trial_id, {})
                if "BLEUScore" in sem:
                    row["BLEUScore"] = sem["BLEUScore"]
                    all_fields.add("BLEUScore")
                    metrics["BLEUScore"].append(sem["BLEUScore"])
                if "CLIPScore" in sem:
                    row["CLIPScore"] = sem["CLIPScore"]
                    all_fields.add("CLIPScore")
                    metrics["CLIPScore"].append(sem["CLIPScore"])

            if "trajectory_consistency" in data:
                traj = data["trajectory_consistency"].get(task_id, {}).get(episode_id, {}).get(trial_id, {})
                for k in ["hsd", "dyn", "ndtw"]:
                    if k in traj:
                        row[k] = traj[k]
                        all_fields.add(k)
                        try:
                            metrics[k].append(float(traj[k]))
                        except:
                            pass


            sc = scene_dict.get((task_id, episode_id, trial_id), "")
            if sc != "":
                row["scene_consistency"] = sc
                all_fields.add("scene_consistency")
                metrics["scene_consistency"].append(sc)

            logic = logic_dict.get((task_id, episode_id, trial_id), "")
            if logic != "":
                row["logic_constraints"] = logic
                all_fields.add("logic_constraints")
                try:
                    metrics["logic_constraints"].append(int(logic))
                except:
                    pass
            

            str_task_id = str(task_id)
            str_episode_id = str(episode_id)
            # if trial_id == "1":
            #     div_val = diversity_data.get(str_task_id, {}).get(str_episode_id, "-")
            #     row["diversity"] = div_val
            #     all_fields.add("diversity")
            #     if div_val != "-":
            #         metrics["diversity"].append(div_val)
            # else:
            #     row["diversity"] = "-"
            #     all_fields.add("diversity")

            if trial_id == "1" and "diversity" in data:
                div_val = diversity_data.get(str_task_id, {}).get(str_episode_id, "-")
                row["diversity"] = div_val
                all_fields.add("diversity")
                if div_val != "-":
                    metrics["diversity"].append(div_val)
            elif "diversity" in data:
                row["diversity"] = "-"
                all_fields.add("diversity")


            rows.append(row)
        
        field_list = []
        for f in ["task_id", "episode_id", "trial_id"] + sorted(all_fields - {"task_id", "episode_id", "trial_id"}):
            if any(f in r and r[f] != "" for r in rows):
                field_list.append(f)

        non_empty_fields = list(field_list)

        mean_row = {f: "" for f in non_empty_fields}
        mean_row["task_id"] = "MEAN"
        for f in non_empty_fields:
            if f in metrics:
                vals = metrics[f]
                if vals:
                    mean_row[f] = round(sum(vals) / len(vals), 6)

        rows.append(mean_row)

        with open(save_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=non_empty_fields)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({})
            writer.writerow({non_empty_fields[0]: "# Diversity is calculated based on the generation results of different trails within the same episode. Only shown in trial_id=1. Others are marked with '-'."})

        print(f"✅ Cleaned metrics written to {save_path}")


    def evaluate(self, data_base, data_name, dimension_list=None, local=False, gt_path=None, **kwargs):
        results_dict = {}

        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()

        submodules_dict = init_submodules(dimension_list, local=local, **kwargs)

        cur_full_info_path = self.build_full_info_json(data_base, data_name, dimension_list, **kwargs)

        json_path = os.path.join(self.output_path, f"{data_name}_results.json")

        for dimension in dimension_list:
            print0(f"Evaluating: {dimension}")
            
            if dimension == 'trajectory_consistency':
                results = compute_trajectory_consistency(
                    gt_path=gt_path, data_base=data_base
                )

            elif dimension == 'semantics':  
                submodules_list = submodules_dict[dimension] 
                caption_model = submodules_list['caption_model'] 
                semantics_model = submodules_list['clip_model'] 
                caption = caption_reference(
                                    model_name=data_name,
                                    model_path = caption_model,
                                    video_folder_root = cur_full_info_path,
                                    save_path = self.output_path,
                                    **kwargs
                                    )
                caption_json = os.path.join(self.output_path, f"{data_name}_caption_responses.json")
                with open(caption_json, 'r') as f:
                    data = json.load(f)

                result = {}
                for sample_id, info in data.items():
                    if "Overall_Constraints" in info:
                        result[sample_id] = info["Overall_Constraints"]
                    else:
                        print(f"Warning: No 'Overall_Constraints' found in {sample_id}")
                results_dict['logics'] = result

                gt_caption_json = os.path.join(self.output_path, f"gt_caption_responses.json")
                if not os.path.isfile(gt_caption_json):
                    gt_full_info_path = self.build_full_gt_info_json(gt_path, 'gt', **kwargs)
                    gt_caption = caption_reference(
                                        model_name='gt',
                                        model_path = caption_model,
                                        video_folder_root = gt_full_info_path,
                                        save_path = self.output_path,
                                        **kwargs
                                        )                                                                     
                
                
                results = compute_semantics(caption_json,gt_caption_json,semantics_model)
            elif dimension == 'scene_consistency':

                submodules_list = submodules_dict[dimension]
                results = compute_scene_consistency(cur_full_info_path, submodules_list, **kwargs)
            
            elif dimension == 'diversity':

                submodules_list = submodules_dict[dimension]

                results = compute_diversity(cur_full_info_path, submodules_list, **kwargs)
                
            else:
                raise ValueError(f"[Error] Unsupported evaluation dimension: {dimension}")

            results_dict[dimension] = results
        

        results_json = os.path.join(self.output_path,f'{data_name}_results.json')    
        with open(results_json,'w' ) as f:
            json.dump(results_dict, f, indent=2)

        csv_save_path = os.path.join(self.output_path ,"ewmbm_final_table.csv")
        self.merge_all_metrics_to_csv(data_name, results_dict, csv_save_path)










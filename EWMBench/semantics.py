# caption_metric.py
import os
import json
import multiprocessing
import numpy as np
from tqdm import tqdm
import torch
from torchmetrics.text import BERTScore, ROUGEScore, BLEUScore
from torchmetrics.multimodal.clip_score import CLIPScore

device = "cuda" if torch.cuda.is_available() else "cpu"

results_list = {}

def get_metric_eval(metric_type, clip_model_path, bleu_n_gram=4):
 
    # Sentence Metrics
    if metric_type == 'BLEUScore':
        metric = BLEUScore(n_gram=bleu_n_gram)
        # metric(preds=[generated_summary], target=[[target_summary]]))
        
    # Neural-Based
    elif metric_type == "CLIPScore":
        metric = CLIPScore(model_name_or_path=clip_model_path).to(device)
        
    
    else:
        raise ValueError("Invalid metric type")
        return None
    return metric
    
def run_metric_eval(metric, gen_strings, gt_strings, metric_type):
    try:
        if metric_type == "BLEUScore": 
            return metric(preds=[gen_strings], target=[[gt_strings]]).numpy()
        elif metric_type == "CLIPScore":
            return metric(gen_strings, gt_strings).detach().cpu().numpy()
        else:
            raise ValueError("Invalid metric type")
    except Exception as e:
        print(f"Error in computing {metric_type} score with generated string: {gen_strings} and ground-truth string: {gt_strings}")
        print(e)
        return 0.0



def get_strings(json_path, key):
    with open(json_path, 'r') as f:
        data = json.load(f) # dict
    all_idxs = []
    all_strings = []
    for idx, value in data.items():

        all_idxs.append(idx)
        if key == "General":
            all_strings.append(value["General"])
        elif key == "Events":
            event_strings = ""
            current_events = value["Events"]
            for event in current_events:
                if len(event) < 2:
                    # we discard events with less than 2 words
                    break
                event_strings += event + " "
            all_strings.append(event_strings)
        else:
            raise ValueError("Invalid key")
    return all_idxs, all_strings
                

def compute_metric_scores(semantics_model,metric_type, list_gen_strings, list_gt_strings, bleu_n_gram=4):
    metric = get_metric_eval(metric_type, semantics_model, bleu_n_gram)

    scores = [run_metric_eval(metric, gen_strings, gt_strings, metric_type) for gen_strings, gt_strings in tqdm(list(zip(list_gen_strings, list_gt_strings)), desc="Compute metric scores")]
    return scores

def evaluate_run(semantics_model,eval_config, dt_json, gt_json):

    metric_type = eval_config["metric_type"]
    key = eval_config["key"]
    bleu_n_gram = eval_config.get("bleu_n_gram", 4)
    
    dt_dasset_name = os.path.basename(dt_json).split("_dataset")[0]
    # Load generated and ground-truth strings
    gen_idxs, gen_strings = get_strings(dt_json, key)
    gt_idxs, gt_strings = get_strings(gt_json, key)
    expanded_gt_strings =[]
    expanded_gt_idxs = [] 
    
    for gen_id, gen_string in zip(gen_idxs, gen_strings):
        # task_id, eps_id, trail_num = gen_id.split("_dataset_")[-1].rsplit("_")
        task_eps_trail = gen_id.split("_dataset_")[-1].split("_")
        task_id = task_eps_trail[0]
        trail_num = task_eps_trail[-1]
        if len(task_eps_trail)==3:
            eps_id = task_eps_trail[1]
        else:
            eps_id = "_".join(task_eps_trail[1:-1])

        gt_idx = f"gt_dataset_{task_id}_{eps_id}"
        gt_string = gt_strings[gt_idxs.index(gt_idx)]
        expanded_gt_strings.append(gt_string)
        expanded_gt_idxs.append(gt_idx)
      
    gt_strings = expanded_gt_strings
    
    assert len(gen_strings) == len(gt_strings), "Number of generated and ground-truth strings do not match"
    print(f"the number of generated strings: {len(gen_strings)}")
    # Compute metric scores
    scores = compute_metric_scores(semantics_model, metric_type, gen_strings, gt_strings, bleu_n_gram)
    
    # Save scores to file
    scores_list = [np.around(score, decimals=6).tolist() for score in scores]

    for idx, gen_idx in enumerate(gen_idxs):
        parts = gen_idx.split('_')
        task_id = parts[-3]
        episode_id = parts[-2]
        gid = parts[-1]

        if task_id not in results_list:
            results_list[task_id] = {}
        if episode_id not in results_list[task_id]:
            results_list[task_id][episode_id] = {}
        if gid not in results_list[task_id][episode_id]:
            results_list[task_id][episode_id][gid] = {}

        results_list[task_id][episode_id][gid][metric_type] = scores_list[idx]

    avg_score = np.mean(scores).tolist()

    return scores


def evaluate_runs_single_config(eval_config, json_path, gt_json, semantics_model):


    assert gt_json is not None, "Ground-truth JSON not found"

    scores_list = {}

    if json_path != gt_json:
        dt_dataset_name = os.path.basename(json_path).split("_dataset")[0]
        scores = evaluate_run(semantics_model,eval_config, json_path, gt_json)
        avg_score = np.mean(scores)
        print(f"{dt_dataset_name}: {avg_score:.6f}")
        scores_list[dt_dataset_name] = float(avg_score)

    return scores_list



def evaluate_runs_configs(eval_configs, json_path, gt_path, semantics_model):
    eval_results = {}
    for eval_config in tqdm(eval_configs, desc="Eval different configs"):
        exp_name = eval_config["metric_type"]
        if "bleu_n_gram" in eval_config:
            exp_name += f"_ngram{eval_config['bleu_n_gram']}"
        print(f"Evaluating {exp_name}...")
        scores_list = evaluate_runs_single_config(eval_config, json_path, gt_path, semantics_model)
        print(f"Scores: {scores_list}")
        eval_results[exp_name] = scores_list
    return eval_results
        
def compute_semantics(json_path, gt_path,semantics_model):

    eval_configs = [
        {
            "metric_type": "BLEUScore",
            "key": "General",
            "bleu_n_gram": 4,
        },
        {
            "metric_type": "CLIPScore",
            "key": "General",
        }
    ]
    

        
    results = evaluate_runs_configs(eval_configs, json_path, gt_path, semantics_model)
    return results_list

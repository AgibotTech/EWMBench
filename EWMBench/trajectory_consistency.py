#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os
import argparse
import json
import math
from scipy.stats import norm
from tqdm import tqdm

def one_traj_interpo_fill(data):
    invaild_traj = False
    
    mask = (data != [-1., -1.]).any(axis=1)

    invalid_ratio = 1 - np.mean(mask)

    if invalid_ratio>0.80:
        invaild_traj = True
        # raise ValueError("All points are missing. Cannot perform interpolation.")
        print("Warning: This metric has all invalid values ​​and a score of zero！！！！Interpolation will not work.")
        return data, invaild_traj
    
    if not invaild_traj:
        n = data.shape[0]
        prev = np.full(n, -1, dtype=int)  
        next_ = np.full(n, n, dtype=int)   
        
        last = -1
        for i in range(n):
            if mask[i]:
                last = i
            prev[i] = last
        
        last = n
        for i in range(n-1, -1, -1):
            if mask[i]:
                last = i
            next_[i] = last
        
        missing = np.where(~mask)[0]
        if len(missing) == 0:
            return data, invaild_traj 
        
        p_vals = prev[missing]
        q_vals = next_[missing]
        

        mask_p_invalid = (p_vals == -1)
        mask_q_invalid = (q_vals == n)
        mask_both_valid = ~mask_p_invalid & ~mask_q_invalid

        if np.any(mask_p_invalid):
            data[missing[mask_p_invalid]] = data[q_vals[mask_p_invalid]]
        

        if np.any(mask_q_invalid):
            data[missing[mask_q_invalid]] = data[p_vals[mask_q_invalid]]
        

        if np.any(mask_both_valid):
            valid_missing = missing[mask_both_valid]
            p = p_vals[mask_both_valid]
            q = q_vals[mask_both_valid]
            

            alpha = (valid_missing - p) / (q - p).astype(float)
            alpha = alpha[:, np.newaxis]  

            interpolated = (1 - alpha) * data[p] + alpha * data[q]
            data[valid_missing] = interpolated
    
    return data, invaild_traj


def traj_interpo_fill(traj):
    n_traj = traj.shape[1]
    filled_trajs = []
    invaild_trajs = []
    for i in range(n_traj):
        filled_traj, invaild_traj = one_traj_interpo_fill(traj[:,i].copy())
        filled_trajs.append(filled_traj)
        invaild_trajs.append(invaild_traj)
    
    filled_trajs = np.stack(filled_trajs,axis=1)
    return filled_trajs, invaild_trajs


def hausdorff_distance(traj1, traj2):


    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    d1 = directed_hausdorff(traj1, traj2)[0]
    d2 = directed_hausdorff(traj2, traj1)[0]
    return max(d1, d2)

def HAUSDORFF(traj_pred, traj_gt, invaild_pred_trajs, invalid_gt_trajs):
    """
    traj_pred: np arr N,k,2

    traj_gt: np arr M,k,2
    """
    n_traj = traj_gt.shape[1]

    gt_max_distance_list = []
    for i in range(n_traj):
        # assert not invalid_gt_trajs[i], "Invalid gt trajectory encountered at index {}".format(i)
        if invalid_gt_trajs[i]:
            print("Invalid gt trajectory encountered at index {}".format(i))
            continue
        gt_max_distance = farthest_distance(traj_gt[:, i])
        gt_max_distance_list.append(gt_max_distance)
    assert len(gt_max_distance_list)>=1, "All GT trajectory invalid!"

    max_distance_index, max_distance_traj = max(enumerate(gt_max_distance_list), key=lambda x: x[1])
    
    if invaild_pred_trajs[max_distance_index]:
        hs = 0
    else:
        hi = hausdorff_distance(traj_pred[:,max_distance_index],traj_gt[:,max_distance_index])

        hs = 1/hi
    
    return hs, max_distance_index




def DYNAMICS(traj_pred,traj_gt,invaild_pred_trajs,invalid_gt_trajs,max_distance_index):
    """
    traj_pred: np arr N,k,2

    traj_gt: np arr M,k,2
    """
    
    len_pred = traj_pred.shape[0]
    diff_pred = np.diff(traj_pred, axis=0)
    vel_pred = np.sqrt(diff_pred[:,:,0]**2 + diff_pred[:,:,1]**2) # N-1 2
    acc_pred = np.diff(vel_pred, axis=0) # N-2 2
    vel_weight_pred = [1/(len_pred-1)]*(len_pred-1)
    acc_weight_pred = [1/(len_pred-2)]*(len_pred-2)

    len_gt = traj_gt.shape[0]
    diff_gt = np.diff(traj_gt, axis=0)
    vel_gt = np.sqrt(diff_gt[:,:,0]**2 + diff_gt[:,:,1]**2) # N-1 2
    acc_gt = np.diff(vel_gt, axis=0) # N-2 2
    vel_weight_gt = [1/(len_gt-1)]*(len_gt-1)
    acc_weight_gt = [1/(len_gt-2)]*(len_gt-2)

    eps = 1e-8
    max_gt_vel = max(vel_gt[:,max_distance_index])-min(vel_gt[:,max_distance_index])
    max_pred_vel = max(vel_pred[:,max_distance_index])-min(vel_pred[:,max_distance_index])  
    vr = (min(max_gt_vel, max_pred_vel)+eps)/(max(max_gt_vel, max_pred_vel)+eps)
    max_gt_acc = max(acc_gt[:,max_distance_index])-min(acc_gt[:,max_distance_index])
    max_pred_acc = max(acc_pred[:,max_distance_index])-min(acc_pred[:,max_distance_index])   
    ar = (min(max_gt_acc, max_pred_acc)+eps)/(max(max_gt_acc, max_pred_acc)+eps)

    if invaild_pred_trajs[max_distance_index]:
        vel_score = 0.0
        acc_score = 0.0
    else:

        vel_score = wasserstein_distance(vel_pred[:,max_distance_index],vel_gt[:,max_distance_index], 
                                        u_weights=vel_weight_pred, v_weights=vel_weight_gt)
        
        
        acc_score = wasserstein_distance(acc_pred[:,max_distance_index],acc_gt[:,max_distance_index], 
                                        u_weights=acc_weight_pred, v_weights=acc_weight_gt)

        if vel_score:
            vel_score = vr*(1/vel_score)
        else:
            vel_score = 0.0
        if acc_score:
            acc_score = ar*(1/acc_score)
        else:
            acc_score = 0.0
    
    score = 0.007*vel_score + 0.003*acc_score

    return score

from scipy.spatial import ConvexHull

def farthest_distance(points):

    if len(points) < 2:
        return 0.0
    
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    

    def rotating_calipers(vertices):
        n = len(vertices)
        max_dist = 0.0
        k = 1 
        pts = np.vstack([vertices, vertices[0]])
        
        for i in range(n):
            j = (i + 1) % n

            while True:
                next_k = (k + 1) % n

                cross = (pts[j,0]-pts[i,0])*(pts[next_k,1]-pts[k,1]) - \
                        (pts[j,1]-pts[i,1])*(pts[next_k,0]-pts[k,0])
                if cross < 0:
                    k = next_k
                else:
                    break

            max_dist = max(max_dist, 
                          np.linalg.norm(pts[i]-pts[k]),
                          np.linalg.norm(pts[i]-pts[next_k]))
        return max_dist
    
    return rotating_calipers(hull_points)

def dtw_distance(traj1, traj2):

    distance, dpath = fastdtw(np.array(traj1), np.array(traj2), dist=euclidean)
    return distance,dpath

def NDTW(traj_pred,traj_gt,invaild_pred_trajs,invalid_gt_trajs,max_distance_index):
    """
    traj_pred: np arr N,k,2

    traj_gt: np arr M,k,2
    """

    if invaild_pred_trajs[max_distance_index]:
        ds = 0.0
    else:
        di,pi = dtw_distance(traj_pred[:,max_distance_index],traj_gt[:,max_distance_index])
        di = di/len(pi)

        ds = 1/di
    return ds


def trim_trajectory(traj):

    non_zero = np.where(np.any(np.array(traj) != [0, 0], axis=1))[0]
    return traj[:non_zero[-1]+1] if len(non_zero) > 0 else []


def eval_traj(traj_pred_file, traj_gt_file):

    traj_pred = np.load(traj_pred_file).astype('float32')
    traj_gt = np.load(traj_gt_file).astype('float32')
    traj_pred, invaild_pred_trajs = traj_interpo_fill(traj_pred)
    traj_gt, invalid_gt_trajs = traj_interpo_fill(traj_gt)
    
    # print(traj_gt_file, invaild_pred_trajs, invalid_gt_trajs)

    hsd,max_distance_index = HAUSDORFF(traj_pred,traj_gt,invaild_pred_trajs,invalid_gt_trajs)
    dyn = DYNAMICS(traj_pred,traj_gt,invaild_pred_trajs,invalid_gt_trajs,max_distance_index)
    ndtw = NDTW(traj_pred,traj_gt,invaild_pred_trajs,invalid_gt_trajs,max_distance_index)

    hsd_max = 24.979
    dyn_max = 22.519
    ndtw_max = 50.202
    min = 0.0

    if hsd>hsd_max:
        hsd = 1.0
    else:
        hsd = (hsd-min)/(hsd_max-min)

    if dyn>dyn_max:
        dyn = 1.0
    else:
        dyn = (dyn-min)/(dyn_max-min)
        
    if ndtw>ndtw_max:
        ndtw = 1.0
    else:
        ndtw = (ndtw-min)/(ndtw_max-min)    


    res = {
        # 'lcss': '%.3f' % lcss,
        'hsd': '%.3f' % hsd,
        'dyn': '%.3f' % dyn,
        'ndtw': '%.3f' % ndtw
    }

    return res

def calculate_means_and_variances(data):
    metrics = ['hsd', 'dyn', 'ndtw']
    values = {metric: [] for metric in metrics}


    for task_name, samples in data.items():
        for sample_id, groups in samples.items():
            for gid, traj_eval_res in groups.items():
                for metric in metrics:
                    if metric in traj_eval_res:

                        values[metric].append(float(traj_eval_res[metric]))

    means = {}
    variances = {}
    for metric in metrics:
        means[metric] = np.mean(values[metric]) if values[metric] else None
        variances[metric] = np.var(values[metric]) if values[metric] else None
    
    return means, variances


def compute_trajectory_consistency(gt_path, data_base):

    metric_keys = ['hsd', 'dyn', 'ndtw']

    res = {}

    for task_id in sorted(os.listdir(data_base)):
        task_path = os.path.join(data_base,task_id)

        res[task_id] = {}
        for episode_id in tqdm(sorted(os.listdir(task_path))):
            if episode_id.endswith(('.png', '.json')): 
                continue
            res[task_id][episode_id] = {}
            gt_traj_file = os.path.join(gt_path,task_id,episode_id,'traj','traj.npy')

            episode_path = os.path.join(task_path,episode_id)
            
            for gid in sorted(os.listdir(episode_path)):           
                pred_traj_file = os.path.join(episode_path,gid,'traj','traj.npy')
                traj_eval_res = eval_traj(pred_traj_file, gt_traj_file)
                
                res[task_id][episode_id][gid] = traj_eval_res

    return res
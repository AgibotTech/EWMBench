from collections import defaultdict
import numpy as np
import os
import av
import cv2
from ultralytics import YOLO
import decord
import yaml


def process_video_with_tracking(input_path, output_path, gid=None,
                                 model_path='', data_type='val', saving_fps=5):
    
    # print(f"Processing input: {input_path}")

    model = YOLO(model_path).to('cuda:0')

    if data_type=='val':
        output_video_path = os.path.join(output_path, gid, "gripper_detection")
        output_traj_path = os.path.join(output_path, gid, "traj")
    else:
        output_video_path = os.path.join(output_path, "gripper_detection")
        output_traj_path = os.path.join(output_path, "traj")

    os.makedirs(output_video_path, exist_ok=True)
    os.makedirs(output_traj_path, exist_ok=True)

    output_video_file = os.path.join(output_video_path, "video.mp4")
    output_container = av.open(output_video_file, mode='w', format='mp4')
    output_stream = output_container.add_stream('h264', rate=saving_fps)
    output_stream.width = 640
    output_stream.height = 480
    output_stream.pix_fmt = 'yuv420p'

    trajectory_data = []
    track_history = defaultdict(lambda: [])


    image_files = sorted([
        f for f in os.listdir(input_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    frames = []
    for fname in image_files:
        img_path = os.path.join(input_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    for global_frame_idx, img in enumerate(frames):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        results = model.track(img, persist=True, conf=0.8)
        boxes = results[0].boxes

        clses = boxes.cls.cpu().tolist() if boxes.cls is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []

        selected_indices = {}
        for idx, (cls, conf) in enumerate(zip(clses, confs)):
            if cls not in selected_indices or conf > selected_indices[cls][1]:
                selected_indices[cls] = (idx, conf)
        selected_idxs = [v[0] for v in selected_indices.values()]

        filtered_boxes = boxes[selected_idxs].xywh.cpu()
        filtered_clses = [clses[i] for i in selected_idxs]
        filtered_confs = [confs[i] for i in selected_idxs]

        try:
            track_ids = boxes.id.int().cpu().tolist()
        except AttributeError:
            track_ids = []

        annotated_frame = img.copy()
        left_track = False
        right_track = False
        left_track_data = []
        right_track_data = []

        for box, track_id, cls, conf in zip(filtered_boxes, track_ids, filtered_clses, filtered_confs):
            x_center, y_center, w, h = box.tolist()
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

            color = (0, 255, 0) if cls == 1.0 else (0, 0, 255)
            label = f"ID:{track_id} {model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            track = track_history[track_id]
            track.append((float(x_center), float(y_center)))
            if cls == 0.0:
                left_track = True
                left_track_data.append((float(x_center) / 640, float(y_center) / 480))
            elif cls == 1.0:
                right_track = True
                right_track_data.append((float(x_center) / 640, float(y_center) / 480))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], False, (230, 230, 230), 10)

        if not left_track:
            left_track_data.append((-1, -1))
        if not right_track:
            right_track_data.append((-1, -1))

        trajectory_data.append(left_track_data + right_track_data)

        output_frame = av.VideoFrame.from_ndarray(annotated_frame, format='rgb24')
        for packet in output_stream.encode(output_frame):
            output_container.mux(packet)
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()

    np.save(os.path.join(output_traj_path, 'traj.npy'), np.array(trajectory_data))



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--detect_gt', action='store_false')
    args = parser.parse_args()
    config = load_config(args.config_path)

    data_base = config['data']['val_base']
    gt_path = config['data']['gt_path']
    model_path = config.get('ckpt',{}).get('yolo_world_ckpt', '')


    for task in sorted(os.listdir(data_base)):
        task_path = os.path.join(data_base, task)
        for episode in os.listdir(task_path):
            if episode.endswith(('.png', '.json')):
                continue
            episode_path = os.path.join(task_path, episode)
            
            gt_episode_path = os.path.join(gt_path, task, episode)
            gt_video = os.path.join(gt_episode_path, 'video')
            # user_input = input("Do you want to detect the ground truth trajectory? [y/N]: ").strip().lower()
            # gt_detect_opt = user_input in ['y', 'yes', 'true', '1']
            if args.detect_gt:
                process_video_with_tracking(
                    input_path=gt_video,
                    output_path=gt_episode_path,
                    gid=None,
                    model_path=model_path,
                    data_type='gt'
                )            

            for gid in sorted(os.listdir(episode_path)):
                input_path = os.path.join(episode_path, gid, "video")
                process_video_with_tracking(
                    input_path=input_path,
                    output_path=episode_path,
                    gid=gid,
                    model_path=model_path,
                    data_type='val'
                )
                continue




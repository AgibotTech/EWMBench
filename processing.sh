#!/usr/bin/env bash

config=$1

#resize(640,480)
python ./processing/video_resize.py --config_path "$config"

#detect & traj
python ./processing/detection_tracking.py --config_path "$config"

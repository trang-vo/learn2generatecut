#!/bin/bash

# write bash scripts to run function run_rainbow_maxcut_cycle in src/run_methods_maxcut.py
cd ../src

problem_type="maxcut"
cut_type="cycle"
instance_path="../data/evaluate/maxcut/small/pm100_10_9003.maxcut"
model_path="../models/cut_evaluator/cycle_time"
model_name="best_model.pth"
user_callback_type="RainbowUserCallback"
frequent=1
terminal_gap=0.01
result_path=""
display_log=True
log_path=""
time_limit=3600
dfs_search=True
use_cut_detector=False
cut_detector_path=""
device="cuda"

python rainbow_solve.py $problem_type $cut_type $instance_path $model_path --model-name=$model_name --user-callback-type=$user_callback_type --frequent=$frequent --terminal-gap=$terminal_gap --result-path=$result_path --display-log --log-path=$log_path --time-limit=$time_limit --dfs-search --no-use-cut-detector --cut-detector-path=$cut_detector_path --device=$device

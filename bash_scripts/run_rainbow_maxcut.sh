#!/bin/bash

# write bash scripts to run function run_rainbow_maxcut_cycle in src/run_methods_maxcut.py
cd ../src
data_root="../data/maxcut_instances/120"
nb_cpus=1
model_folder="../models/cut_evaluator/subtour_time"
model_name="best_model.pth"
user_callback_type=HeuristicRainbowUserCallback
frequent=8
terminal_gap=0.01
time_limt=3600
python run_methods_maxcut.py run-rainbow $data_root $nb_cpus $model_folder --model-name=$model_name --user-callback-type=$user_callback_type --frequent=$frequent --terminal-gap=$terminal_gap --time-limit=$time_limt
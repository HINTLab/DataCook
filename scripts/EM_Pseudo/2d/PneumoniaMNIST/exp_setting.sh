#!/bin/bash

# Exp Setting
export config_path=configs/MedMNIST/PneumoniaMNIST
export dataset_type=PneumoniaMNIST
export poison_dataset_type=PoisonPneumoniaMNIST
export attack_type=EM-Pseudo
export perturb_type=samplewise
export base_version=resnet18
export epsilon=8
export step_size=1.6
export batch_size=128
export num_steps=20
export universal_stop_error=0.01
export exp_args=${dataset_type}-eps=${epsilon}-step_size=${step_size}-batch_size=${batch_size}-base_version=${base_version}-universal_stop_error=${universal_stop_error}
export exp_path=experiments/${dataset_type}/${attack_type}_${perturb_type}/${exp_args}

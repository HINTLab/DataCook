#!/bin/bash

# Exp Setting
export config_path=configs/MedMNIST/FractureMNIST3D
export dataset_type=FractureMNIST3D
export poison_dataset_type=PoisonFractureMNIST3D
export attack_type=datacook
export perturb_type=samplewise
export base_version=resnet18_3d
export epsilon=32
export step_size=2
export batch_size=32
export num_steps=20
export universal_stop_error=0.01
export exp_args=${dataset_type}-eps=${epsilon}-step_size=${step_size}-batch_size=${batch_size}-base_version=${base_version}-universal_stop_error=${universal_stop_error}
export exp_path=experiments/${dataset_type}/${attack_type}_${perturb_type}/${exp_args}


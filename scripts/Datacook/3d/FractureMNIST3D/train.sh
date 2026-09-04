#!/bin/bash

# Load EXP Setting
source exp_setting.sh


# Training Setting
model_name=resnet18_3d
poison_rate=1
exp_name=${exp_path}/poison_train_${poison_rate}
echo $exp_name

# Poison Training
cd ../../../../
rm -rf ${exp_name}/${model_name}
python3 -u main.py    --version                 $model_name                 \
                      --exp_name                $exp_name                   \
                      --config_path             $config_path                \
                      --train_data_type         $poison_dataset_type        \
                      --train_data_path         ${exp_path}/resnet18_3d/ae/perturb_data.npz \
                      --recover_test_data_path  ${exp_path}/resnet18_3d/ae_test/perturb_data.npz \
                      --test_data_type          $dataset_type               \
                      --recover_test_data_type  $poison_dataset_type         \
                      --poison_rate             $poison_rate                \
                      --perturb_type            $perturb_type               \
                      --perturb_tensor_filepath ${exp_path}/perturbation.pt \
                      --perturb_test_tensor_filepath  ${exp_path}/test_perturbation.pt \
                      --train

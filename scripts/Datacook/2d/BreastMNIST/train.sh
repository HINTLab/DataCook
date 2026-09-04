#!/bin/bash

# Load EXP Setting
source exp_setting.sh


# Training Setting
model_name=resnet50

exp_name=${exp_path}/poison_train
echo $exp_name

# Poison Training
cd ../../../../

python3 -u main.py    --version                 $model_name                 \
                      --exp_name                $exp_name                   \
                      --config_path             $config_path                \
                      --train_data_type         $poison_dataset_type        \
                      --train_data_path         ${exp_path}/resnet18/ae/ \
                      --recover_test_data_path  ${exp_path}/resnet18/ae_test/ \
                      --test_data_type          $dataset_type               \
                      --recover_test_data_type  $poison_dataset_type         \
                      --train

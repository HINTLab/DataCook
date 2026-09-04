#!/bin/bash

# Load Exp Settings
source exp_setting.sh


# Remove previous files
echo $exp_path


# Search Universal Perturbation and build datasets
cd ../../../../
pwd
rm -rf $exp_name
python3 perturbation.py --config_path             $config_path       \
                        --exp_name                $exp_path          \
                        --version                 $base_version      \
                        --train_data_type         $dataset_type      \
                        --test_data_type          $dataset_type      \
                        --recover_test_data_type  $dataset_type      \
                        --noise_shape             97477 3 28 28      \
                        --test_noise_shape        1000 3 28 28       \
                        --epsilon                 $epsilon           \
                        --num_steps               $num_steps         \
                        --step_size               $step_size         \
                        --attack_type             $attack_type       \
                        --universal_stop_error    $universal_stop_error\
                        --train_batch_size        $batch_size          \
                        --eval_batch_size         $batch_size          \

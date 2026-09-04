#!/bin/bash

# Load Exp Settings
source exp_setting.sh


# Remove previous files
echo $exp_path


# Search Universal Perturbation and build datasets
cd ../../../../
pwd
rm -rf $exp_name
python3 perturbation_3d.py --config_path             $config_path       \
                        --exp_name                $exp_path          \
                        --version                 $base_version      \
                        --train_data_type         $dataset_type      \
                        --test_data_type          $dataset_type      \
                        --recover_test_data_type  $dataset_type      \
                        --noise_shape             1027 1 28 28 28      \
                        --test_noise_shape        240 1 28 28 28       \
                        --train_step              33               \
                        --epsilon                 $epsilon           \
                        --num_steps               $num_steps         \
                        --step_size               $step_size         \
                        --attack_type             $attack_type       \
                        --perturb_type            $perturb_type      \
                        --universal_train_target  $universal_train_target\
                        --universal_stop_error    $universal_stop_error\
                        --num_clusters            $num_clusters       \
                        --train_batch_size        $batch_size          \
                        --eval_batch_size         $batch_size          \


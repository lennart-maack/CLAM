#!/bin/bash

MODEL_NAME=vit_large_patch16_224_MAE_histopatho_pretr
CUDA_VISIBLE_DEVICES=0 python main.py \
    --drop_out \
    --lr 2e-4 \
    --exp_code CMB_vs_DN \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task CMB_vs_DN \
    --model_type clam_sb \
    --log_data \
    --data_root_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
    --results_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME}/training_01 \
    --split_dir CMB_vs_DN_train_val_100 \
    --max_epochs 180

MODEL_NAME=vit_large_patch16_384_norm_pretr
CUDA_VISIBLE_DEVICES=0 python main.py \
    --drop_out \
    --lr 2e-4 \
    --exp_code CMB_vs_DN \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task CMB_vs_DN \
    --model_type clam_sb \
    --log_data \
    --data_root_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
    --results_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME}/training_01 \
    --split_dir CMB_vs_DN_train_val_100 \
    --max_epochs 180


MODEL_NAME=vit_large_patch16_224_norm_pretr
CUDA_VISIBLE_DEVICES=0 python main.py \
    --drop_out \
    --lr 2e-4 \
    --exp_code CMB_vs_DN \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task CMB_vs_DN \
    --model_type clam_sb \
    --log_data \
    --data_root_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
    --results_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME}/training_01 \
    --split_dir CMB_vs_DN_train_val_100 \
    --max_epochs 180

MODEL_NAME=resnet_50
CUDA_VISIBLE_DEVICES=0 python main.py \
    --drop_out \
    --lr 2e-4 \
    --exp_code CMB_vs_DN \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task CMB_vs_DN \
    --model_type clam_sb \
    --log_data \
    --data_root_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
    --results_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME}/training_01 \
    --split_dir CMB_vs_DN_train_val_100 \
    --max_epochs 180
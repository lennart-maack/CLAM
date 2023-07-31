#!/bin/bash

# MODEL_NAME=vit_large_patch16_224_norm_pretr
# CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
#     --data_h5_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
#     --data_slide_dir /data/Maack/PANT/ndpi_reduced/NDPI \
#     --csv_path /data/Maack/PANT/CLAM/experiments/${MODEL_NAME}/process_list_autogen.csv \
#     --feat_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
#     --batch_size 2048 \
#     --slide_ext .ndpi \
#     --model ${MODEL_NAME}


# MODEL_NAME=vit_large_patch16_384_norm_pretr
# CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
#     --data_h5_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
#     --data_slide_dir /data/Maack/PANT/ndpi_reduced/NDPI \
#     --csv_path /data/Maack/PANT/CLAM/experiments/${MODEL_NAME}/process_list_autogen.csv \
#     --feat_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
#     --batch_size 2048 \
#     --slide_ext .ndpi \
#     --model ${MODEL_NAME}


MODEL_NAME=resnet_50_histo_pretr
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
    --data_h5_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
    --data_slide_dir /data/Maack/PANT/ndpi_reduced/NDPI \
    --csv_path /data/Maack/PANT/CLAM/experiments/${MODEL_NAME}/process_list_autogen.csv \
    --feat_dir /data/Maack/PANT/CLAM/experiments/${MODEL_NAME} \
    --batch_size 1024 \
    --slide_ext .ndpi \
    --model ${MODEL_NAME} \
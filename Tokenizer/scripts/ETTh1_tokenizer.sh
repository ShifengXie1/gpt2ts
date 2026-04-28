#!/bin/bash

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

DATA_NAME="ETTh1"
SEQ_LEN=96
WAVE_LENGTH=8
N_EMBED=128
ENC_IN=7
EPOCHS=30

VQ_MODEL="GPT2ClusterVQ"
GPT_CLUSTER_PATH="./checkpoints/gpt2_${DATA_NAME}_emb${N_EMBED}_clusters.pt"
SAVE_PATH="./checkpoints/${DATA_NAME}_${SEQ_LEN}_dm64_dr0.2_emb${N_EMBED}_wl${WAVE_LENGTH}_bl2_${VQ_MODEL}_fixed_gpt2_codebook"

mkdir -p "$SAVE_PATH"

python -u main.py \
    --vq_model "$VQ_MODEL" \
    --seq_len "$SEQ_LEN" \
    --token_len "$SEQ_LEN" \
    --wave_length "$WAVE_LENGTH" \
    --n_embed "$N_EMBED" \
    --enc_in "$ENC_IN" \
    --gpt2_cluster_path "$GPT_CLUSTER_PATH" \
    --num_epoch "$EPOCHS" \
    --eval_per_epoch \
    --save_path "$SAVE_PATH"

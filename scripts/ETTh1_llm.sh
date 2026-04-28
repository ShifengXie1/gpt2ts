#!/bin/bash

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

DATA_NAME="ETTh1"
ROOT_PATH="./data/ETT"
DATA_PATH="ETTh1.csv"
TOKENIZER_PATH="./Tokenizer/checkpoints/ETTh1_96_dm64_dr0.2_emb128_wl8_bl2_GPT2ClusterVQ_fixed_gpt2_codebook"

PRED_LENS=(96)
LEARNING_RATE=0.00005
TRAIN_EPOCHS=10
PATIENCE=3
SEED=42

for pred_len in "${PRED_LENS[@]}"; do
    python -u run.py \
        --is_training 1 \
        --data "$DATA_NAME" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --pred_len "$pred_len" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --learning_rate "$LEARNING_RATE" \
        --train_epochs "$TRAIN_EPOCHS" \
        --patience "$PATIENCE" \
        --use_multivariate true \
        --seed "$SEED"
done

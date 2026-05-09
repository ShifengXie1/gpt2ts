#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

DATA_NAME="ETTh1"
RESULTS_DIR="./results"
SEED=42

PRED_LENS=(96)
LEARNING_RATE=0.00005
TRAIN_EPOCHS=10
PATIENCE=3
SEQ_LEN=720
BATCH_SIZE=8
TOKEN_BATCH_SIZE=8
PATCH_SIZE=8
STRIDE=8
N_LAYERS=0
CLUSTER_NUM=128
TOKEN_TRAIN_STRIDE=1
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05


for pred_len in "${PRED_LENS[@]}"; do
    python -u run.py \
        --data "$DATA_NAME" \
        --features S \
        --seq_len "$SEQ_LEN" \
        --pred_len "$pred_len" \
        --batch_size "$BATCH_SIZE" \
        --token_batch_size "$TOKEN_BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --train_epochs "$TRAIN_EPOCHS" \
        --patience "$PATIENCE" \
        --results_dir "$RESULTS_DIR" \
        --patch_len "$PATCH_SIZE" \
        --stride "$STRIDE" \
        --n_layers "$N_LAYERS" \
        --cluster_num "$CLUSTER_NUM" \
        --token_train_stride "$TOKEN_TRAIN_STRIDE" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --gpt_local_path ./gpt \
        --loss mse \
        --lradj type3 \
        --seed "$SEED"
done

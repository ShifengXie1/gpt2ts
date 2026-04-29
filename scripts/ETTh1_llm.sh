#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATA_NAME="ETTh1"
RESULTS_DIR="./results"
SEED=42

PRED_LENS=(96)
LEARNING_RATE=0.00005
TRAIN_EPOCHS=10
PATIENCE=3
PATCH_SIZE=16


for pred_len in "${PRED_LENS[@]}"; do
    python -u run.py \
        --is_training 1 \
        --data "$DATA_NAME" \
        --pred_len "$pred_len" \
        --learning_rate "$LEARNING_RATE" \
        --train_epochs "$TRAIN_EPOCHS" \
        --patience "$PATIENCE" \
        --results_dir "$RESULTS_DIR" \
        --patch_size "$PATCH_SIZE" \
        --seed "$SEED"
done

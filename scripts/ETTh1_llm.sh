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
PATCH_SIZE=16
STRIDE=8
N_LAYERS=6
NUM_CLUSTERS=64
FORECAST_TOP_K=64
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05


for pred_len in "${PRED_LENS[@]}"; do
    python -u run.py \
        --data "$DATA_NAME" \
        --features M \
        --seq_len "$SEQ_LEN" \
        --pred_len "$pred_len" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --train_epochs "$TRAIN_EPOCHS" \
        --patience "$PATIENCE" \
        --results_dir "$RESULTS_DIR" \
        --patch_len "$PATCH_SIZE" \
        --stride "$STRIDE" \
        --n_layers "$N_LAYERS" \
        --num_clusters "$NUM_CLUSTERS" \
        --forecast_top_k "$FORECAST_TOP_K" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --gpt_local_path ./gpt \
        --loss mse \
        --lradj type3 \
        --seed "$SEED"
done

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATA_NAME="ETTh1"
RESULTS_DIR="./results"
SEED=42

PRED_LENS=(96)
LEARNING_RATE=0.00005
TRAIN_EPOCHS=10
PATIENCE=3
SEQ_LEN=720
LABEL_LEN=48
BATCH_SIZE=8
PATCH_SIZE=16
STRIDE=8
N_LAYERS=6
NUM_CLUSTERS=64
CLUSTER_SAMPLE_SIZE=8192
VOCAB_CLUSTER_SAMPLE_SIZE=20000
KMEANS_ITERS=8
FORECAST_TOP_K=64
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
NUM_WORKERS=0


for pred_len in "${PRED_LENS[@]}"; do
    python -u run.py \
        --is_training 1 \
        --data "$DATA_NAME" \
        --features M \
        --seq_len "$SEQ_LEN" \
        --label_len "$LABEL_LEN" \
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
        --cluster_sample_size "$CLUSTER_SAMPLE_SIZE" \
        --vocab_cluster_sample_size "$VOCAB_CLUSTER_SAMPLE_SIZE" \
        --kmeans_iters "$KMEANS_ITERS" \
        --forecast_top_k "$FORECAST_TOP_K" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --gpt_local_path ./gpt \
        --gpt_local_files_only true \
        --use_pretrained_gpt2 true \
        --loss mse \
        --lradj none \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED"
done

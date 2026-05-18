#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

DATA_NAME="ETTh1"
RESULTS_DIR="./results"
SEED=42

PRED_LENS=(96)
LEARNING_RATE=0.001
TRAIN_EPOCHS=10
PATIENCE=3
SEQ_LEN=720
BATCH_SIZE=8
TOKEN_BATCH_SIZE=8
PATCH_SIZE=8
STRIDE=4
CLUSTER_NUM=512
TOKEN_TRAIN_STRIDE=1
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
CANDIDATE_TOKEN_NUM=4096
PATCH_BANK_TOPK=8
ASSIGNMENT_METHOD="hungarian"
USE_TRAINABLE_PATCH_PROJECTOR=False
PATCH_ENCODER_DIM=256
PATCH_BANK_ATTN_DIM=128
USE_PATCH_BANK_ATTENTION=True
LAMBDA_CE=0.3
LAMBDA_MSE=1.0
LAMBDA_ALIGN=0.0
LAMBDA_SMOOTH=0.05
MSE_TEMPERATURE=1.0
ALIGN_TEMPERATURE=1.0
RESIDUAL_SCALE=0.1


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
        --cluster_num "$CLUSTER_NUM" \
        --candidate_token_num "$CANDIDATE_TOKEN_NUM" \
        --patch_bank_topk "$PATCH_BANK_TOPK" \
        --assignment_method "$ASSIGNMENT_METHOD" \
        --token_train_stride "$TOKEN_TRAIN_STRIDE" \
        --use_trainable_patch_projector "$USE_TRAINABLE_PATCH_PROJECTOR" \
        --patch_encoder_dim "$PATCH_ENCODER_DIM" \
        --patch_bank_attn_dim "$PATCH_BANK_ATTN_DIM" \
        --use_patch_bank_attention "$USE_PATCH_BANK_ATTENTION" \
        --lambda_ce "$LAMBDA_CE" \
        --lambda_mse "$LAMBDA_MSE" \
        --lambda_align "$LAMBDA_ALIGN" \
        --lambda_smooth "$LAMBDA_SMOOTH" \
        --mse_temperature "$MSE_TEMPERATURE" \
        --align_temperature "$ALIGN_TEMPERATURE" \
        --residual_scale "$RESIDUAL_SCALE" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --gpt_local_path ./gpt \
        --lradj type1 \
        --seed "$SEED"
done

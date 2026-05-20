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
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
PATCH_TOKENIZER="native_gpt_vocab"
CANDIDATE_TOKEN_MODE="numeric"
NATIVE_TOKEN_K=4
PATCH_ENCODER_HIDDEN_DIM=512
PATCH_DECODER_HIDDEN_DIM=512
VQ_TAU=1.0
VQ_TAU_MIN=0.05
USE_STRAIGHT_THROUGH=1
USE_INPUTS_EMBEDS_FOR_TRAINING=1
COMMITMENT_LOSS_WEIGHT=0.25
USAGE_LOSS_WEIGHT=0.01
RECON_LOSS_WEIGHT=1.0
TOKEN_CE_LOSS_WEIGHT=1.0
FORECAST_LOSS_WEIGHT=1.0
TRAIN_STAGE="joint_train"
FREEZE_GPT_CODEBOOK=1


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
        --patch_tokenizer "$PATCH_TOKENIZER" \
        --candidate_token_mode "$CANDIDATE_TOKEN_MODE" \
        --native_token_k "$NATIVE_TOKEN_K" \
        --patch_encoder_hidden_dim "$PATCH_ENCODER_HIDDEN_DIM" \
        --patch_decoder_hidden_dim "$PATCH_DECODER_HIDDEN_DIM" \
        --vq_tau "$VQ_TAU" \
        --vq_tau_min "$VQ_TAU_MIN" \
        --use_straight_through "$USE_STRAIGHT_THROUGH" \
        --use_inputs_embeds_for_training "$USE_INPUTS_EMBEDS_FOR_TRAINING" \
        --commitment_loss_weight "$COMMITMENT_LOSS_WEIGHT" \
        --usage_loss_weight "$USAGE_LOSS_WEIGHT" \
        --recon_loss_weight "$RECON_LOSS_WEIGHT" \
        --token_ce_loss_weight "$TOKEN_CE_LOSS_WEIGHT" \
        --forecast_loss_weight "$FORECAST_LOSS_WEIGHT" \
        --train_stage "$TRAIN_STAGE" \
        --freeze_gpt_codebook "$FREEZE_GPT_CODEBOOK" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --gpt_local_path ./gpt \
        --lradj type1 \
        --seed "$SEED"
done

#!/bin/bash
# LoRA fine-tune AHA on the reset-decision task.
#
# Starts from the merged AHA model (scripts/merge_aha_lora.py) and trains a
# fresh LoRA on the ABD-collected reset-VQA train split
# (eval/build_ft_dataset.py). Run with the `aha` conda env.
#
#   bash scripts/finetune_aha_reset.sh [model_base] [data] [image_folder] [out] [gpus] [epochs]
set -e

cd /home/uhnam/workspaces/ABD/RoboPoint

MODEL_BASE=${1:-/ckpt/AHA/aha-merged-7b}
DATA_FILE=${2:-/data/abd/eval/aha_ft/train.json}
IMAGE_FOLDER=${3:-/data/abd/eval/reset_vqa}
OUT_DIR=${4:-/result/AHA/reset_ft}
GPUS=${5:-2,3}
NUM_EPOCHS=${6:-3}
DS_CONFIG=${7:-scripts/zero3.json}      # single-GPU: pass an offload config
BATCH=${8:-8}                           # per-device train batch size
ACC=${9:-1}                             # gradient accumulation steps

export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:$PATH
export DS_SKIP_CUDA_CHECK=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline

mkdir -p "$OUT_DIR"
echo "Fine-tuning AHA on reset detection"
echo "  base   : $MODEL_BASE"
echo "  data   : $DATA_FILE"
echo "  out    : $OUT_DIR"
echo "  gpus   : $GPUS   epochs: $NUM_EPOCHS"
echo "  ds_cfg : $DS_CONFIG   batch/acc: $BATCH/$ACC"

# No --pretrain_mm_mlp_adapter: the merged AHA model already carries a trained
# mm projector; train.py keeps it when the flag is absent.
deepspeed --include localhost:"$GPUS" robopoint/train/train_mem.py \
    --deepspeed "$DS_CONFIG" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path "$MODEL_BASE" \
    --version v1 \
    --data_path "$DATA_FILE" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$ACC" \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none

#!/usr/bin/bash
# pretrain
# export WANDB_PROJECT=scalinglaw
# export WANDB_ENTITY=296834189-guangdong-huarun-paints-co-
# export WANDB_API_KEY=78b6a1d47413854a43a66cce02c839f62e998fed
export TRANSFORMERS_CACHE=/home/caslx/scalinglaw/MINI_LLM/cache
export HF_DATASETS_CACHE=/home/caslx/scalinglaw/MINI_LLM/cache
# sft
export WANDB_PROJECT=dpo
export WANDB_ENTITY=296834189-guangdong-huarun-paints-co-
export WANDB_API_KEY=78b6a1d47413854a43a66cce02c839f62e998fed
# NCCL_DEBUG=INFO python dpo_train.py
NCCL_DEBUG=INFO accelerate launch --multi_gpu  --config_file accelerate_multi_gpu.yaml dpo_train.py
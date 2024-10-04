#!/bin/bash

python vit_trainer.py \
    --batch_size 64 \
    --num_steps 200 \
    --eval_steps 100 \
    --num_classes 10 \
    --image_size 256 \
    --max_iter 30 \
    --embed_dim 256 \
    --depth 12 \
    --num_heads 8 \
    --lr_embedding 1e-4 \
    --lr_rest 1e-4 \
    --optimizer_type adam \
    --warmup_steps 100 \
    --device cuda \
    --run_name 'fractal_vit_run_1' \
    --seed 0
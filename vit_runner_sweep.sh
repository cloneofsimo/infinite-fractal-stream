#!/bin/bash

WIDTHS=(64 128 256 512)
log_lrs=(-6 -5 -4 -3 -2 -1 0 1)


for seed in {0..3}; do
    for width in "${WIDTHS[@]}"; do
        for i in {0..7}; do
            lr=${log_lrs[$i]}
            this_device=0
            # new lr is 2**lr / width
            this_lr=$(python -c "print(2**${lr})")

            run_name="vit_run_width_${width}_lr_${this_lr}_seed_${seed}_dropout_0.0"

            num_heads=$(python -c "print(${width} // 16)")

            echo "Running ${run_name} with lr=${this_lr} and num_heads=${num_heads}"
            
            CUDA_VISIBLE_DEVICES=${this_device} python vit_trainer.py \
                --batch_size 128 \
                --num_steps 3000 \
                --eval_steps 100 \
                --num_classes 20 \
                --image_size 256 \
                --max_iter 30 \
                --embed_dim ${width} \
                --depth 12 \
                --num_heads ${num_heads} \
                --lr_embedding 1e-3 \
                --lr_base ${this_lr} \
                --optimizer_type adam \
                --warmup_steps 0 \
                --device cuda \
                --run_name ${run_name} \
                --seed ${seed} \
                --val_seed 0 \
                --num_samples_per_class 1000000 \
                --eval_once_every 200 \
                --proj_name "vit_sweep_step_1000_val_seed_0_kaiming_normal_2"
        done
    done
done

echo "All jobs completed."
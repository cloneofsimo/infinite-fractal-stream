

# sweep width and learning rate (lr-rest)

WIDTHS=(128 256 512 1024)
log_lrs=(-7 -6 -5 -4 -3 -2 -1 0)

for width in ${WIDTHS[@]}; do
    for lr in ${log_lrs[@]}; do
        # new lr is 2**lr / width
        this_lr=$(python -c "print(2**${lr} / ${width})")

        run_name="vit_run_${width}_${this_lr}"

        num_heads=$(python -c "print(${width} // 8)")


        echo "Running ${run_name} with lr=${this_lr} and num_heads=${num_heads}"
        # use device 1
        CUDA_VISIBLE_DEVICES=1 python vit_trainer.py \
            --batch_size 64 \
            --num_steps 3000 \
            --eval_steps 100 \
            --num_classes 1000 \
            --image_size 256 \
            --max_iter 30 \
            --embed_dim ${width} \
            --depth 12 \
            --num_heads ${num_heads} \
            --lr_embedding 1e-4 \
            --lr_rest ${this_lr} \
            --optimizer_type adam \
            --warmup_steps 100 \
            --device cuda \
            --run_name ${run_name} \
            --seed 0 \
            --num_samples_per_class 1000000 \
            --eval_once_every 200
    done
done
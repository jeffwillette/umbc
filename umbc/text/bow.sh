GPU=${1:-0}
SET_SIZE=${2:-200}
FINETUNE=${3:-True}
DS=${4:-eurlex}
LR=${5:-5e-5}
for RUN in 0 1 2 3 4
    do CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python umbc/text/trainer.py \
        --run $RUN \
        --gpu $GPU \
        --train_set_size $SET_SIZE \
        --test_set_size -1 \
        --agg False \
        --lr $LR \
        --opt adamw \
        --finetune $FINETUNE \
        --dataset $DS \
        --epochs 20 \
        --bow True 
done


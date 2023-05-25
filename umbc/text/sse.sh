GPU=${1:-0}
GRAD_SIZE=${2:-200}
FINETUNE=True
AGG=${3:-True}
ATTN_ACT=${4:-slot-sigmoid}
DS=${5:-eurlex}
LR=${6:-5e-5}

for RUN in 0 1 2 3 4
    do CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python umbc/text/trainer.py \
        --run $RUN \
        --gpu $GPU \
        --train_set_size -1 \
        --test_set_size -1 \
        --agg $AGG \
        --umbc_grad_size $GRAD_SIZE \
        --lr $LR \
        --opt adamw \
        --finetune $FINETUNE \
        --dataset $DS \
        --k 1 \
        --batch-size 8 \
        --epochs 20 \
        --heads 1 \
        --sse True 
done


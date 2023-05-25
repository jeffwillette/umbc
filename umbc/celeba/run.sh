GPU=${1:-0}
SET_SIZE=${2:-100}
MODEL=${3:-umbc}
K=${4:-128}
AGG=${5:-false}
MODE=${6:-train}

for RUN in 0 1 2 3 4
    do CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python umbc/celeba/trainer.py \
        --run $RUN \
        --gpu $GPU \
        --model $MODEL \
        --train_set_size $SET_SIZE \
        --test_set_size $SET_SIZE \
        --agg $AGG \
        --attn_act softmax \
        --mode $MODE \
        --k $K
done


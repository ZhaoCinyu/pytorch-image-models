# dense model
# MODEL=mobilevitv2_100 # not pretrained
# MODEL=mobilevitv2_100.cvnets_in1k # pretrained

# not pretrained
MODEL=mobilevitv2_100_moe

# pretrained, TODO: map the dense weights and partition into MoE
# MODEL=mobilevitv2_100_moe.cvnets_in1k

DATASET='hfids/uoft-cs/cifar10'

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset $DATASET --train-num-samples 50000 --val-split '' --input-key 'img' \
    --model $MODEL \
    --sched cosine --epochs 100 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 128 --amp -j 1 \
    # --pretrained \ # for pretrained model
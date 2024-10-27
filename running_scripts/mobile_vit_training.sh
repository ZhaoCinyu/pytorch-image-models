# dense model
# MODEL=mobilevitv2_100 # not pretrained
# MODEL=mobilevitv2_100.cvnets_in1k # pretrained

# not pretrained

HOME=/data/mufan/jie/pytorch-image-models

MODEL=mobilevitv2_100_moe

# pretrained, TODO: map the dense weights and partition into MoE
# MODEL=mobilevitv2_100_moe.cvnets_in1k

DATASET='hfids/uoft-cs/cifar10'

CUDA_VISIBLE_DEVICES=0 python mobilevit_training.py --num_epochs 1
    # --pretrained \ # for pretrained model
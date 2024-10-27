export HOME=/data/mufan/jie/pytorch-image-models

CUDA_VISIBLE_DEVICES=0 python src/param_cluster.py --model_path .cache/huggingface/hub/models--timm--mobilevitv2_100.cvnets_in1k/snapshots/75eb7f2bb7f9bd8662106fd0e43b675c0607f4dc/pytorch_model.bin \
                             --res_path model/mobilevitv2_100_s --num-layer 2,4,3 --num-expert 8\
                             --templates "stages.2.1.transformer.{}.mlp.fc1.weight,stages.3.1.transformer.{}.mlp.fc1.weight,stages.4.1.transformer.{}.mlp.fc1.weight"
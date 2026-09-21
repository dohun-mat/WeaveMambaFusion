#!/bin/bash
set -e

# conda 활성화
source /home/dhkim/anaconda3/etc/profile.d/conda.sh
conda activate dhk

echo "Using python: $(which python)"

echo "===== [1/2] Training eresnet ====="
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --standalone --nproc_per_node=1 train.py --network eresnet


echo ""
echo "===== [2/2] Testing on WIDER FACE (multi-scale) ====="
CUDA_VISIBLE_DEVICES=2 python test_widerface_multi_scale.py \
    --trained_model ./weights/eresnet_Final.pth \
    --network eresnet \
    --test_scales 0.5 1.0 1.5 2.0 \
    --do_flip

echo ""
echo "===== Done ====="

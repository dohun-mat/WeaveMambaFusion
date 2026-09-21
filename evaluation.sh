#!/bin/bash
set -e
# conda 활성화
source /home/dhkim/anaconda3/etc/profile.d/conda.sh
conda activate dhk
echo "Using python: $(which python)"

# 평가할 모델 목록 (Final 먼저, 그 다음 340, 330, 320, 310, 300)
MODELS=(
    "eresnet_best.pth"
    "eresnet_Final.pth"
    "eresnet_epoch_440.pth"
    "eresnet_epoch_430.pth"
    "eresnet_epoch_420.pth"
    "eresnet_epoch_410.pth"
    "eresnet_epoch_400.pth"
)

# 현재 작업 디렉토리 저장 (반복마다 돌아오기 위함)
ROOT_DIR=$(pwd)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "############################################################"
    echo "##### Evaluating: ${MODEL}"
    echo "############################################################"

    cd "${ROOT_DIR}"

    echo "===== [1/2] Test eresnet (${MODEL}) ====="
    CUDA_VISIBLE_DEVICES=2 python test_widerface_multi_scale.py \
        --trained_model ./weights/${MODEL} \
        --network eresnet \
        --test_scales 0.5 1.0 1.5 2.0 \
        --do_flip

    echo ""
    echo "===== [2/2] Evaluation (${MODEL}) ====="
    cd ./widerface_evaluate
    python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth

    # 다음 모델 평가를 위해 루트로 복귀
    cd "${ROOT_DIR}"

    echo ""
    echo "===== Done: ${MODEL} ====="
done

echo ""
echo "############################################################"
echo "##### All evaluations completed!"
echo "############################################################"

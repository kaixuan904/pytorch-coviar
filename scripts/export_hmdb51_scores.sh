#!/usr/bin/env bash
set -euo pipefail

cd ~/pytorch-coviar
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coviar310

mkdir -p results
export CUDA_VISIBLE_DEVICES=0

run_one () {
    split=$1
    rep=$2
    weight=$3

    echo "=========================================="
    echo "HMDB51 split${split} ${rep}"
    echo "weight: ${weight}"
    echo "=========================================="

    python -u test.py \
        --data-name hmdb51 \
        --representation "${rep}" \
        --data-root / \
        --test-list "./data/hmdb51/list_cvt/hmdb51_split${split}_test_coviar.txt" \
        --weights "${weight}" \
        --arch resnet18 \
        --save-scores "results/hmdb51_split${split}_${rep}_scores.npz" \
        2>&1 | tee "results/hmdb51_split${split}_${rep}_test.log"
}

# split1
run_one 1 iframe   "./hmdb51_split1_iframe_resnet18_iframe_model_best.pth.tar"
run_one 1 mv       "./hmdb51_split1_mv_resnet18_mv_model_best.pth.tar"
run_one 1 residual "./hmdb51_split1_residual_resnet18_residual_model_best.pth.tar"

# split2
run_one 2 iframe   "./hmdb51_split2_iframe_resnet18_iframe_model_best.pth.tar"
run_one 2 mv       "./hmdb51_split2_mv_resnet18_mv_model_best.pth.tar"
run_one 2 residual "./hmdb51_split2_residual_resnet18_residual_model_best.pth.tar"

# split3
run_one 3 iframe   "./hmdb51_split3_iframe_resnet18_iframe_model_best.pth.tar"
run_one 3 mv       "./hmdb51_split3_mv_resnet18_mv_model_best.pth.tar"
run_one 3 residual "./hmdb51_split3_residual_resnet18_residual_model_best.pth.tar"

echo
echo "========= DONE ========="
ls -lh results/hmdb51_split*_scores.npz

#!/bin/bash
set -e

cd ~/pytorch-coviar
mkdir -p results

for s in 1 2 3
do
  echo "===== split${s} iframe ====="
  python test.py \
    --data-name ucf101 \
    --representation iframe \
    --data-root /mnt/d/dataset/ucf101/mpeg4_videos \
    --test-list ./data/datalists/ucf101_split${s}_test.txt \
    --weights ./ucf101_split${s}_iframe_resnet18_iframe_model_best.pth.tar \
    --arch resnet18 \
    --save-scores results/ucf101_split${s}_iframe_scores.npz \
    | tee results/ucf101_split${s}_iframe_test.log

  echo "===== split${s} mv ====="
  python test.py \
    --data-name ucf101 \
    --representation mv \
    --data-root /mnt/d/dataset/ucf101/mpeg4_videos \
    --test-list ./data/datalists/ucf101_split${s}_test.txt \
    --weights ./ucf101_split${s}_mv_resnet18_mv_model_best.pth.tar \
    --arch resnet18 \
    --save-scores results/ucf101_split${s}_mv_scores.npz \
    | tee results/ucf101_split${s}_mv_test.log

  echo "===== split${s} residual ====="
  python test.py \
    --data-name ucf101 \
    --representation residual \
    --data-root /mnt/d/dataset/ucf101/mpeg4_videos \
    --test-list ./data/datalists/ucf101_split${s}_test.txt \
    --weights ./ucf101_split${s}_residual_resnet18_residual_model_best.pth.tar \
    --arch resnet18 \
    --save-scores results/ucf101_split${s}_residual_scores.npz \
    | tee results/ucf101_split${s}_residual_test.log
done

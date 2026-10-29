#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate coviar310
cd ~/pytorch-coviar

DATA_ROOT="/mnt/d/dataset/ucf101/mpeg4_videos"
ARCH="resnet18"
NUM_SEGMENTS=3
BATCH_SIZE=40
EPOCHS=500
WORKERS=2
LR=0.001
GPU_ID=0
LR_STEPS=200,300,400
for SPLIT in 1 2 3
do
  for REP in iframe mv residual
  do
    echo "======================================"
    echo "Start UCF101 split${SPLIT} ${REP} at $(date)"
    echo "======================================"

    python train.py \
      --data-name ucf101 \
      --data-root ${DATA_ROOT} \
      --train-list ./data/datalists/ucf101_split${SPLIT}_train.txt \
      --test-list ./data/datalists/ucf101_split${SPLIT}_test.txt \
      --representation ${REP} \
      --arch ${ARCH} \
      --num_segments ${NUM_SEGMENTS} \
      --batch-size ${BATCH_SIZE} \
      --epochs ${EPOCHS} \
      --workers ${WORKERS} \
      --lr ${LR} \
      --gpus ${GPU_ID} \
      --model-prefix ucf101_split${SPLIT}_${REP}_${ARCH}

    echo "Finish UCF101 split${SPLIT} ${REP} at $(date)"
  done
done

echo "All UCF101 jobs finished at $(date)"

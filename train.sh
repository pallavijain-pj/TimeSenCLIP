#!/bin/bash
# Train TimeSenCLIP
#
# Dataset: https://datapub.fz-juelich.de/sen4map/
# Pre-compute LUCAS embeddings first:
#   python scripts/precompute_embeddings.py \
#       --output_path ./data/embeddings/lucas_clipemb512_ViTB32.pt
#
# Adjust --device to your GPU index (0-based).

set -euo pipefail

python train.py \
    --root_data_dir    ./data/Sentinel2_Lucas/ \
    --emb_path         ./data/embeddings/lucas_clipemb512_ViTB32.pt \
    --sen_path         ./data/Benchmark_Path_Files/train_sentinel_paths.npy \
    --h5data_train_path ./data/sen4map/train.h5 \
    --h5data_val_path   ./data/sen4map/val.h5 \
    --version_fold     TimeSenCLIP_TSMixAug \
    --saved_model      ./checkpoints/ \
    --ts_arch          TimeSenCLIP \
    --BATCH_SIZE       64 \
    --NUM_WORKERS      8 \
    --NUM_EPOCHS       400 \
    --device           0 \
    --dropout_type     TSMixAug \
    --logit_learn
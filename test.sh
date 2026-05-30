#!/bin/bash
# Zero-shot evaluation with TimeSenCLIP
#
# Download weights first (requires huggingface-hub):
#   pip install huggingface-hub
#   huggingface-cli download YOUR_HF_USERNAME/TimeSenCLIP \
#       TimeSenCLIP_TSMixAug_encoder.pt --local-dir ./checkpoints/
#
# Or use push_to_hub.py to upload your own checkpoint.

set -euo pipefail

python zeroshot.py \
    --dataset_path  ./data/sen4map/test.h5 \
    --checkpoint    ./checkpoints/TimeSenCLIP_TSMixAug_encoder.pt \
    --input_resolution 1 \
    --BATCH_SIZE    512 \
    --NUM_WORKERS   8 \
    --device        cuda:0 \
    --version_fold  test \
    --ts_arch       TimeSenCLIP \
    --train_size    0.999 \
    --label_type    lc

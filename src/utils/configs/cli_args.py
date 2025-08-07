# File: config/cli_args.py

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--root_data_dir', type=str, default='./Datasets/Sentinel2_Lucas')
    parser.add_argument('--emb_path', type=str, default='./Datasets/Lucas_Frozen_Embeddings/lucas_clipemb512_ViTB32.pt')
    parser.add_argument('--sen_path', type=str, default='./Datasets/sentinel_paths.npy')
    parser.add_argument('--h5data_train_path', type=str, default='./Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split-wise/1x1_crops/train.h5')
    parser.add_argument('--h5data_val_path', type=str, default='./Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split-wise/1x1_crops/val.h5')
    parser.add_argument('--version_fold', type=str, default='v1')
    parser.add_argument('--ts_arch', type=str, default='TimeSenCLIP')
 
    parser.add_argument('--OPT', type=str, default='adamw')
    parser.add_argument('--LR', type=float, default=1e-4)
    parser.add_argument('--BATCH_SIZE', type=int, default=1024)
    parser.add_argument('--NUM_EPOCHS', type=int, default=300)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--input_resolution', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dropout_type', type=str, default='TSMixAug')
    parser.add_argument('--channels', nargs='+', default=['B4', 'B3', 'B2', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']) # Sentinel-2 bands
    parser.add_argument('--time_frames', type=int, default=12)
    parser.add_argument('--ARCH', type=str, default='ViT-B/32')
    parser.add_argument('--pooling', type=str, default='avgpool')
    parser.add_argument('--pool_out', type=int, default=512)
    parser.add_argument('--queue_size', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--logit_learn', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--saved_model', type=str, default='./checkpoints/')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_ckpt', type=str, default='last.ckpt')
    parser.add_argument('--id', type=str, default=None)

    return parser.parse_args()

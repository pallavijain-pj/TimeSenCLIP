# File: config/cli_args.py

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version_fold', type=str, default='v1')
    parser.add_argument('--ts_arch', type=str, default='TimeSenCLIP')
    parser.add_argument('--LOSS_TYPE', type=str, default='ce')
    parser.add_argument('--OPT', type=str, default='adamw')
    parser.add_argument('--LR', type=float, default=5e-5)
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--NUM_EPOCHS', type=int, default=30)
    parser.add_argument('--input_resolution', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--aug_type', type=str, default='default')
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--time_frames', type=int, default=8)
    parser.add_argument('--ARCH', type=str, default='RN50')
    parser.add_argument('--pooling', type=str, default='avg')
    parser.add_argument('--pool_out', type=int, default=512)
    parser.add_argument('--queue_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--logit_learn', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--saved_model', type=str, default='checkpoints/')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_ckpt', type=str, default='last.ckpt')
    parser.add_argument('--id', type=str, default=None)

    return parser.parse_args()

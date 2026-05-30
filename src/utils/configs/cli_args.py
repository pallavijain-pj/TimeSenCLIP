# File: src/utils/configs/cli_args.py

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="TimeSenCLIP: Temporal-Spectral CLIP for Remote Sensing Time Series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--root_dir", type=str, default="./",
        help="Root directory of the project",
    )
    parser.add_argument(
        "--root_data_dir", type=str, default="./data/Sentinel2_Lucas/",
        help="Path to LUCAS Sentinel-2 dataset directory",
    )
    parser.add_argument(
        "--emb_path", type=str,
        default="./data/embeddings/lucas_clipemb512_ViTB32.pt",
        help="Path to pre-computed LUCAS CLIP text embeddings (.pt file). "
             "Generate with: python scripts/precompute_embeddings.py",
    )
    parser.add_argument(
        "--sen_path", type=str,
        default="./data/Benchmark_Path_Files/train_sentinel_paths.npy",
        help="Path to Sentinel-2 training paths .npy file",
    )
    parser.add_argument(
        "--h5data_train_path", type=str,
        default="./data/sen4map/train.h5",
        help="Path to Sen4Map training HDF5 file",
    )
    parser.add_argument(
        "--h5data_val_path", type=str,
        default="./data/sen4map/val.h5",
        help="Path to Sen4Map validation HDF5 file",
    )
    parser.add_argument(
        "--saved_model", type=str, default="./checkpoints/",
        help="Directory where checkpoints are saved",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument("--ts_arch", type=str, default="TimeSenCLIP",
                        help="Temporal-spectral encoder architecture name")
    parser.add_argument("--ARCH", type=str, default="ViT-B/32",
                        help="CLIP backbone architecture")
    parser.add_argument(
        "--dropout_type", type=str, default="RandomTS",
        choices=["None", "RandomTS", "TSMixAug", "TSMS"],
        help="Temporal/spectral augmentation strategy",
    )
    parser.add_argument(
        "--channels", nargs="+",
        default=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        help="Sentinel-2 spectral bands to use",
    )
    parser.add_argument("--time_frames", type=int, default=12,
                        help="Number of temporal observations (months)")
    parser.add_argument("--input_resolution", type=int, default=1,
                        help="Spatial resolution of input patches (pixels)")
    parser.add_argument("--crop_size", type=int, default=1,
                        help="Crop size for dataset patches")
    parser.add_argument("--pooling", type=str, default="attpool_perimage",
                        choices=["avgpool", "attpool_perimage", "attpool_perdim"],
                        help="Pooling strategy for text embeddings")
    parser.add_argument("--pool_out", type=str, default="sum",
                        choices=["sum", "mean", "max"],
                        help="Aggregation method for attention pooling output")

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--OPT", type=str, default="adamw", help="Optimizer")
    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--BATCH_SIZE", type=int, default=1024)
    parser.add_argument("--NUM_EPOCHS", type=int, default=300)
    parser.add_argument("--NUM_WORKERS", type=int, default=8)
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index for training")
    parser.add_argument("--queue_size", type=int, default=2048,
                        help="Negative sample queue size for contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Contrastive loss temperature")
    parser.add_argument("--logit_learn", action="store_true",
                        help="Learn the logit scale parameter")
    parser.add_argument("--version_fold", type=str, default="v1",
                        help="Experiment name / checkpoint sub-directory")

    # ── Resume ───────────────────────────────────────────────────────────────
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--resume_ckpt", type=str, default="last.ckpt",
                        help="Checkpoint filename to resume from")
    parser.add_argument("--id", type=str, default=None,
                        help="Weights & Biases run ID for resuming a logged run")

    # ── Inference ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--return_coords", action="store_true",
        help="Include geographic coordinates in dataset outputs",
    )
    parser.add_argument("--train_size", type=float, default=0.999,
                        help="Fraction of data used as training split for inference")
    parser.add_argument(
        "--label_type", type=str, default="lc",
        choices=["lc", "lu", "crop", "bioregion", "eunis"],
        help="Label taxonomy to evaluate",
    )

    return parser.parse_args()

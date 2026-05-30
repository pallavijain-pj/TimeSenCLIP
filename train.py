

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.utils.configs.cli_args import get_args
from src.models.trainer import TimeSenCLIPLearner
from src.Data.dataloader import load_data
from src.utils.model_utils import before_load_weights
from src.utils.callbacks import get_callbacks
import numpy as np
import torch

def main(args):
    args.version_fold = f'{args.version_fold}'

    train_loader, val_loader, classes = load_data(args)
    
    wandb_logger = WandbLogger(name=f'{args.version_fold}', project='Granular_TS', id=args.id if args.resume else None)

    model = TimeSenCLIPLearner(args=args, classes=classes)

    
    if args.resume:
        before_load_weights(f'{args.saved_model}{args.version_fold}/{args.resume_ckpt}')

    trainer = pl.Trainer(
        accelerator='gpu', devices=[args.device],
        max_epochs=args.NUM_EPOCHS,
        track_grad_norm=2,
        detect_anomaly=True,
        accumulate_grad_batches=8,
        enable_progress_bar=True,
        callbacks=get_callbacks(args),
        amp_backend="apex", amp_level="O1",
        logger=wandb_logger,
        profiler="simple",
        resume_from_checkpoint=f'{args.saved_model}{args.version_fold}/{args.resume_ckpt}' if args.resume else None,
        # fast_dev_run = 20
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    args = get_args()
    main(args)

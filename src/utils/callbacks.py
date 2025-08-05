import os
def get_callbacks(args):
    base_path = os.path.join(args.saved_model, args.version_fold)
    return [
        ModelCheckpoint(
            monitor='train_loss',
            save_top_k=2,
            dirpath=base_path,
            filename=f'{args.version_fold}-{{epoch:02d}}-{{loss:.2f}}',
            every_n_epochs=10
        ),
        ModelCheckpoint(
            monitor='val_top1',
            mode='max',
            save_top_k=1,
            dirpath=base_path,
            filename=f'{args.version_fold}-{{epoch:02d}}-{{val_top1:.2f}}',
            save_weights_only=True,
            every_n_epochs=1
        ),
        ModelCheckpoint(
            monitor='Val_Avg_Cls_Acc',
            mode='max',
            save_top_k=1,
            dirpath=base_path,
            filename=f'{args.version_fold}-{{epoch:02d}}-{{Val_Avg_Cls_Acc:.2f}}',
            save_weights_only=True,
            every_n_epochs=1
        ),
        LearningRateMonitor(logging_interval='step')
    ]

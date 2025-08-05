# File: models/learner.py

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from src.utils.model_utils import model_config_load, pretrained_weights_ts
from src.models.model import CrossViewModel
from src.models.encoder import TimeSenCLIPEncoder as TimeSenCLIP
from src.Evaluation import metrics
from src.Evaluation.zeroshot_train_eval import val_run_tsms
import clip

class TimeSenCLIPLearner(pl.LightningModule):
    def __init__(self, args, classes):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.classes = classes

        self.batch_size = args.BATCH_SIZE
        self.learning_rate = args.LR
        self.loss_type = args.LOSS_TYPE
        self.optimizer_name = args.OPT
        self.queue_size = args.queue_size
        self.pooling = args.pooling
        self.logit_learn = args.logit_learn
        self.temperature = args.temperature

        if self.logit_learn:
            print(f"Learnable logit scale with initial temp: {self.temperature}")
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.tensor(np.log(1 / self.temperature), dtype=torch.float32)

        self.train_model_kwargs = model_config_load(aug_type=args.aug_type)
        self.val_model_kwargs = model_config_load(aug_type='None')

        self.TS_ViT = TimeSenCLIP(**self.train_model_kwargs).to(args.device)

        self.learner = CrossViewModel(
            image_size=args.input_resolution,
            embed_dim=1024 if args.ARCH == 'RN50' else 512,
            ts_seq=args.time_frames,
            channels=args.channels,
            architecture=args.ARCH,
            pooling=args.pooling,
            device=args.device,
            queue_size=args.queue_size,
            pool_out=args.pool_out,
            tsvit_model=self.TS_ViT,
        )
        torch.backends.cudnn.benchmark = True

    def shared_step(self, batch):
        logits, labels = self.learner.forward(batch)
        logit_scale = self.logit_scale.exp() if self.logit_learn else self.logit_scale
        logits *= logit_scale.to(logits.device)
        loss = F.cross_entropy(logits, labels)
        self.log("logit_scale", logit_scale, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        with torch.cuda.amp.autocast():
            batch = [tensor.to(self.args.device) for tensor in batch]
            loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_start(self):
        self.TS_ViT = TimeSenCLIP(self.val_model_kwargs).to(self.args.device)
        self.clip_model, _ = clip.load(self.args.ARCH, device="cpu")
        self.clip_model = self.clip_model.to(self.args.device).eval()
        self.top1_acc = 0.0
        self.top5_acc = 0.0
        self.total_samples = 0.0

        print("Loading weights...")
        model_state = self.learner.state_dict()
        message = self.TS_ViT.load_state_dict(pretrained_weights_ts(model_state, self.args.ts_arch), strict=True)
        self.TS_ViT = self.TS_ViT.to(self.args.device).eval()
        print(message)

    def validation_step(self, batch, batch_idx):
        batch = [tensor.to(self.args.device) for tensor in batch]
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        accuracy_scores = val_run_tsms(
            self.TS_ViT,
            self.clip_model,
            batch,
            self.classes,
            input_resolution=self.args.input_resolution,
            crop_size=self.args.crop_size,
            device=self.args.device,
        )

        self.top1_acc += accuracy_scores["total_correct_top1"]
        self.top5_acc += accuracy_scores["total_correct_top5"]
        self.total_samples += accuracy_scores["total_samples"]

    def validation_epoch_end(self, outputs):
        if self.total_samples == 0:
            return
        overall_top1 = self.top1_acc / self.total_samples
        overall_top5 = self.top5_acc / self.total_samples
        self.log("val_top1", overall_top1)
        self.log("val_top5", overall_top5)
        cls_results, avg_cls_acc = metrics.class_wise_accuracy(logits_stk, labels_stk, self.classes)

        for cls_name in self.classes:
            self.log(f"{cls_name}_Val_ACC", cls_results[cls_name]["Accuracy"])
        self.log("Val_Avg_Cls_Acc", avg_cls_acc)

        self.top1_acc = 0.0
        self.top5_acc = 0.0
        self.total_samples = 0.0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def on_before_zero_grad(self, *args, **kwargs):
        if self.logit_learn:
            self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)

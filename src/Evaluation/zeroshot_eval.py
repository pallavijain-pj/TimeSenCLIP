import os
import torch
import clip
import logging
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter

from Data import sen4map_data
from models.encoder import TimeSenCLIPEncoder as TimeSenCLIP
from Evaluation.zeroshot_train_eval import test_run_tsms
from utils.text_template import template_generic
from .metrics import class_wise_accuracy, accuracy

def get_args():
    parser = ArgumentParser(description='CLIP Temporal Spectral Model Inference', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default='./DATA/Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split_wise/9x9_crops/test_with_eunis.h5')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/TimeSenCLIP.ckpt')
    parser.add_argument('--input_resolution', type=int, default=9)
    parser.add_argument('--time_frames', type=int, default=12)
    parser.add_argument('--channels', nargs='+', default=['B4', 'B3', 'B2', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    parser.add_argument('--model_arch', type=str, default="TimeSenCLIP")
    parser.add_argument('--clip_arch', type=str, default='ViT-B/32')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

class ModelManager:
    def __init__(self, device):
        self.device = device

    def load_model(self, weight_path, model_kwargs):
        model = TimeSenCLIP(**model_kwargs).to(self.device).eval()
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = self._process_checkpoint(checkpoint['state_dict'])
        model.load_state_dict(state_dict, strict=True)
        logging.info(f"Loaded model from {weight_path}")
        return model

    def _process_checkpoint(self, checkpoint):
        return {
            k.replace('learner.TS_ViT.', ''): v
            for k, v in checkpoint.items()
            if k.startswith('learner.TS_ViT.')
        }

def main():
    args = get_args()

    # Load dataset
    dataset, _, _, classes = sen4map_data.dataloader_sen4map(
        args.dataset_path,
        test_path=None,
        crop_size=args.input_resolution,
        channels=args.channels,
        annual_composite=True if args.time_frames != 1 else False,
        train_size=0.999,
        label_type="labels",
        resize=args.input_resolution
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4)
    
    print(f"Loaded dataset with {len(dataset)} samples and {len(classes)} classes.")

    # Load models
    model_kwargs = {
        "image_size": args.input_resolution,
        "time_frames": args.time_frames,
        "dim": 512,
        "depth": 6,
        "heads": 8,
        "mlp_dim": 256,
        "spectral_bands": len(args.channels),
        "dim_head": 64,
        "dropout": 0.2,
        "training": False
    }

    model = ModelManager(args.device).load_model(args.checkpoint, model_kwargs)
    clip_model, _ = clip.load(args.clip_arch, device=args.device)
    clip_model.eval()

    template_name, template = list(template_generic.items())[0]

    # Evaluation loop
    print(f"Running inference using {template_name} template...")
    logits_list, labels_list = [], []
    top1_acc, top5_acc, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            acc_scores, logits, labels, _ = test_run_tsms(
                model, clip_model, batch, classes, template,
                time_frames=args.time_frames,
                channels=args.channels,
                device=args.device
            )

            top1_acc += acc_scores['total_correct_top1']
            top5_acc += acc_scores['total_correct_top5']
            total_samples += acc_scores['total_samples']

            logits_list.append(logits)
            labels_list.append(labels)

    # Final Metrics
    logits_stk = torch.cat(logits_list, dim=0)
    labels_stk = torch.cat(labels_list, dim=0)

    acc1, acc5 = accuracy(logits_stk, labels_stk, topk=(1, 5))
    class_results, avg_cls_acc = class_wise_accuracy(logits_stk, labels_stk, classes)

    print("\n=== Evaluation Results ===")
    print(f"Top-1 Accuracy (Normalized): {acc1.item():.2f}%")
    print(f"Top-5 Accuracy (Normalized): {acc5.item():.2f}%")
    print(f"Top-1 Accuracy (Raw): {(top1_acc / total_samples) * 100:.2f}%")
    print(f"Top-5 Accuracy (Raw): {(top5_acc / total_samples) * 100:.2f}%")
    print(f"Average Class Accuracy: {avg_cls_acc:.2f}%")
    print(f"Class-wise Accuracy:\n{class_results}")

if __name__ == '__main__':
    main()

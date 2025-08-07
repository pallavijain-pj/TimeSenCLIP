import os
import torch
import clip
import logging
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter

from src.Data import sen4map_data
from src.Data.dataloader import load_data
from src.models.encoder import TimeSenCLIPEncoder as TimeSenCLIP
from src.Evaluation.zeroshot_train_eval import test_run_tsms

from src.utils.prompt_template import template_generic
from src.Evaluation.metrics import class_wise_accuracy, accuracy

def get_args():
    parser = ArgumentParser(description='TimeSenCLIP Zeroshot Inference', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default='./Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split_wise/1x1_crops/test_with_eunis.h5')
    parser.add_argument('--h5data_testpath', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/TimeSenCLIP.ckpt')
    parser.add_argument('--input_resolution', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=1)
    parser.add_argument('--return_coords', type=bool, default=False)
    parser.add_argument('--train_size', type=float, default=0.999)
    parser.add_argument('--time_frames', type=int, default=12)
    parser.add_argument('--channels', nargs='+', default=['B4', 'B3', 'B2', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    parser.add_argument('--ts_arch', type=str, default="TimeSenCLIP")
    parser.add_argument('--clip_arch', type=str, default='ViT-B/32')
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--version_fold', type=str, default='test')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--label_type', type=str, default='lc', choices=['lc', 'lu', 'crop', 'bio','eunis'], help='Label type for evaluation')
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
    label_type = args.label_type
    # Load dataset
    dataloader, _,  classes = load_data(args, state='test')

    print(f"Loaded dataset with {len(dataloader.dataset)} samples and {len(classes)} classes.")

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
        "dropout_type": "None"
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
                device=args.device,
                label_type="lc",
                
            )

            top1_acc += acc_scores['total_correct_top1']
            top5_acc += acc_scores['total_correct_top5']
            total_samples += acc_scores['total_samples']


            logits_list.append(logits)
            labels_list.append(labels)

    # Final Metrics
    logits_stk = torch.cat(logits_list, dim=0)
    labels_stk = torch.cat(labels_list, dim=0)
    logits_stk = logits_stk / (logits_stk.mean(0,keepdim=True) + 1e-6)
    acc1, acc5 = accuracy(logits_stk, labels_stk, topk=(1, 5))
  
   
    class_results, avg_cls_acc = class_wise_accuracy(logits_stk, labels_stk, classes[label_type])

    print("\n=== Evaluation Results ===")
    print(f"Top-1 Accuracy (Normalized): {(acc1 / logits_stk.size(0)) * 100:.2f}%")
    print(f"Top-5 Accuracy (Normalized): {(acc5 / logits_stk.size(0)) * 100:.2f}%")
    print(f"Top-1 Accuracy (Raw): {(top1_acc / total_samples) * 100:.2f}%")
    print(f"Top-5 Accuracy (Raw): {(top5_acc / total_samples) * 100:.2f}%")
    print(f"Average Class Accuracy: {avg_cls_acc:.2f}%")
    print(f"Class-wise Accuracy:\n{class_results}")

if __name__ == '__main__':
    main()

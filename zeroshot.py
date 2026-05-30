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
from src.Evaluation.zeroshot_train_eval import test_zeroshot
from src.utils.model_utils import pretrained_weights_ts
from src.utils.prompt_template import template_generic
from src.Evaluation.metrics import class_wise_accuracy, accuracy

def get_args():
    parser = ArgumentParser(
        description="TimeSenCLIP — Zero-shot Inference",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument('--dataset_path', type=str, default='./Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split_wise/1x1_crops/test.h5')
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/TimeSenCLIP_1x1.ckpt",
        help="Path to model checkpoint (.ckpt or _encoder.pt). "
             "Download from HuggingFace Hub:\n"
             "  huggingface-cli download YOUR_HF_USERNAME/TimeSenCLIP",
    )'
    parser.add_argument('--input_resolution', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=1)
    parser.add_argument('--return_coords', type=bool, default=False)
    parser.add_argument('--train_size', type=float, default=0.999)
    parser.add_argument('--time_frames', type=int, default=12)
    parser.add_argument('--channels', nargs='+', default=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    parser.add_argument('--ts_arch', type=str, default="TimeSenCLIP")
    parser.add_argument('--clip_arch', type=str, default='ViT-B/32')
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--version_fold', type=str, default='test')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--label_type', type=str, default='lc', choices=['lc', 'lu', 'crop', 'bioregion','eunis'],  help="Label taxonomy to evaluate")
    return parser.parse_args()

class ModelManager:
    def __init__(self, device):
        self.device = device

    def load_model(self, weight_path, model_kwargs):
        model = TimeSenCLIP(**model_kwargs).to(self.device).eval()
        # print(model)
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = self._process_state_dict(checkpoint)
        message=model.load_state_dict(state_dict, strict=True)
        print(f"Model loaded with message: {message}")
        logging.info(f"Loaded model from {weight_path}")
        # save_model(self, model)
        return model
    def _process_state_dict(self, checkpoint):
        """Process checkpoint state_dict to extract TimeSenCLIP weights."""
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        return pretrained_weights_ts(checkpoint)
    
    
def main():
    args = get_args()
    label_type = args.label_type
    # Load dataset
    dataloader, classes = load_data(args, state='test')

    print(f"Loaded dataset with {len(dataloader.dataset)} samples and classes: {len(classes[label_type]) if isinstance(classes, dict) else len(classes)}")

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
    print(f"Using prompt template: {template_name},{template}")

    # Evaluation loop
    print(f"Running inference using {template_name} template...")
    logits_list, labels_list = [], []
    top1_acc, top5_acc, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            
            logits, labels, _ = test_zeroshot(
                model, clip_model, batch, classes, template,
                time_frames=args.time_frames,
                device=args.device,
                label_type=label_type
            )
           
            logits_list.append(logits)
            labels_list.append(labels)
            
          
    # Final Metrics
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels_list], dim=0)
    logits_norm = logits / (logits.mean(0,keepdim=True) + 1e-6)
    top1_norm, top5_norm = accuracy(logits_norm.float(), labels, topk=(1, 5))
    top1_acc, top5_acc = accuracy(logits.float(), labels, topk=(1, 5)) 
  
   
    class_results, avg_cls_acc = class_wise_accuracy(logits, labels, classes[label_type])

    print("\n=== Evaluation Results ===")
    print(f"Top-1 Accuracy (Normalized): {(top1_norm / logits.size(0)) * 100:.2f}%")
    print(f"Top-5 Accuracy (Normalized): {(top5_norm / logits.size(0)) * 100:.2f}%")
    print(f"Top-1 Accuracy (Raw): {(top1_acc / logits.size(0)) * 100:.2f}%")
    print(f"Top-5 Accuracy (Raw): {(top5_acc / logits.size(0)) * 100:.2f}%")
    print(f"Average Class Accuracy: {avg_cls_acc* 100:.2f}%")
    # print(f"Class-wise Accuracy:\n{class_results}")

if __name__ == '__main__':
    main()

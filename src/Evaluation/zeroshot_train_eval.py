import os
import json
import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from tqdm import tqdm
import clip
import logging
from src.utils.image_preprocessor import TimeSenCLIPPreprocessor
from src.Evaluation.metrics import accuracy
# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def zeroshot_classifier(model, classnames, templates, temp_context="class_context", device="cuda:3"):
    """
    Generate zero-shot classifier weights from CLIP using class name prompts.
    
    Args:
        model: CLIP model
        classnames: list of class labels
        templates: list or dict of text templates
        temp_context: type of template context
        device: CUDA device string

    Returns:
        torch.Tensor of zeroshot weights (num_classes x feature_dim)
    """
    zeroshot_weights = []

    with torch.no_grad():
        for i, classname in tqdm(enumerate(classnames), total=len(classnames)):
            # Select text templates
            if temp_context == "class_context":
                texts = templates[classname] if isinstance(templates, dict) else [templates[i].format(classname)]
            else:
                texts = [template.format(classname) for template in templates]

            # Tokenize and encode
            tokens = torch.cat([clip.tokenize(text) for text in texts]).to(device)
            class_embeddings = model.encode_text(tokens)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            # Average the text embeddings
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)

    return zeroshot_weights


def val_run_tsms(ts_model, clip_model, batch, classes, device="cuda:0", time_frames=12):
    """
    Evaluate temporal model against CLIP zero-shot classifier.

    Args:
        ts_model: Temporal model (e.g., transformer)
        clip_model: CLIP model for text prompts
        batch: Input batch of (ids, images, labels, coords)
        classes: List of class names
        device: CUDA device
        temporal: Temporal window (1, 4, or 12)
        

    Returns:
        Dictionary of accuracy scores and prediction metadata
    """
    templates = [
        'an image of {}.',
        'an image of a {}.',
        'an image of an {}.',
        'a land cover photo of {}.',
        'a land cover photo of a {}.',
        'a land cover photo of an {}.',
    ]

    acc_scores = {}


    encoded_prompts = zeroshot_classifier(clip_model, classes, templates, temp_context='unified', device=device).T

    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0

    _, images, labels = batch[:3]
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        B, S, T, H, W = images.shape

        # Temporal aggregation
        if time_frames == 1:
            images = images.median(dim=2, keepdim=True)[0]
        elif time_frames == 4:
            images = images.view(B, S, 4, 3, H, W).median(dim=3)[0]

        # Encode with TS model
        image_ts_features = ts_model.inference(images)
        image_ts_features /= image_ts_features.norm(dim=-1, keepdim=True)

        # Similarity with text prompts
        logits = (image_ts_features @ encoded_prompts) / 2 + 1

        correct_top1, correct_top5 = accuracy(logits, labels, topk=(1, 5))
        total_correct_top1 += correct_top1
        total_correct_top5 += correct_top5
        total_samples += images.size(0)

    acc1 = total_correct_top1 / total_samples
    acc5 = total_correct_top5 / total_samples

    print(f"Top-1 Accuracy: {acc1:.4f}, Top-5 Accuracy: {acc5:.4f}")

    acc_scores = {
        'total_correct_top1': total_correct_top1,
        'total_correct_top5': total_correct_top5,
        'total_samples': total_samples,
        'logits': logits,
        'labels': labels
    }

    return acc_scores





def test_zeroshot(ts_model, clip_model, batch, classes, 
                              template, label_type,  time_frames=12
                              , device ="cuda:3", return_embed=False):
    
    cls_type = label_type
    
    acc_scores = {}
    logits_list = []
    labels_list = []
   
    if type(template) is dict:
        temp_context='class_context'
    else:
        temp_context='unified'


    cls_list = classes[label_type] if isinstance(classes, dict) else classes
    if 'nuts' in label_type:
        cls_list = [nuts_labels[cls] for cls in cls_list if cls in nuts_labels]
        # print(cls_list)
    encoded_prompts = zeroshot_classifier(clip_model, cls_list, template, temp_context=temp_context, device = device).T

    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0

    images = batch['image'].to(device) if 'image' in batch else batch['embedding'].to(device)
    
    if 'bio_region' in label_type:
        labels = batch['bioregion_label']#.to(device)
    elif 'eunis' in label_type:
        labels = batch['eunis_label']#.to(device)
    elif 'nuts' in label_type:
        labels = [nuts_labels[nut] for nut in  batch['nuts']]
    elif 'crops' in label_type:
        labels = batch['crop_label']#.to(device)
    elif 'lu' in label_type:
        labels = batch['lu_label']#.to(device)
    else:
        labels = batch['lc_label']#.to(device)
        
    label2idx = {label: idx for idx, label in enumerate(cls_list[label_type])} if isinstance(cls_list, dict) else {label: idx for idx, label in enumerate(cls_list)}
    if isinstance(labels[0], str):  # If labels are strings
        labels = torch.tensor([label2idx[lbl] for lbl in labels], device=device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        
        if images.dim() == 5:
            B, T, S, H, W = images.size()
        else:
            B, T, D = images.size()
        
        if time_frames == 1 and T==12:
            images = images.median(dim=1, keepdim=True)[0]
        elif time_frames == 4 and T==12:
            images = images.view(B, 4, 3, S, H, W).median(dim=2)[0] if images.dim() == 5 else images.view(B, 4, 3, D).median(dim=2)[0]

        # images = images[:, :time_frames, :, :, :]
        image_ts_features = ts_model(images)
        image_ts_features /= image_ts_features.norm(dim=-1, keepdim=True)
        # logits = 100.0 * image_ts_features @ encoded_prompts
        logits = (image_ts_features @ encoded_prompts)/2+1
        logits_list= logits.cpu()
        labels_list=labels
        

    return logits_list, labels_list, encoded_prompts

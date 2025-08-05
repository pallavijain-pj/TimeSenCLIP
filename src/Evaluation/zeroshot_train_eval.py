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


def val_run_tsms(ts_model, clip_model, batch, classes, device="cuda:0", temporal=12):
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
        if temporal == 1:
            images = images.median(dim=2, keepdim=True)[0]
        elif temporal == 4:
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

def test_run_tsms(
    ts_model,
    clip_model,
    batch,
    classes,
    template,
    time_frames=12,
    channels=None,
    device="cuda:3",
    clip_eval=False,
):
   
    # Set logging level if not already configured
    logging.basicConfig(level=logging.INFO)

    acc_scores = {}
    logits_list = []
    labels_list = []

    # Determine template context
    temp_context = 'class_context' if isinstance(template, dict) else 'unified'
   
    cls_list = classes

    # Generate zero-shot classifier prompts
    encoded_prompts = zeroshot_classifier(
        clip_model, cls_list, template, temp_context=temp_context, device=device
    ).T  # Transpose to match (embedding_dim, num_classes)

    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0

    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    logging.debug(f"Input images shape: {images.shape}")

    
    with torch.no_grad(), torch.cuda.amp.autocast():
        if time_frames == 4:
            # Temporal median over 3 channels per time step
            images = images.view(images.size(0), 4, 3, *images.shape[2:]).median(dim=2)[0]
            
        preprocessor = TimeSenCLIPPreprocessor(device=device)

        # Get temporal-spectral features
        image_ts_features = ts_model(images)
        image_ts_features = F.normalize(image_ts_features, dim=-1)  # Normalize embeddings

        # Compute logits (adjusted similarity score)
        logits = (image_ts_features @ encoded_prompts) / 2 + 1

        predictions = torch.argmax(logits, dim=1)
        logging.debug(f"Predicted: {predictions}")
        logging.debug(f"Ground Truth: {labels}")

        logits_list.append(logits)
        labels_list.append(labels)

        # Compute top-1 and top-5 accuracy
        correct_top1, correct_top5 = accuracy(logits, labels, topk=(1, 5))

        total_correct_top1 += correct_top1
        total_correct_top5 += correct_top5
        total_samples += labels.size(0)

    top1_acc = total_correct_top1 / total_samples
    top5_acc = total_correct_top5 / total_samples
    logging.info(f"[{cls_type}] Top-1 Acc: {top1_acc:.4f}, Top-5 Acc: {top5_acc:.4f}")

    # Store results
    acc_scores = {
        'total_correct_top1': total_correct_top1,
        'total_correct_top5': total_correct_top5,
        'total_samples': total_samples
    }

    return acc_scores, logits_list, labels_list, encoded_prompts

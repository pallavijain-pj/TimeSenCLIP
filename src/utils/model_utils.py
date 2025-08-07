import os
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader


def replace_placeholders(config_section, replacements):
    for key, value in config_section.items():
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            config_section[key] = replacements.get(value[1:-1], value)
        elif isinstance(value, dict):
            replace_placeholders(value, replacements)

def model_config_load(args, dropout_type='None'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    replacements = {
        'input_resolution': args.input_resolution,
        "time_frames": args.time_frames,
        'num_channels': len(args.channels),
        'device': args.device,
        'dropout_type': dropout_type
    }

    model_config = config[args.ts_arch]
    replace_placeholders(model_config, replacements)
    print(f"Model Config: {model_config}")
    return model_config

def before_load_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items()
                                if not (k.startswith('ts_encoder.') or k.startswith('clip_model'))}
    torch.save(checkpoint, checkpoint_path)


def pretrained_weights_ts(checkpoint):
    return {k.replace('ts_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('ts_encoder.')}

def pretrained_weights_val(checkpoint):
    """Process checkpoint to extract relevant weights."""
    # Filter the state_dict to exclude TS_ViT and clip_model
    filtered_state_dict = {
        k: v for k, v in checkpoint.items()
        if not (k.startswith('ts_encoder.') or k.startswith('clip_model'))
    }
    checkpoint = filtered_state_dict
    return {key.replace('learner.ts_encoder.', ''): checkpoint[key] for key in list(checkpoint.keys()) if key.strip().startswith('learner.ts_encoder.')}
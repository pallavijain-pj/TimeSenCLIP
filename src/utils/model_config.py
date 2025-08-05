import os
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

def before_load_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items()
                                if not (k.startswith('TS_ViT.') or k.startswith('clip_model'))}
    torch.save(checkpoint, checkpoint_path)


def replace_placeholders(config_section, replacements):
    for key, value in config_section.items():
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            config_section[key] = replacements.get(value[1:-1], value)
        elif isinstance(value, dict):
            replace_placeholders(value, replacements)

def model_config_load(args, ts_dropout=False, ms_dropout=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs', 'model_configs.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    replacements = {
        'ARCH': args.ARCH,
        'input_resolution': args.input_resolution,
        "time_frames": args.time_frames,
        'num_channels': len(args.channels),
        'device': args.device,
        'aug_type': args.aug_type,
    }

    model_config = config[args.ts_arch]
    replace_placeholders(model_config, replacements)
    return model_config

def pretrained_weights_ts(checkpoint):
    return {k.replace('TS_ViT.', ''): v for k, v in checkpoint.items() if k.startswith('TS_ViT.')}

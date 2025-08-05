# dataset_utils.py
import torch
import time
import numpy as np
import h5py

# Sentinel-2 mean/std for 10 bands
SENTINEL_MEAN = [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27]
SENTINEL_STD =  [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45]
BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']


def load_embeddings(emb_path):
    print("Loading embedding dictionary...")
    start = time.time()
    try:
        emb_dict = torch.load(emb_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Error loading embedding file: {e}")
    print(f"Loaded embeddings in {(time.time() - start) / 60:.2f} min")
    return emb_dict


def normalize_tensor(tensor, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return (tensor - mean) / (std + 1e-6)

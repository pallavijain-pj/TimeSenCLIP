import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils,  io
from torch import nn

import cv2 
from sklearn import preprocessing
import rasterio as rio
import numpy as np
from PIL import Image
import random, os, time
import clip
from torchvision import utils
from tqdm import tqdm
import h5py
from .label_mapping import labels, crop_labels, class_list_lc

"""
lulc = ['lc1', 'lc1_label', 'lu1', 'lu1_label']

"""

"""
Mean across RGB channels: tensor([0.5057, 0.4965, 0.5013])
Standard deviation across RGB channels: tensor([0.2457, 0.2420, 0.2416])

Mean across RGB channels: tensor([0.2150, 0.2181, 0.1285])
Standard deviation across RGB channels: tensor([0.0992, 0.0700, 0.0569])
"""

def load_embeddings(emb_path):
        start_time = time.time()
        print("Loading embedding dictionary...")
        try:
            emb_dict = torch.load(emb_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading embedding file: {e}")
        print(f"Embedding dictionary loaded in {(time.time() - start_time) / 60:.2f} minutes")
        return emb_dict

class BaseSen4MapDataset(Dataset):
    def __init__(self, h5data_path: str, crop_size: int = 1, channels: list = None, 
                 annual_composite: bool = True, resize: int = 224, device: str = "cpu"):
        
        self.bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        self.mean = [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27]
        self.std = [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45]

    
        self.resize = resize
        self.crop_size = crop_size
        self.channels = channels if channels is not None else self.bands
        self.band_idx = [self.bands.index(channel) for channel in self.channels]
        self.annual_composite = annual_composite
        self.device = device

        self._load_h5data(h5data_path)
        # self._load_embeddings(emb_path)

    def _load_h5data(self, h5data_path):
        start_time = time.time()
        print("Loading H5Data...")
        try:
            self.h5data_f = h5py.File(h5data_path, 'r')
            self.h5data = list(self.h5data_f.keys())
        except Exception as e:
            raise RuntimeError(f"Error opening HDF5 file: {e}")
        print(f"H5Data loaded in {time.time() - start_time:.2f} seconds")

   
    def __len__(self):
        return len(self.h5data)

    def crop_center(self, img: torch.Tensor, cropx: int, cropy: int):
        """Crop the center of the image."""
        c, t, y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[:, :, starty:starty + cropy, startx:startx + cropx]

    def calculate_annual_composite(self, image: torch.Tensor, image_ids: list):
        """Generate annual composite image."""
        months = [f'2018{str(i).zfill(2)}' for i in range(1, 13)]
        month_composites = []

        for month in months:
            indices = [i for i, img_id in enumerate(image_ids) if month in img_id]
            if not indices:
                indices = range(image.shape[1])

            monthly_stack = torch.stack([image[:, i, :, :] for i in indices])
            month_composites.append(torch.median(monthly_stack, dim=0)[0])

        return torch.stack(month_composites, dim=0) + 1

    def get_data(self, im):
        """Get and preprocess the image data."""
        mask = torch.tensor(im['SCL'] < 9, dtype=torch.bool)
        image = torch.stack([
            torch.tensor(np.where(mask.cpu(), im[channel].astype(np.float32), 0), dtype=torch.float32)
            for channel in self.channels
        ])

        composite_image = (
            self.calculate_annual_composite(image, im.attrs['Image_ID'].tolist())
            if self.annual_composite
            else torch.median(image, dim=1, keepdim=True).values.permute(1, 0, 2, 3)
        )

        if self.crop_size < composite_image.shape[2]:
            composite_image = self.crop_center(composite_image, self.crop_size, self.crop_size)

        return composite_image


        

class Sen4MapDataset(BaseSen4MapDataset):
    def __init__(self, h5data_path: str, sen_paths: str, emb_dict: dict, crop_size: int = 1, lulc_type: str = 'lc1', channels: list = None, annual_composite: bool = True, resize: int = 128, label_type: str = 'labels', device: str = 'cpu'):
        super().__init__(h5data_path, crop_size, channels, annual_composite, resize)

        path_sen = np.load(sen_paths)
        paths_sen = [f'Lucas_Point_{path.split("/")[-1].split(".")[0]}' for path in path_sen]
        self.h5data = list(set(self.h5data) & set(paths_sen))
        self.emb_dict = emb_dict
        self.label_map = labels if label_type == "labels" else crop_labels
        self.lulc_type = lulc_type
        self.classes = class_list_lc

    def _transform(self, image):
        mean = [self.mean[i] for i in self.band_idx]
        std = [self.std[i] for i in self.band_idx]
    
        transform = transforms.Compose([
                transforms.Normalize(mean, std), 
            ])
        return transform(image)

    def _transform_resize(self, image):
        transform = transforms.Compose([
                transforms.Resize((self.resize,self.resize), interpolation= transforms.InterpolationMode.BILINEAR),
            ])
        return transform(image)
    def __getitem__(self, index: int):
        im_id = self.h5data[index]
        im = self.h5data_f[im_id]
        nuts = im.attrs['nuts0']
       
        emb_id = f"{nuts}_{im_id.split('_')[-1]}"
        embeddings = [self.emb_dict[f'{emb_id}{dir}'] for dir in ['W', 'E', 'N', 'S']]
        frozen_emb = torch.cat(embeddings, dim=0)

        image = self.get_data(im)
      
        image = self._transform(image)
        
        label = self.label_map[im.attrs[self.lulc_type]]
        label = torch.tensor(label, dtype=torch.long)
        return frozen_emb, image, label

    


 
class Sen4MapDataset_1x1(BaseSen4MapDataset):
    def __init__(self, h5data_path: str, sen_paths: str, emb_dict: dict, lulc_type: str = 'lc1', channels: list = None, annual_composite: bool = True, label_type: str = 'labels', device: str = 'cpu', return_coords: bool = False):
        super().__init__(h5data_path, 1, channels, annual_composite, 1)

   
        rem_keys = ['Lucas_Point_45061526', "Lucas_Point_45281654", "Lucas_Point_47501690"]
        path_sen = np.load(sen_paths)
        paths_sen = [f'Lucas_Point_{path.split("/")[-1].split(".")[0]}' for path in path_sen]
        self.h5data = list(set(self.h5data) & set(paths_sen))
        self.h5data = list(set(self.h5data) - set(rem_keys))
        self.emb_dict = emb_dict
        self.label_map = labels if label_type == "labels" else crop_labels
        self.lulc_type = lulc_type
        self.classes = class_list_lc
        self.return_coords = return_coords
        
    def __len__(self):
        return len(self.h5data)
    def _transform(self, image):
      
        mean = [self.mean[i] for i in self.band_idx]
        std = [self.std[i] for i in self.band_idx]
        
        transform = transforms.Compose([
                transforms.Normalize(mean, std), 
            ])
        return transform(image)
    def __getitem__(self, index):
        im_id = self.h5data[index]
        im = self.h5data_f[im_id]
        nuts = im.attrs['nuts0']

        emb_id = f"{nuts}_{im_id.split('_')[-1]}"
        embeddings = [self.emb_dict[f'{emb_id}{dir}'] for dir in ['W', 'E', 'N', 'S']]
        frozen_emb = torch.cat(embeddings, dim=0)

        image = im['image'][:]
        if len(self.channels) < 10:
            image = self.stack_selected_channels(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        
        if not self.annual_composite:
            image = self.get_composite(image)
     
        image = self._transform(image)
        image = torch.clamp(image, 0.0, 1.0)
        

        label = self.label_map[im.attrs[self.lulc_type]]
        label = torch.tensor(label, dtype=torch.long)
        if self.return_coords:
            coords = im.attrs['Coordinates']
            coords = torch.tensor(coords, dtype=torch.float32)
            return frozen_emb, image, label, coords
        return frozen_emb, image, label

    def stack_selected_channels(self, image):
        """
        Stack only the selected channels.
        """
        selected_channels =  [torch.tensor(image[:, i, :, :], dtype=torch.float32) for i in self.band_idx]
        stacked_image = torch.stack(selected_channels, dim=1)
        return stacked_image  

    def get_composite(self, im):
        composite_image = torch.median(im, dim=0, keepdim=True).values#.permute(1, 0, 2, 3)
        return composite_image  # Return composite image tensor



def dataset_sen4map(h5data_path: str, sen_path: str, emb_path: str, test_path: str = None, lulc_type: str = 'lc1', crop_size: int = 1, channels: list = None, annual_composite: bool = True, label_type: str = 'labels', resize: int = None, device: str = 'cpu',return_coords: bool = False):
    
    emb_dict = load_embeddings(emb_path)
       
    if crop_size>1 and crop_size!=9 and crop_size!=5:
        train_dataset = Sen4MapDataset(h5data_path, sen_path, emb_dict, crop_size=crop_size, lulc_type=lulc_type, channels=channels, annual_composite=annual_composite, resize=resize, label_type=label_type, device=device)
    else:
        train_dataset = Sen4MapDataset_1x1(h5data_path, sen_path, emb_dict, lulc_type=lulc_type, channels=channels, annual_composite=annual_composite, label_type=label_type, device=device, return_coords=return_coords)
    print(f"Train Dataset: {len(train_dataset)}")

    if test_path:
        if crop_size>1 and crop_size!=9 and crop_size!=5:
            test_dataset = Sen4MapDataset(test_path, sen_path, emb_dict, crop_size=crop_size, lulc_type=lulc_type, channels=channels, annual_composite=annual_composite, resize=resize, label_type=label_type, device=device)
        else:
            test_dataset = Sen4MapDataset_1x1(test_path, sen_path, emb_dict, lulc_type=lulc_type, channels=channels, annual_composite=annual_composite, label_type=label_type, device=device, return_coords=return_coords)

        class_list = test_dataset.classes
        print(f"Test Dataset: {len(test_dataset)}")
        return train_dataset, test_dataset, class_list

    return train_dataset



 


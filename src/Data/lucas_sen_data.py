# sen4map_dataset.py
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from .dataset_utils import SEN4MAP_SENTINEL_MEAN, SEN4MAP_SENTINEL_STD, BAND_NAMES, load_embeddings, normalize_tensor
from .lucas_label_mapping import labels, crop_labels, class_list_lc


class BaseSen4MapDataset(Dataset):
    def __init__(self, h5data_path, crop_size=1, channels=None, annual_composite=True, resize=224, device="cpu"):
        self.channels = channels or BAND_NAMES
        self.band_idx = [BAND_NAMES.index(ch) for ch in self.channels]
        self.crop_size = crop_size
        self.resize = resize
        self.annual_composite = annual_composite
        self.device = device
        self.mean = [SEN4MAP_SENTINEL_MEAN[i] for i in self.band_idx]
        self.std = [SEN4MAP_SENTINEL_STD[i] for i in self.band_idx]
        self._load_h5data(h5data_path)

    def _load_h5data(self, h5_path):
        print("Loading HDF5 data...")
        self.h5data_f = h5py.File(h5_path, 'r')
        self.h5data = list(self.h5data_f.keys())
        print(f"Loaded {len(self.h5data)} samples.")

    def crop_center(self, img, cropx, cropy):
        _, t, y, x = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[:, :, starty:starty + cropy, startx:startx + cropx]

    def calculate_annual_composite(self, img, image_ids):
        months = [f'2018{str(m).zfill(2)}' for m in range(1, 13)]
        composites = []

        for month in months:
            idxs = [i for i, id_ in enumerate(image_ids) if month in id_]
            idxs = idxs if idxs else list(range(img.shape[1]))
            stack = torch.stack([img[:, i, :, :] for i in idxs])
            composites.append(torch.median(stack, dim=0)[0])

        return torch.stack(composites) + 1

    def get_data(self, im):
        mask = torch.tensor(im['SCL'] < 9, dtype=torch.bool)
        image = torch.stack([
            torch.tensor(np.where(mask, im[ch], 0), dtype=torch.float32) for ch in self.channels
        ])
        img = self.calculate_annual_composite(image, im.attrs['Image_ID'].tolist()) \
            if self.annual_composite else torch.median(image, dim=1, keepdim=True).values.permute(1, 0, 2, 3)

        if self.crop_size < img.shape[2]:
            img = self.crop_center(img, self.crop_size, self.crop_size)

        return normalize_tensor(img, self.mean, self.std)

    
class CrossViewDataset(BaseSen4MapDataset):
    def __init__(self, h5data_path, sen_paths, emb_dict, lulc_type='lc1', channels=None,
                 annual_composite=True, label_type='labels', device='cpu', return_coords=False):
        super().__init__(h5data_path, 1, channels, annual_composite, 1, device)

        paths_sen = [f'Lucas_Point_{p.split("/")[-1].split(".")[0]}' for p in np.load(sen_paths)]
        rem_keys = ['Lucas_Point_45061526', "Lucas_Point_45281654", "Lucas_Point_47501690"]
        self.h5data = list(set(self.h5data) & set(paths_sen))
        self.h5data = list(set(self.h5data) - set(rem_keys))
        self.emb_dict = emb_dict
        self.label_map = labels if label_type == "labels" else crop_labels
        self.lulc_type = lulc_type
        self.classes = class_list_lc
        self.return_coords = return_coords
    def __len__(self):
        return len(self.h5data)

    def __getitem__(self, idx):
        im_id = self.h5data[idx]
        im = self.h5data_f[im_id]
        nuts = im.attrs['nuts0']
        emb_id = f"{nuts}_{im_id.split('_')[-1]}"
        embeddings = torch.cat([self.emb_dict[f"{emb_id}{d}"] for d in ['W', 'E', 'N', 'S']], dim=0)

        image = im['image'][:]
        image = torch.tensor(image, dtype=torch.float32)
        image = self.get_composite(image) if not self.annual_composite else image
        image = normalize_tensor(image, self.mean, self.std)
        image = torch.clamp(image, 0, 1)

        label = torch.tensor(self.label_map[im.attrs[self.lulc_type]], dtype=torch.long)

        if self.return_coords:
            coords = torch.tensor(im.attrs['Coordinates'], dtype=torch.float32)
            return embeddings, image, label, coords
        return embeddings, image, label

    def get_composite(self, img):
        return torch.median(img, dim=0, keepdim=True).values
    

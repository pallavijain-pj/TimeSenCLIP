#----------------------------------------------------------
# File: Consists Sen4Map data loader and label mapping for the Lucas dataset.
#-----------------------------------------------------------
import torch
import h5py
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split

from .lucas_label_mapping import (
    labels, crop_labels, class_list_lc, lu_labels, class_list_lu
    
)
from .lucas_class_mapping import (
    LC1_class_list, LU1_class_list, LU1_class_map, LC1_class_map,
    bio_class_list, crop_class_map, crop_class_list, eunis_class_list, nuts_labels, class_list_nuts
)
from .dataset_utils import SEN4MAP_SENTINEL_MEAN, SEN4MAP_SENTINEL_STD, BAND_NAMES

class BaseSen4MapDataset(Dataset):
    def __init__(self, h5data_path, channels, mean, std, annual_composite=True, transform=True):
        self.h5data_f = h5py.File(h5data_path, 'r')
        self.h5data = list(self.h5data_f.keys())
        self.channels = channels
        self.band_idx = [self.all_bands().index(c) for c in channels]
        self.mean = [SEN4MAP_SENTINEL_MEAN[i] for i in self.band_idx]
        self.std = [SEN4MAP_SENTINEL_STD[i] for i in self.band_idx]
        self.annual_composite = annual_composite
        self.transform = transform

    def __len__(self):
        return len(self.h5data)

    def all_bands(self):
        return BAND_NAMES

    def _transform(self, image):
        mean = [self.mean[i] for i in self.band_idx]
        std = [self.std[i] for i in self.band_idx]
        normalize = transforms.Normalize(mean, std)
        return normalize(image)

    def stack_selected_channels(self, image):
        selected = [torch.tensor(image[:, i, :, :], dtype=torch.float32) for i in self.band_idx]
        return torch.stack(selected, dim=1)

    def get_composite_image(self, image):
        return torch.median(image, dim=0, keepdim=True).values

    def fetch_coords(self, im_id):
        return self.h5data_f[im_id].attrs['Coordinates']

    def fetch_idx(self, im_id):
        return self.h5data.index(im_id)

    def key_coord(self):
        return {k: self.h5data_f[k].attrs['Coordinates'] for k in self.h5data}


class Sen4MapDataset_withlabels(BaseSen4MapDataset):
    def __init__(self, 
                 h5data_path, 
                 channels=None, 
                 annual_composite=True, 
                 transform=True, 
                 return_coords=False, 
                 label_type='all'):
        """
        label_type: 'all', 'crop', 'lc', 'lu'
        """
        channels = channels or ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        super().__init__(h5data_path, channels, SEN4MAP_SENTINEL_MEAN, SEN4MAP_SENTINEL_STD, annual_composite, transform)
        
        self.return_coords = return_coords
        self.label_type = label_type.lower()
        self.bio_classes = bio_class_list
        self.eunis_classes = eunis_class_list
        self.classes = {
            "bio": bio_class_list,
            "eunis": eunis_class_list,
            "nuts": class_list_nuts
        }
        # Set label maps and class lists
        if self.label_type == 'crop':
            self.label_map = crop_class_map
            self.classes = crop_class_list
            crop_keys, nuts_list = self.crop_key_filter()
            self.h5data = crop_keys
            self.nuts_list = list(nuts_list)
            self.classes["crop"] = crop_class_list
        elif self.label_type in ['all', 'lc', 'lu']:
            self.lc_label_map = labels
            self.lu_label_map = lu_labels
            self.lc_classes = class_list_lc
            self.lu_classes = class_list_lu
            self.nuts_list = class_list_nuts
            self.classes["lc"] = class_list_lc
            self.classes["lu"] = class_list_lu
            
        

    def crop_key_filter(self):
        key_lists = []
        nuts_set = set()
        class_counts = defaultdict(int)

        for im_id in set(self.h5data):
            im = self.h5data_f[im_id]
            lc1_label = im.attrs['lc1_label']
            nuts_set.add(im.attrs['nuts0'])
            if lc1_label in self.classes:
                key_lists.append(im_id)
                class_counts[lc1_label] += 1

        print("NUTS Regions:", nuts_set)
        print("Crop Class Counts:", dict(class_counts))
        return key_lists, nuts_set

    def __getitem__(self, index):
        im_id = self.h5data[index]
        im = self.h5data_f[im_id]

        image = torch.tensor(im['image'][:], dtype=torch.float32)
        if len(self.channels) < 10:
            image = self.stack_selected_channels(im['image'][:])
        if not self.annual_composite:
            image = self.get_composite_image(image)
        if self.transform:
            image = self._transform(image)
            image = torch.clamp(image, 0, 1)

        result = {
            "image": image,
            "im_id": im_id,
            "bio": im.attrs['bioregion_class_code'],
            "eunis": im.attrs['EUNIS_L2'],
        }

        if self.label_type == 'crop':
            label_key = im.attrs['lc1']
            result['crop_label'] = self.classes[self.label_map[label_key]]
            result['nuts'] = im.attrs['nuts0']

        elif self.label_type == 'all':
            lc_key = im.attrs['lc1']
            lu_key = im.attrs['lu1']
            result['lc'] = self.lc_classes[self.lc_label_map[lc_key]]
            result['lu'] = self.lu_classes[self.lu_label_map[lu_key]]
            result['nuts'] = nuts_labels[im.attrs['nuts0']]

        elif self.label_type == 'lc':
            lc_key = im.attrs['lc1']
            result['lc'] = self.lc_classes[self.lc_label_map[lc_key]]
            result['nuts'] = nuts_labels[im.attrs['nuts0']]

        elif self.label_type == 'lu':
            lu_key = im.attrs['lu1']
            result['lu'] = self.lu_classes[self.lu_label_map[lu_key]]
            result['nuts'] = nuts_labels[im.attrs['nuts0']]

        if self.return_coords:
            result["coordinates"] = im.attrs['Coordinates']

        return result




# test_dataset = Sen4MapDataset_withlabels(
#             h5data_path='./DATA/Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split_wise/1x1_crops/test_with_eunis.h5',
#             channels=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
#             annual_composite=True,
#             label_type='lc',
#             transform=True,
#             return_coords=True
#         )
# print(test_dataset[0])  # Example to check the first item in the test dataset
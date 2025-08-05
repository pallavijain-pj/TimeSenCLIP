import torch
import h5py
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split

from label_mapping import (
    labels, crop_labels, class_list_lc, lu_labels, class_list_lu,
    class_list_nuts, nuts_labels
)
from lucas_class_mapping import (
    LC1_class_list, LU1_class_list, LU1_class_map, LC1_class_map,
    bio_class_list, crop_class_map, crop_class_list, eunis_class_list
)


class BaseSen4MapDataset(Dataset):
    def __init__(self, h5data_path, channels, mean, std, annual_composite=True, transform=True):
        self.h5data_f = h5py.File(h5data_path, 'r')
        self.h5data = list(self.h5data_f.keys())
        self.channels = channels
        self.band_idx = [self.all_bands().index(c) for c in channels]
        self.mean = [mean[i] for i in self.band_idx]
        self.std = [std[i] for i in self.band_idx]
        self.annual_composite = annual_composite
        self.transform = transform

    def __len__(self):
        return len(self.h5data)

    def all_bands(self):
        return ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    def _transform(self, image):
        normalize = transforms.Normalize(self.mean, self.std)
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


class Sen4MapDataset_1x1(BaseSen4MapDataset):
    def __init__(self, h5data_path, lulc_type='lc1', channels=None, annual_composite=True, label_type='labels', transform=True, return_coords=False):
        channels = channels or ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        mean = [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27]
        std = [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45]
        super().__init__(h5data_path, channels, mean, std, annual_composite, transform)
        
        self.return_coords = return_coords
        self.lulc_type = lulc_type
        self.label_map = self.select_label_map(label_type, lulc_type)
        self.classes = self.select_classes(label_type, lulc_type)
        self.bio_classes = bio_class_list
        self.eunis_classes = eunis_class_list
        self.nuts_list = class_list_nuts

        with open("/home/pallavi/DATA/Datasets/Sen4Map/eunis_missing_ids.txt", "r") as f:
            missing_ids = f.read().splitlines()
        self.h5data = list(set(self.h5data) - set(missing_ids))

    def select_label_map(self, label_type, lulc_type):
        if label_type == "labels":
            return labels if lulc_type == 'lc1' else lu_labels
        elif label_type == "full_labels":
            return LC1_class_map if lulc_type == 'lc1' else LU1_class_map
        return crop_labels

    def select_classes(self, label_type, lulc_type):
        if label_type == "labels":
            return class_list_lc if lulc_type == 'lc1' else class_list_lu
        return LC1_class_list if lulc_type == 'lc1' else LU1_class_list

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

        label_key = im.attrs[self.lulc_type]
        label = self.classes[self.label_map[label_key]]
        nuts = nuts_labels[im.attrs['nuts0']]
        bioregion_class = im.attrs['bioregion_class_code']
        eunis_class = im.attrs['EUNIS_L2']
        coordinates = im.attrs['Coordinates']

        if self.return_coords:
            return image, label, im_id, nuts, bioregion_class, eunis_class, coordinates
        return image, label, im_id, nuts, bioregion_class, eunis_class


class Sen4MapDataset_1x1_cropslabels(BaseSen4MapDataset):
    def __init__(self, h5data_path, lulc_type='lc1', channels=None, annual_composite=True, transform=True, return_coords=False):
        channels = channels or ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        mean = [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27]
        std = [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45]
        super().__init__(h5data_path, channels, mean, std, annual_composite, transform)

        self.return_coords = return_coords
        self.lulc_type = lulc_type
        self.label_map = crop_class_map
        self.classes = crop_class_list
        self.bio_classes = bio_class_list
        self.eunis_classes = eunis_class_list

        # Filter crop keys
        crop_keys, nuts_list = self.crop_key_filter()
        self.nuts_list = list(nuts_list)
        self.h5data = crop_keys

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

        label_key = im.attrs[self.lulc_type]
        label = self.classes[self.label_map[label_key]]
        nuts = im.attrs['nuts0']
        bioregion_class = im.attrs['bioregion_class_code']
        eunis_class = im.attrs['EUNIS_L2']
        coordinates = im.attrs['Coordinates']

        if self.return_coords:
            return image, label, im_id, nuts, bioregion_class, eunis_class, coordinates
        return image, label, im_id, nuts, bioregion_class, eunis_class


def dataloader_sen4map(
    h5data_path,
    test_path=None,
    lulc_type='lc1',
    crop_size=1,
    channels=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
    annual_composite=True,
    train_size=0.7,
    label_type='labels',
    return_coords=False
):
    print("Initializing 1x1 crop dataset...")
    
    # Instantiate dataset
    dataset = Sen4MapDataset_1x1_witheunis(
        h5data_path=h5data_path,
        lulc_type=lulc_type,
        channels=channels,
        annual_composite=annual_composite,
        label_type=label_type,
        transform=True,
        return_coords=return_coords
    )

    class_list = dataset.classes
    dataset_size = len(dataset)
    train_len = int(train_size * dataset_size)

    if test_path is not None:
        # Use a separate test dataset
        val_len = dataset_size - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

        test_dataset = Sen4MapDataset_1x1(
            h5data_path=test_path,
            lulc_type=lulc_type,
            channels=channels,
            annual_composite=annual_composite,
            label_type=label_type,
            transform=True,
            return_coords=return_coords
        )
    else:
        # Split into train/val/test from a single dataset
        val_len = int((1.0 - train_size) / 2.0 * dataset_size)
        test_len = dataset_size - train_len - val_len
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    print(f"Train Dataset: {len(train_dataset)}, Val Dataset: {len(val_dataset)}, Test Dataset: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, class_list

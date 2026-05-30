import os
import time
import torch
from torch.utils.data import DataLoader, random_split
from .dataset_utils import load_embeddings
from .lucas_sen_data import CrossViewDataset
from .sen4map_data import Sen4MapDataset_withlabels, Sen4MapDataset_wImages_alllabels


def create_crossview_training_dataset(
    h5data_path,
    sen_path,
    emb_path,
    test_path=None,
    lulc_type="lc1",
    crop_size=1,
    channels=None,
    annual_composite=True,
    label_type="labels",
    resize=None,
    device="cpu",
    return_coords=False,
):
    """Creates a CrossView training (and optionally validation) dataset."""
    emb_dict = load_embeddings(emb_path)

    train_dataset = CrossViewDataset(
        h5data_path, sen_path, emb_dict,
        lulc_type, channels, annual_composite,
        label_type, device, return_coords,
    )

    if test_path:
        test_dataset = CrossViewDataset(
            test_path, sen_path, emb_dict,
            lulc_type, channels, annual_composite,
            label_type, device, return_coords,
        )
        return train_dataset, test_dataset, test_dataset.classes

    return train_dataset


def create_sen4map_inference_dataset(
    h5data_path,
    crop_size=1,
    channels=None,
    annual_composite=True,
    return_coords=False,
    metadata_csv=None,
):
    """Creates a Sen4Map inference dataset with all label types.

    Args:
        h5data_path: Path to the Sen4Map HDF5 test file.
        metadata_csv: Optional path to the labels CSV file.  If None, the
            bundled ``src/Data/labels_csv/Sen4Map_Test_all_labels.csv`` is used.
    """
    if channels is None:
        channels = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    if metadata_csv is None:
        metadata_csv = os.path.join(
            os.path.dirname(__file__),
            "labels_csv",
            "Sen4Map_Test_all_labels.csv",
        )

    print(f"Initializing dataset from {h5data_path}")
    print(f"Using metadata CSV: {metadata_csv}")

    dataset = Sen4MapDataset_wImages_alllabels(
        h5data_path=h5data_path,
        metadata_csv=metadata_csv,
        channels=channels,
        annual_composite=annual_composite,
        return_coords=return_coords,
    )
    return dataset, dataset.classes


def collate_fn(batch):
    """Custom collate to handle mixed string + tensor batches."""
    out = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        out[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return out


def load_data(args, state="train"):
    """Main data loader interface. Handles CrossView (train) and Sen4Map (test)."""
    print("Loading data...")
    start_time = time.time()

    if state == "train":
        train_dataset, val_dataset, classes = create_crossview_training_dataset(
            h5data_path=args.h5data_train_path,
            sen_path=args.sen_path,
            emb_path=args.emb_path,
            test_path=args.h5data_val_path,
            crop_size=args.crop_size,
            channels=args.channels,
            annual_composite=args.time_frames > 1,
            device=args.device,
        )
        _loader_kwargs = dict(
            batch_size=args.BATCH_SIZE,
            num_workers=args.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
            drop_last=True,
        )
        train_loader = DataLoader(train_dataset, shuffle=True,  **_loader_kwargs)
        val_loader   = DataLoader(val_dataset,   shuffle=False, **_loader_kwargs)

        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        print(f"Data loaded in {(time.time() - start_time) / 60:.2f} min")
        return train_loader, val_loader, classes

    else:
        # Inference — metadata_csv resolved inside create_sen4map_inference_dataset
        test_dataset, classes = create_sen4map_inference_dataset(
            h5data_path=args.dataset_path,
            crop_size=args.crop_size,
            annual_composite=args.time_frames > 1,
            channels=args.channels,
            return_coords=args.return_coords,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=False,
            num_workers=args.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        print(f"Test: {len(test_dataset)}")
        print(f"Data loaded in {(time.time() - start_time) / 60:.2f} min")
        return test_loader, classes

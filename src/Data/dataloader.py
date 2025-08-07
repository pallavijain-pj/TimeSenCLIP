import time
from torch.utils.data import DataLoader, random_split
from .dataset_utils import load_embeddings
from .lucas_sen_data import CrossViewDataset
from .sen4map_data import Sen4MapDataset_withlabels


def create_crossview_training_dataset(
    h5data_path,
    sen_path,
    emb_path,
    test_path=None,
    lulc_type='lc1',
    crop_size=1,
    channels=None,
    annual_composite=True,
    label_type='labels',
    resize=None,
    device='cpu',
    return_coords=False
):
    """
    Creates a CrossView training (and optionally validation) dataset.
    """
    emb_dict = load_embeddings(emb_path)

    train_dataset = CrossViewDataset(
        h5data_path, sen_path, emb_dict,
        lulc_type, channels, annual_composite,
        label_type, device, return_coords
    )

    if test_path:
        test_dataset = CrossViewDataset(
            test_path, sen_path, emb_dict,
            lulc_type, channels, annual_composite,
            label_type, device, return_coords
        )
        return train_dataset, test_dataset, test_dataset.classes

    return train_dataset


def create_sen4map_inference_dataset(
    h5data_path,
    test_path=None,
    crop_size=1,
    channels=None,
    annual_composite=True,
    train_size=0.7,
    label_type='all',
    return_coords=False
):
    """
    Creates train/val/test splits for the Sen4Map dataset.
    """
    if channels is None:
        channels = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    print("Initializing 1x1 crop dataset...")

    dataset = Sen4MapDataset_withlabels(
        h5data_path=h5data_path,
        channels=channels,
        annual_composite=annual_composite,
        label_type=label_type,
        transform=True,
        return_coords=return_coords
    )

    class_dict = dataset.classes
    dataset_size = len(dataset)
    train_len = int(train_size * dataset_size)

    if test_path:
        val_len = dataset_size - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

        test_dataset = Sen4MapDataset_withlabels(
            h5data_path=test_path,
            channels=channels,
            annual_composite=annual_composite,
            label_type=label_type,
            transform=True,
            return_coords=return_coords
        )
    else:
        val_len = int((1.0 - train_size) / 2.0 * dataset_size)
        test_len = dataset_size - train_len - val_len
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_len, val_len, test_len]
        )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, class_dict


def load_data(args, state='train'):
    """
    Main data loader interface. Handles both CrossView and Sen4Map datasets.
    """
    print("Script Launched....\nLoading Data....")
    start_time = time.time()

    if state == 'train':
        train_dataset, val_dataset, classes = create_crossview_training_dataset(
            h5data_path=args.h5data_train_path,
            sen_path=args.sen_path,
            emb_path=args.emb_path,
            test_path=args.h5data_val_path,
            crop_size=args.crop_size,
            channels=args.channels,
            annual_composite=args.time_frames > 1,
            device=args.device
        )
    else:
        train_dataset, val_dataset, test_dataset, classes = create_sen4map_inference_dataset(
            h5data_path=args.dataset_path,
            test_path=args.h5data_testpath,
            crop_size=args.crop_size,
            annual_composite=args.time_frames > 1,
            channels=args.channels,
            label_type=args.label_type,
            return_coords=args.return_coords,
            train_size=args.train_size
        )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=args.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=args.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    print(f"Data Loaded in {(time.time() - start_time)/60:.2f} minutes.")

    return train_loader, val_loader, classes

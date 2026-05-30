"""
src/Data/sen4map_data.py

Sen4Map Sentinel-2 dataset classes for TimeSenCLIP.

Two public dataset classes:
  - Sen4MapDataset_withlabels         — reads labels directly from HDF5 attributes
  - Sen4MapDataset_wImages_alllabels  — reads labels from a companion CSV file
                                        (supports all label taxonomies)

HDF5 files are opened lazily per worker so the datasets are fully compatible
with torch DataLoader multiprocessing (num_workers > 0).
"""

from __future__ import annotations

import torch
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Optional

from .lucas_label_mapping import (
    labels, crop_labels, class_list_lc, lu_labels, class_list_lu,
)
from .lucas_class_mapping import (
    LC1_class_list, LU1_class_list, LU1_class_map, LC1_class_map,
    bio_class_list, crop_class_map, crop_class_list,
    eunis_class_list, nuts_labels, class_list_nuts,
)
from .dataset_utils import SEN4MAP_SENTINEL_MEAN, SEN4MAP_SENTINEL_STD, BAND_NAMES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _band_indices(channels: List[str]) -> List[int]:
    """Return the integer indices into BAND_NAMES for the requested channels."""
    try:
        return [BAND_NAMES.index(c) for c in channels]
    except ValueError as e:
        raise ValueError(
            f"Unknown channel in {channels}. Valid channels: {BAND_NAMES}"
        ) from e


def _normalize(image: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Normalise a (T, C, H, W) float tensor channel-wise.

    Uses torchvision Normalize which expects input in (C, H, W) or (*, C, H, W).
    We iterate over time steps so the per-channel stats align with C.
    """
    normalizer = transforms.Normalize(mean=mean, std=std)
    # image: (T, C, H, W) — apply normalizer to each time step
    return torch.stack([normalizer(image[t]) for t in range(image.shape[0])], dim=0)
    


# ── Base class ────────────────────────────────────────────────────────────────

class _BaseSen4MapDataset(Dataset):
    """Common functionality shared by both Sen4Map dataset classes.

    HDF5 files are opened lazily inside ``__getitem__`` so each DataLoader
    worker gets its own file handle — required for ``num_workers > 0``.
    """

    DEFAULT_CHANNELS: List[str] = BAND_NAMES  # all 10 Sentinel-2 bands

    def __init__(
        self,
        h5data_path: str,
        channels: Optional[List[str]] = None,
        annual_composite: bool = True,
        transform: bool = True,
        return_coords: bool = False,
    ) -> None:
        self.h5data_path = h5data_path
        self.channels = channels or self.DEFAULT_CHANNELS
        self.annual_composite = annual_composite
        self.transform = transform
        self.return_coords = return_coords

        self.band_idx = _band_indices(self.channels)
        self.mean = [SEN4MAP_SENTINEL_MEAN[i] for i in self.band_idx]
        self.std  = [SEN4MAP_SENTINEL_STD[i]  for i in self.band_idx]

        # Eagerly open once just to read the key list, then close.
        with h5py.File(h5data_path, "r") as f:
            self._keys: List[str] = list(f.keys())

        # Lazy handle — opened per worker in __getitem__
        self._h5file: Optional[h5py.File] = None

    # ── HDF5 handle management ───────────────────────────────────────────────

    @property
    def h5file(self) -> h5py.File:
        """Return a per-process HDF5 handle, opening it lazily if needed."""
        if self._h5file is None:
            self._h5file = h5py.File(self.h5data_path, "r", libver="latest", swmr=True)
        return self._h5file

    def __del__(self) -> None:
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass

    # ── Shared tensor operations ─────────────────────────────────────────────

    def _select_channels(self, raw: np.ndarray) -> torch.Tensor:
        """Convert raw HDF5 array (T, C, H, W) → float tensor with selected bands."""
        t = torch.from_numpy(raw).float()          # (T, C, H, W)
        return t[:, self.band_idx, :, :]           # (T, len(channels), H, W)

    def _apply_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Normalise (T, C, H, W) tensor channel-wise."""
        return _normalize(image, self.mean, self.std)

    def _temporal_composite(self, image: torch.Tensor) -> torch.Tensor:
        """Reduce the time dimension to a single median frame → (1, C, H, W)."""
        return torch.median(image, dim=0, keepdim=True).values

    # ── Required Dataset API ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int):
        raise NotImplementedError


# ── Dataset 1: labels from HDF5 attributes ───────────────────────────────────

class Sen4MapDataset_withlabels(_BaseSen4MapDataset):
    """Sen4Map dataset that reads labels directly from HDF5 group attributes.

    Args:
        h5data_path:      Path to the Sen4Map HDF5 file.
        channels:         Sentinel-2 band names to load. Defaults to all 10.
        annual_composite: If False, reduce 12 time steps to a single median frame.
        transform:        Apply per-channel normalisation.
        return_coords:    Include geographic coordinates in each sample dict.
        label_type:       One of ``'all'``, ``'lc'``, ``'lu'``, ``'crop'``.
    """

    def __init__(
        self,
        h5data_path: str,
        channels: Optional[List[str]] = None,
        annual_composite: bool = True,
        transform: bool = True,
        return_coords: bool = False,
        label_type: str = "all",
    ) -> None:
        super().__init__(h5data_path, channels, annual_composite, transform, return_coords)

        self.label_type = label_type.lower()
        self.classes: dict = {
            "bio":   bio_class_list,
            "eunis": eunis_class_list,
            "nuts":  class_list_nuts,
        }

        if self.label_type == "crop":
            self.label_map = crop_class_map
            self.crop_classes = crop_class_list
            self.classes["crop"] = crop_class_list
            self._keys, nuts_set = self._filter_crop_keys()

        elif self.label_type in {"all", "lc", "lu"}:
            self.lc_label_map = labels
            self.lu_label_map = lu_labels
            self.lc_classes = class_list_lc
            self.lu_classes = class_list_lu
            self.classes["lc"] = class_list_lc
            self.classes["lu"] = class_list_lu

        else:
            raise ValueError(
                f"Unknown label_type '{label_type}'. "
                "Choose from: 'all', 'lc', 'lu', 'crop'."
            )

    def _filter_crop_keys(self):
        """Keep only HDF5 keys whose lc1_label belongs to a crop class."""
        valid_keys = []
        nuts_set: set = set()
        class_counts: dict = defaultdict(int)

        with h5py.File(self.h5data_path, "r") as f:
            for im_id in self._keys:
                attrs = f[im_id].attrs
                lc1 = attrs.get("lc1_label", "")
                nuts_set.add(attrs.get("nuts0", "?"))
                if lc1 in self.crop_classes:
                    valid_keys.append(im_id)
                    class_counts[lc1] += 1

        print(f"[crop filter] kept {len(valid_keys)}/{len(self._keys)} samples")
        print(f"[crop filter] NUTS regions: {sorted(nuts_set)}")
        print(f"[crop filter] class counts: {dict(class_counts)}")
        return valid_keys, nuts_set

    def __getitem__(self, index: int) -> dict:
        im_id = self._keys[index]
        grp = self.h5file[im_id]

        image = self._select_channels(grp["image"][:])
        if self.transform:
            image = self._apply_transform(image)
        if not self.annual_composite:
            image = self._temporal_composite(image)

        sample: dict = {
            "image":     image,
            "im_id":     im_id,
            "bioregion": grp.attrs.get("bioregion_class_code", -1),
            "eunis":     grp.attrs.get("EUNIS_L2", ""),
        }

        if self.label_type == "crop":
            lc1 = grp.attrs["lc1"]
            sample["crop"] = self.crop_classes[self.label_map[lc1]]
            sample["nuts"] = grp.attrs.get("nuts0", "")

        elif self.label_type == "all":
            sample["lc"]   = self.lc_classes[self.lc_label_map[grp.attrs["lc1"]]]
            sample["lu"]   = self.lu_classes[self.lu_label_map[grp.attrs["lu1"]]]
            sample["nuts"] = nuts_labels.get(grp.attrs.get("nuts0", ""), "")

        elif self.label_type == "lc":
            sample["lc"]   = self.lc_classes[self.lc_label_map[grp.attrs["lc1"]]]
            sample["nuts"] = nuts_labels.get(grp.attrs.get("nuts0", ""), "")

        elif self.label_type == "lu":
            sample["lu"]   = self.lu_classes[self.lu_label_map[grp.attrs["lu1"]]]
            sample["nuts"] = nuts_labels.get(grp.attrs.get("nuts0", ""), "")

        if self.return_coords:
            sample["coordinates"] = grp.attrs.get("Coordinates", None)

        return sample


# ── Dataset 2: labels from companion CSV ─────────────────────────────────────

class Sen4MapDataset_wImages_alllabels(_BaseSen4MapDataset):
    """Sen4Map dataset that reads labels from a companion CSV metadata file.

    This variant supports all label taxonomies simultaneously (LC, LU, crop,
    EUNIS, bioregion, NUTS).  The CSV must contain an ``im_id`` column whose
    values match HDF5 group keys, plus per-taxonomy integer label columns.

    Args:
        h5data_path:      Path to the Sen4Map HDF5 file.
        metadata_csv:     Path to the labels CSV file.
        channels:         Sentinel-2 band names to load. Defaults to all 10.
        annual_composite: If False, reduce 12 time steps to a single median frame.
        return_coords:    Include geographic coordinates if ``lat``/``lon``
                          columns are present in the CSV.
        transform:        Apply per-channel normalisation.
    """

    # CSV column name → (sample dict key, class list used to build str→int map)
    # crop_label uses -1 as the sentinel for non-crop samples ('NA' in the CSV).
    _LABEL_SPEC: List[tuple] = [
        ("lc_label",        "lc_label",        "lc"),
        ("lu_label",        "lu_label",        "lu"),
        ("crop_label",      "crop_label",      "crop"),
        ("nuts_label",      "nuts_label",      "nuts"),
        ("bioregion_class", "bioregion_label", "bioregion"),
        ("eunis_class",     "eunis_label",     "eunis"),
    ]

    def __init__(
        self,
        h5data_path: str,
        metadata_csv: str,
        channels: Optional[List[str]] = None,
        annual_composite: bool = True,
        return_coords: bool = False,
        transform: bool = True,
    ) -> None:
        super().__init__(h5data_path, channels, annual_composite, transform, return_coords)

        # ── Class lists ───────────────────────────────────────────────────────
        self.classes = {
            "lc":        class_list_lc,
            "lu":        class_list_lu,
            "crop":      crop_class_list,
            "bioregion": bio_class_list,
            "eunis":     eunis_class_list,
            "nuts":      class_list_nuts,
        }

        # Build str→int lookup tables from the canonical class lists.
        # crop_label is NaN (pandas null) for non-crop samples → -1.
        self._label_encoders: dict = {
            key: {name: idx for idx, name in enumerate(cls_list)}
            for key, cls_list in self.classes.items()
        }

        # ── Load and validate metadata CSV ───────────────────────────────────
        self.meta = pd.read_csv(metadata_csv, dtype={"im_id": str})

        csv_cols = {spec[0] for spec in self._LABEL_SPEC}
        missing = [c for c in csv_cols if c not in self.meta.columns]
        if missing:
            raise ValueError(
                f"metadata_csv is missing expected columns: {missing}\n"
                f"Available columns: {list(self.meta.columns)}"
            )

        # Keep only rows whose im_id exists in the HDF5 file
        h5_key_set = set(self._keys)
        before = len(self.meta)
        self.meta = self.meta[self.meta["im_id"].isin(h5_key_set)].reset_index(drop=True)
        dropped = before - len(self.meta)
        if dropped:
            print(f"[Sen4MapDataset_wImages_alllabels] "
                  f"dropped {dropped} CSV rows whose im_id was not found in the HDF5 file.")

        # Pre-encode all string labels to integers once at init so __getitem__
        # does zero string lookups at runtime.
        for csv_col, sample_key, cls_key in self._LABEL_SPEC:
            encoder = self._label_encoders[cls_key]
            col = self.meta[csv_col]

            # crop_label is NaN for non-crop rows — fill with -1 before mapping
            if cls_key == "crop":
                self.meta[sample_key] = col.map(encoder).fillna(-1).astype(int)
                continue

            encoded = col.map(encoder)
            unknown_mask = encoded.isna() & col.notna()
            if unknown_mask.any():
                unknown_vals = set(col[unknown_mask].unique())
                raise ValueError(
                    f"Column '{csv_col}' contains values not in the '{cls_key}' class list: "
                    f"{unknown_vals}"
                )
            if encoded.isna().any():
                raise ValueError(
                    f"Column '{csv_col}' has {encoded.isna().sum()} unexpected NaN values."
                )
            self.meta[sample_key] = encoded.astype(int)

        # Cast all label columns to int once, not per sample
        for _, sample_key, _ in self._LABEL_SPEC:
            self.meta[sample_key] = self.meta[sample_key].astype(int)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict:
        row = self.meta.iloc[index]
        im_id = row["im_id"]

        image = self._select_channels(self.h5file[im_id]["image"][:])
        if self.transform:
            image = self._apply_transform(image)
        if not self.annual_composite:
            image = self._temporal_composite(image)

        sample: dict = {
            "image": image,
            "im_id": im_id,
        }

        # Labels are already integer-encoded in self.meta — plain dict lookup
        for _, sample_key, _ in self._LABEL_SPEC:
            sample[sample_key] = int(row[sample_key])

        if self.return_coords and {"lat", "lon"}.issubset(row.index):
            sample["coords"] = torch.tensor([row["lat"], row["lon"]], dtype=torch.float32)

        return sample

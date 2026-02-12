
"""
Graph dataset loader for PyTorch-Geometric.
Expects files saved as: ../GraphDataset/<ISOFORM>/<split>/*.pt
Each .pt file should contain a torch_geometric.data.Data object (saved by torch.save).
"""

import os
from glob import glob
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader as GeometricDataLoader

# -----------------------------------------------------------------------------
# Helper Dataset that loads .pt graph files on demand (or optionally preloads)
# -----------------------------------------------------------------------------
class GraphFolderDataset(Dataset):
    """
    Wraps a folder of .pt files (each a torch_geometric.data.Data) as a Dataset.

    Args:
        folder: path to folder containing .pt files
        preload: if True, load all Data objects into memory at construction
                 (faster iteration but more memory use). Default: False.
        transform: optional callable(Data) -> Data to apply on-the-fly.
    """

    def __init__(self, folder: str, preload: bool = False, transform=None):
        self.folder = folder
        self.transform = transform

        # Find all .pt files. Sort for deterministic ordering.
        self.files: List[str] = sorted(glob(os.path.join(folder, "*.pt")))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {folder}")

        self.preload = preload
        self._data_cache: Optional[List[Data]] = None

        if preload:
            self._data_cache = [torch.load(p) for p in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        # Load from cache if preloaded
        if self.preload:
            data = self._data_cache[idx]
        else:
            # Load file on demand
            path = self.files[idx]
            data = torch.load(path)

        # Optionally apply a transform (e.g., augment, normalization)
        if self.transform is not None:
            data = self.transform(data)

        return data

# -----------------------------------------------------------------------------
# Factory: create train/val/test PyG DataLoaders
# -----------------------------------------------------------------------------
def make_graph_dataloaders(
    isoform: str,
    root: str = os.path.join("..", "GraphDataset"),
    batch_size: int = 64,
    num_workers: int = 4,
    preload: bool = False,
    transform=None,
    pin_memory: bool = True,
    shuffle_train: bool = True
) -> Tuple[GeometricDataLoader, Optional[GeometricDataLoader], Optional[GeometricDataLoader]]:
    """
    Build DataLoaders for train/val/test for a given isoform.

    Directory layout expected:
      <root>/<ISOFORM>/train/*.pt
      <root>/<ISOFORM>/val/*.pt
      <root>/<ISOFORM>/test/*.pt

    Returns:
      (train_loader, val_loader_or_None, test_loader_or_None)
    """

    def _folder_path(split: str) -> str:
        return os.path.join(root, isoform, split)

    # Construct datasets (if folder exists)
    train_folder = _folder_path("train")
    val_folder   = _folder_path("val")
    test_folder  = _folder_path("test")

    if not os.path.isdir(train_folder):
        raise FileNotFoundError(f"Train folder not found: {train_folder}")

    train_ds = GraphFolderDataset(train_folder, preload=preload, transform=transform)
    val_ds   = None
    test_ds  = None

    if os.path.isdir(val_folder):
        # If val exists but empty .pt will raise inside GraphFolderDataset
        val_ds = GraphFolderDataset(val_folder, preload=preload, transform=transform)

    if os.path.isdir(test_folder):
        test_ds = GraphFolderDataset(test_folder, preload=preload, transform=transform)

    # Create PyG DataLoaders
    train_loader = GeometricDataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = None
    test_loader = None

    if val_ds is not None:
        val_loader = GeometricDataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=max(0, num_workers//2), pin_memory=pin_memory
        )

    if test_ds is not None:
        test_loader = GeometricDataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=max(0, num_workers//2), pin_memory=pin_memory
        )

    return train_loader, val_loader, test_loader

# -----------------------------------------------------------------------------
# Utility: quick sanity function to list counts
# -----------------------------------------------------------------------------
def count_graphs_in_split(isoform: str, root: str = os.path.join("..","GraphDataset")) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        folder = os.path.join(root, isoform, split)
        out[split] = len(sorted(glob(os.path.join(folder, "*.pt")))) if os.path.isdir(folder) else 0
    return out

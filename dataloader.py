"""
Script 2: dataloader.py
PyTorch Dataset & DataLoader for the KTH Human Action dataset.

Each video (AVI) is decoded into a fixed number of frames.
Frames are stacked → Tensor of shape (T, C, H, W) for the model.

Usage (standalone test):
    python dataloader.py
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.io as tvio


# ─────────────────────────── CONFIG ───────────────────────────────────────────
ACTIONS = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
LABEL2IDX = {a: i for i, a in enumerate(ACTIONS)}

DEFAULT_DATA_ROOT  = Path("data/kth_actions")
DEFAULT_NUM_FRAMES = 16        # frames sampled uniformly per video
DEFAULT_IMG_SIZE   = (112, 112)
# ──────────────────────────────────────────────────────────────────────────────


class KTHActionDataset(Dataset):
    """
    KTH Human Action Video Dataset.

    Folder structure expected (created by download_dataset.py):
        data/kth_actions/
            walking/       *.avi
            jogging/       *.avi
            running/       *.avi
            boxing/        *.avi
            handwaving/    *.avi
            handclapping/  *.avi

    Returns:
        frames  : FloatTensor (T, C, H, W)  values in [0, 1]
        label   : LongTensor scalar
    """

    def __init__(
        self,
        root: Path = DEFAULT_DATA_ROOT,
        num_frames: int = DEFAULT_NUM_FRAMES,
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        transform: Optional[T.Compose] = None,
        split: Optional[str] = None,          # "train" | "val" | "test" | None
        train_persons: List[int] = list(range(1, 17)),   # persons 1-16  → train+val
        test_persons:  List[int] = list(range(17, 26)),  # persons 17-25 → test
        seed: int = 42,
    ):
        """
        Args:
            root        : path to dataset root (created by download_dataset.py)
            num_frames  : number of frames to sample uniformly from each video
            img_size    : (H, W) to resize each frame to
            transform   : optional additional augmentation (applied per-frame)
            split       : if None, all videos are loaded.
                          "train"/"val" split persons 1-16 as 80/20;
                          "test" uses persons 17-25.
            train_persons / test_persons : person IDs for the respective splits
            seed        : random seed for reproducible train/val split
        """
        self.root       = Path(root)
        self.num_frames = num_frames
        self.img_size   = img_size
        self.transform  = transform
        self.split      = split

        # Base frame transforms: resize + normalize
        self.base_transform = T.Compose([
            T.Resize(img_size),
            T.ConvertImageDtype(torch.float32),   # uint8 → float [0,1]
        ])

        # Collect (path, label_idx) pairs
        self.samples: List[Tuple[Path, int]] = []
        self._collect_samples(train_persons, test_persons, seed)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _person_id_from_filename(self, fname: str) -> int:
        """Extract person ID from filenames like 'person01_boxing_d1_uncomp.avi'."""
        try:
            return int(fname.split("_")[0].replace("person", ""))
        except (IndexError, ValueError):
            return -1

    def _collect_samples(
        self,
        train_persons: List[int],
        test_persons: List[int],
        seed: int,
    ) -> None:
        """Walk the dataset folder and fill self.samples."""
        all_samples: List[Tuple[Path, int]] = []

        for action in ACTIONS:
            class_dir = self.root / action
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing class folder: {class_dir}\n"
                    f"Run download_dataset.py first."
                )
            for avi in sorted(class_dir.glob("*.avi")):
                all_samples.append((avi, LABEL2IDX[action]))

        if self.split is None:
            self.samples = all_samples
            return

        # Person-based split (mirrors the original KTH paper protocol)
        if self.split == "test":
            self.samples = [
                (p, l) for p, l in all_samples
                if self._person_id_from_filename(p.name) in test_persons
            ]
        else:
            train_val = [
                (p, l) for p, l in all_samples
                if self._person_id_from_filename(p.name) in train_persons
            ]
            # 80/20 split for train vs val
            rng = random.Random(seed)
            rng.shuffle(train_val)
            cut = int(0.8 * len(train_val))
            self.samples = train_val[:cut] if self.split == "train" else train_val[cut:]

    def _load_frames(self, video_path: Path) -> torch.Tensor:
        """
        Load a video and uniformly sample self.num_frames frames.
        Returns a Tensor of shape (T, C, H, W), dtype float32, values [0,1].
        """
        # tvio.read_video returns (video, audio, info)
        # video shape: (T, H, W, C) uint8
        try:
            video, _, _ = tvio.read_video(str(video_path), pts_unit="sec", output_format="TCHW")
        except Exception as e:
            raise RuntimeError(f"Failed to read {video_path}: {e}")

        total_frames = video.shape[0]

        if total_frames == 0:
            # Fallback: return zeros
            return torch.zeros(self.num_frames, 3, *self.img_size)

        # Uniform sampling
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        else:
            # Repeat last frame if video is shorter than num_frames
            indices = torch.arange(total_frames)
            pad = torch.full((self.num_frames - total_frames,), total_frames - 1)
            indices = torch.cat([indices, pad])

        frames = video[indices]  # (T, C, H, W) uint8

        # Resize & convert to float
        processed = []
        for frame in frames:
            frame = self.base_transform(frame)   # (C, H, W) float32
            if self.transform is not None:
                frame = self.transform(frame)
            processed.append(frame)

        return torch.stack(processed)  # (T, C, H, W)

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)


# ─────────────────────── DataLoader Factory ───────────────────────────────────

def get_dataloaders(
    data_root: Path = DEFAULT_DATA_ROOT,
    num_frames: int = DEFAULT_NUM_FRAMES,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Augmentation is applied only to the training split.
    """
    train_aug = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    ])

    train_ds = KTHActionDataset(
        root=data_root, num_frames=num_frames, img_size=img_size,
        transform=train_aug, split="train"
    )
    val_ds = KTHActionDataset(
        root=data_root, num_frames=num_frames, img_size=img_size,
        transform=None, split="val"
    )
    test_ds = KTHActionDataset(
        root=data_root, num_frames=num_frames, img_size=img_size,
        transform=None, split="test"
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ─────────────────────── Standalone Test ──────────────────────────────────────

if __name__ == "__main__":
    print("Testing KTHActionDataset...")
    ds = KTHActionDataset(split=None)
    print(f"Total videos found: {len(ds)}")

    frames, label = ds[0]
    print(f"Sample → frames: {frames.shape}, label: {label.item()} ({ACTIONS[label.item()]})")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    print(f"\nSplit sizes → train: {len(train_loader.dataset)}, "
          f"val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

    batch_frames, batch_labels = next(iter(train_loader))
    print(f"Batch → frames: {batch_frames.shape}, labels: {batch_labels}")
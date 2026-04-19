"""
BiosignalDataset
----------------
Loads pre-sliced multimodal biosignal windows (EEG, GSR, PPG)
from a directory of .npy files and a split index file.

Expected directory structure:
    DATA_SLICED/
        sample_00000_eeg.npy   # shape: (EEG_CHANNELS, T)
        sample_00000_gsr.npy   # shape: (T,)
        sample_00000_ppg.npy   # shape: (T,)
        sample_00000_label.npy # scalar: 0 or 1
        ...

    data_splits/
        train_idx.npy
        val_idx.npy
        test_idx.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BiosignalDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train',
                 split_path: str = None):
        """
        Args:
            data_dir   : path to DATA_SLICED folder
            split      : 'train' | 'val' | 'test'
            split_path : path to folder containing {split}_idx.npy
        """
        assert split in ('train', 'val', 'test'), \
            "split must be 'train', 'val', or 'test'"

        self.data_dir = data_dir

        # load split indices
        if split_path is not None:
            idx_file = os.path.join(split_path, f"{split}_idx.npy")
            self.indices = np.load(idx_file).tolist()
        else:
            # fallback: use all samples in directory
            all_labels = sorted([
                f for f in os.listdir(data_dir) if f.endswith("_label.npy")
            ])
            self.indices = list(range(len(all_labels)))

        # load all data into memory for faster iteration
        self.eeg_data, self.gsr_data, self.ppg_data, self.labels = \
            self._load_all()
        print(f"[INFO] {split}: {len(self.labels)} samples loaded into memory ✅")

    def _load_all(self):
        eeg_list, gsr_list, ppg_list, label_list = [], [], [], []
        for idx in self.indices:
            prefix = os.path.join(self.data_dir, f"sample_{idx:05d}")
            eeg_list.append(np.load(f"{prefix}_eeg.npy").astype(np.float32))
            gsr_list.append(np.load(f"{prefix}_gsr.npy").astype(np.float32))
            ppg_list.append(np.load(f"{prefix}_ppg.npy").astype(np.float32))
            label_list.append(float(np.load(f"{prefix}_label.npy")))
        return eeg_list, gsr_list, ppg_list, label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        eeg = torch.tensor(self.eeg_data[i]).unsqueeze(0)  # (1, C, T)
        gsr = torch.tensor(self.gsr_data[i]).unsqueeze(0)  # (1, T)
        ppg = torch.tensor(self.ppg_data[i]).unsqueeze(0)  # (1, T)
        label = torch.tensor(self.labels[i], dtype=torch.float32)
        return eeg, gsr, ppg, label

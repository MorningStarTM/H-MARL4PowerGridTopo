import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from HMARL.Utils.logger import logger



class RegionalDataset(Dataset):
    def __init__(self, npy_file):
        data = np.load(npy_file)
        self.actions = data[:, 0].astype(np.int64)           # Labels
        self.states = data[:, 1:].astype(np.float32)         # Features

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    

    
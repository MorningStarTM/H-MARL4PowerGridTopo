import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from HMARL.Utils.logger import logger
from HMARL.Utils.custom_dataset import RegionalDataset
from HMARL.Agents.neural_network import RegionNetwork

from collections import Counter

class RegionalTrainer:
    def __init__(self, model: RegionNetwork, data_dir, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.data_dir = data_dir
        self.action_dim = config["action_dim"]

        # Load multiple .npy files and combine them
        npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        datasets = [RegionalDataset(f) for f in npy_files]
        full_dataset = torch.utils.data.ConcatDataset(datasets)
        self.dataloader = DataLoader(full_dataset, batch_size=config["batch_size"], shuffle=True)

        # Calculate class weights
        all_labels = []
        for f in npy_files:
            data = np.load(f)
            all_labels.extend(data[:, 0].astype(int))

        label_counts = Counter(all_labels)
        total = sum(label_counts.values())
        weights = []

        for i in range(self.action_dim):
            count = label_counts.get(i, 0)
            if count == 0:
                weights.append(1e6)  # Penalize missing classes
            else:
                weights.append(total / count)

        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])

    def train(self, num_epochs=10):
        self.model.train()
        history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for states, actions in self.dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(states)
                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * actions.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()

            avg_loss = epoch_loss / total
            accuracy = 100.0 * correct / total
            history.append((avg_loss, accuracy))
            logger.info(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return history

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from HMARL.Utils.logger import logger
from HMARL.Utils.custom_dataset import RegionalDataset
from HMARL.Agents.neural_network import RegionNetwork

class RegionalTrainer:
    def __init__(self, model:RegionNetwork, data_dir, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load multiple .npy files and combine them
        npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        datasets = [RegionalDataset(f) for f in npy_files]
        full_dataset = torch.utils.data.ConcatDataset(datasets)
        self.dataloader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

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
            accuracy = 100. * correct / total
            history.append((avg_loss, accuracy))
            logger.info(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return history

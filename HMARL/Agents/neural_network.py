import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from HMARL.Utils.logger import logger



class LinearResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)
    

class LinearResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.resblocks = nn.ModuleList([
            LinearResBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=-1)
        )



    def forward(self, x):
        x = self.input_layer(x)
        for block in self.resblocks:
            x = block(x)
        policy = self.policy_head(x)
        return policy

    def get_param_size_million(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params / 1e6


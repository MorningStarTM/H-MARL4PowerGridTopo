import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from HMARL.Utils.logger import logger



class RegionNetwork(nn.Module):
    def __init__(self, config):
        super(RegionNetwork, self).__init__()
        self.config = config
        self.name = self.config['name']

        self.network = nn.Sequential(
                    nn.Linear(self.config['state_dim'], 512),
                    nn.Tanh(),
                    nn.Linear(512, 256),
                    nn.Tanh(),
                    nn.Linear(256, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, self.config['action_dim']),
                    nn.Softmax(dim=-1)
                )
        
    def forward(self, x):
        x = self.network(x)
        return x
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, f"region_network_{self.config['name']}.pth"))
        logger.info(f"Region Network saved at {path}")

    def load(self, path):
        if os.path.exists(os.path.join(path, f"region_network_{self.config['name']}.pth")):
            self.load_state_dict(torch.load(os.path.join(path, f"region_network_{self.config['name']}.pth")))
            logger.info(f"Region Network loaded from {path}")
        else:
            logger.error(f"Region Network model not found at {path}")


    

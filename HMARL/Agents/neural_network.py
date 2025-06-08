import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from HMARL.Utils.logger import logger
from torch_geometric.nn import GCNConv, GATConv


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





class GCN(nn.Module):
    def __init__(self, input_dim, num_nodes=57, output_dim=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.flattened_dim = num_nodes * input_dim * 2
        self.conv1 = GCNConv(input_dim, input_dim*2)
        self.conv2 = GCNConv(input_dim*2, input_dim*2)
        self.lin1 = nn.Linear(self.flattened_dim, self.flattened_dim//2)
        self.lin3 = nn.Linear(self.flattened_dim//2, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = x.view(-1)  
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.softmax(x)
        return x




class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        self.lin1 = nn.Linear(output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x




####################################################################################################################################################################
class ResGCNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        identity = x
        out = self.conv1(x, edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)
        out += identity  # residual connection
        out = self.relu(out)
        return out


class ResGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_blocks=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_proj = GCNConv(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResGCNBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.flattened_dim = num_nodes * hidden_dim
        self.lin1 = nn.Linear(self.flattened_dim, self.flattened_dim//4)
        self.lin2 = nn.Linear(self.flattened_dim//4, self.flattened_dim//8)
        self.lin3 = nn.Linear(self.flattened_dim//8, output_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        x = self.input_proj(x, edge_index)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x, edge_index)
        x = x.view(-1)  # flatten all node features
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.softmax(x)
        return x
    

    


#GCN
#GAT
#GraphSAGE
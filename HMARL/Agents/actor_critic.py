import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from HMARL.Agents.neural_network import LinearResNet, GCN, GAT, ResGCN


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        
        if self.config['network'] == 'resnet':
            self.action_layer = LinearResNet(input_dim=self.config['state_dim'], hidden_dim=128, output_dim=self.config['action_dim'], num_blocks=6)
            self.value_layer = LinearResNet(input_dim=self.config['state_dim'], hidden_dim=128, output_dim=1, num_blocks=6)
        elif self.config['network'] == 'gcn':
            self.action_layer = GCN(input_dim=self.config['state_dim'], output_dim=self.config['action_dim'])
            self.value_layer = GCN(input_dim=self.config['state_dim'], output_dim=1)
        elif self.config['network'] == 'gat':
            self.action_layer = GAT(input_dim=self.config['state_dim'], hidden_dim=128, output_dim=self.config['action_dim'], heads=6)
            self.value_layer = GAT(input_dim=self.config['state_dim'], hidden_dim=128, output_dim=1, heads=3)
        elif self.config['network'] == 'resgcn':
            self.action_layer = ResGCN(input_dim=self.config['state_dim'], hidden_dim=128, output_dim=self.config['action_dim'], num_nodes=57, num_blocks=6)
            self.value_layer = ResGCN(input_dim=self.config['state_dim'], hidden_dim=128, output_dim=1, num_nodes=57, num_blocks=6)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        
        state_value = self.value_layer(state)
        
        action_probs = self.action_layer(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
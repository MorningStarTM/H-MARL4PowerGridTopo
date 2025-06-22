import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from HMARL.Utils.action_converter import ActionConverter, MADiscActionConverter
from HMARL.Agents.neural_network import LinearResNet, GCN, GAT, ResGCN
from HMARL.Utils.logger import logger


class ActorCritic(nn.Module):
    def __init__(self, state, sublist, env, config):
        super(ActorCritic, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = MADiscActionConverter(env, sublist)
        action_dim = self.ac.action_size()
        

        if self.config['network'] == 'resnet':
            self.action_layer = LinearResNet(input_dim=self.config['input_dim'], hidden_dim=128, output_dim=action_dim, num_blocks=6)
            self.value_layer = LinearResNet(input_dim=self.config['input_dim'], hidden_dim=128, output_dim=1, num_blocks=6)
        elif self.config['network'] == 'gcn':
            self.action_layer = GCN(input_dim=self.config['input_dim'], output_dim=action_dim)
            self.value_layer = GCN(input_dim=self.config['input_dim'], output_dim=1)
        elif self.config['network'] == 'gat':
            self.action_layer = GAT(input_dim=self.config['input_dim'], hidden_dim=128, output_dim=action_dim, heads=6)
            self.value_layer = GAT(input_dim=self.config['input_dim'], hidden_dim=128, output_dim=1, heads=3)
        elif self.config['network'] == 'resgcn':
            self.action_layer = ResGCN(input_dim=self.config['input_dim'], hidden_dim=128, output_dim=action_dim, num_nodes=57, num_blocks=6)
            self.value_layer = ResGCN(input_dim=self.config['input_dim'], hidden_dim=128, output_dim=1, num_nodes=57, num_blocks=6)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.step_counter = 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        self.to(self.device)

    def _select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        
        state_value = self.value_layer(state)
        
        action_probs = self.action_layer(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        action = action.item()
        grid_action = self.ac.act(action)
        return action, grid_action
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        state_value = self.value_layer(state)
        action_probs = self.action_layer(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        logprob = action_distribution.log_prob(action)

        action = action.item()
        grid_action = self.ac.act(action)
        
        return action, grid_action, logprob, state_value

    
    def calculateLoss(self, gamma=0.99):
        if not (self.logprobs and self.state_values and self.rewards):
            logger.error("Warning: Empty memory buffers!")
            return torch.tensor(0.0, device=self.device)
        

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
       
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward.unsqueeze(0))
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    
    def save(self, checkpoint, filename="actor_critic.pt"):
        os.makedirs(checkpoint, exist_ok=True)
        logger.info(f"model save folder created")
        model_path = os.path.join(checkpoint, filename)
        torch.save(self.state_dict(), model_path)
        print(f"[INFO] Model saved to {model_path}")

    def load(self, checkpoint, filename="actor_critic.pt"):
        model_path = os.path.join(checkpoint, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Model file not found: {model_path}")
        self.load_state_dict(torch.load(model_path))
        self.eval()
        print(f"[INFO] Model loaded from {model_path}")
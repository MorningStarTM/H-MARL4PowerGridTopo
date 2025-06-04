import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from HMARL.Utils.logger import logger
from HMARL.Utils.custom_dataset import RegionalDataset
from HMARL.Agents.neural_network import RegionNetwork
from HMARL.MultiAgents.imarl import IMARL
from collections import Counter
from grid2op import Environment



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





class MARLTrainer:
    def __init__(self, imarl:IMARL, env:Environment, config):
        self.imarl = imarl  # instance of IMARL
        self.env = env
        self.config = config

        self.log_dir = os.path.join("model_logs", config['env_name'])
        self.model_dir = os.path.join("Models", config['env_name'])
        self.reward_dir = os.path.join("rewards", config['env_name'])
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.reward_dir, exist_ok=True)

        self.episode_rewards = {cid: [] for cid in imarl.agents.keys()}
        self.step_rewards = {cid: [] for cid in imarl.agents.keys()}
        self.step_counter = {cid: 0 for cid in imarl.agents.keys()}

    def train(self, min_episode, max_episodes):
        logger.info("Starting MARL Training")
        start_time = datetime.now().replace(microsecond=0)

        for episode in range(min_episode, max_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = {cid: 0.0 for cid in self.imarl.agents.keys()}

            for step in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode}"):
                is_safe = self.imarl.is_safe(obs)
                action_idx, grid_action = self.imarl.agent_action(obs, is_safe, sample=True)
                obs_, reward, done, _ = self.env.step(grid_action)

                sub = self.imarl.sub_picker.prev_sub
                cid = self.imarl.sub_to_cluster[sub]

                self.imarl.agents[cid].buffer.rewards.append(reward)
                self.imarl.agents[cid].buffer.is_terminals.append(done)

                self.episode_rewards[cid].append(reward)
                self.step_rewards[cid].append(reward)
                self.step_counter[cid] += 1

                obs = obs_

                # Train when buffer is large enough
                if self.step_counter[cid] >= self.config['update_timestep']:
                    logger.info(f"Training agent {cid} at step {step}...")
                    self.imarl.agents[cid].update()
                    self.step_counter[cid] = 0

                if done:
                    break

            if (episode + 1) % self.config['save_model_freq'] == 0:
                self.imarl.save_model()

        self.save_rewards()
        logger.info("Training completed")
        logger.info(f"Total training time: {datetime.now().replace(microsecond=0) - start_time}")

    def save_rewards(self):
        for cid in self.imarl.agents:
            np.save(os.path.join(self.reward_dir, f"ppo_cluster{cid}_episode_rewards.npy"), np.array(self.episode_rewards[cid]))
            np.save(os.path.join(self.reward_dir, f"ppo_cluster{cid}_step_rewards.npy"), np.array(self.step_rewards[cid]))
        logger.info("Saved reward logs for all agents.")

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from HMARL.Utils.logger import logger
from HMARL.MultiAgents.imarl import IMARL
from collections import Counter
from grid2op import Environment
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from HMARL.Utils.custom_reward import LossReward, MarginReward
from grid2op.Exceptions import *
import random
import matplotlib.pyplot as plt
import math





class MARLTrainer:
    def __init__(self, imarl:IMARL, env, config):
        self.imarl = imarl  # instance of IMARL
        self.env = env

        self.config = config

        self.log_dir = os.path.join("model_logs", config['ENV_NAME'])
        self.model_dir = os.path.join("models", config['ENV_NAME'])
        self.model_dir = os.path.join(self.model_dir, config['agent_type'])
        self.reward_dir = os.path.join("rewards", config['ENV_NAME'])
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.reward_dir, exist_ok=True)

        self.episode_rewards = []
        self.step_rewards = {cid: [] for cid in imarl.agents.keys()}
        self.step_counter = {cid: 0 for cid in imarl.agents.keys()}

    

    def train(self, min_episode, max_episodes):
        logger.info("Starting MARL Training")
        start_time = datetime.now().replace(microsecond=0)
        episode_steps = []

        for episode in range(min_episode, max_episodes):
            total_steps = 0
            self.env.set_id(episode)
            logger.info(f"Episode ID : {episode} --- Episode name : {self.env.chronics_handler.get_name()}")


            obs = self.env.reset()
            done = False
            episode_reward = {cid: 0.0 for cid in self.imarl.agents.keys()}
            episode_reward_tot = 0


            for step in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode}"):
                try:
                    is_safe = self.imarl.is_safe(obs)

                    if not is_safe:
                        action_idx, grid_action, logprob, value, state_vec, cid = self.imarl.agent_action(obs, sample=True)
                    else:
                        total_steps += 1
                        grid_action = self.env.action_space()
                        action_idx, logprob, value, state_vec, cid = -1, None, None, None, None

                    obs_, reward, done, _ = self.env.step(grid_action)
                    episode_reward_tot += reward

                    if not is_safe:
                        
                        agent = self.imarl.agents[cid]
                        agent.buffer.states.append(torch.FloatTensor(state_vec).to(agent.device))
                        agent.buffer.actions.append(torch.tensor(action_idx).to(agent.device))
                        agent.buffer.logprobs.append(logprob)
                        agent.buffer.state_values.append(value)
                        agent.buffer.rewards.append(reward)
                        agent.buffer.is_terminals.append(done)

                    
                    obs = obs_

                    if done:
                        self.env.set_id(episode)
                        obs = self.env.reset()
                        done = False
                        reward = self.env.reward_range[0]
                        self.env.fast_forward_chronics(step - 1)

                        is_safe = self.imarl.is_safe(obs)
                        if not is_safe:
                            if self.config['agent_type'] == 'graph_ppo':
                                action_idx, grid_action, logprob, value, state_vec, cid, t_state_, t_adj = self.imarl.agent_action(obs, sample=True)
                            else:
                                action_idx, grid_action, logprob, value, state_vec, cid = self.imarl.agent_action(obs, sample=True)
                            obs_, reward, done, _ = self.env.step(grid_action)

                            
                            agent = self.imarl.agents[cid]
                            agent.buffer.states.append(torch.FloatTensor(state_vec).to(agent.device))
                            agent.buffer.actions.append(torch.tensor(action_idx).to(agent.device))
                            agent.buffer.logprobs.append(logprob)
                            agent.buffer.state_values.append(value)
                            agent.buffer.rewards.append(reward)
                            agent.buffer.is_terminals.append(done)


                    # Train when buffer is large enough
                    if all((agent.buffer) >= self.config['update_timestep'] for agent in self.imarl.agents.values()):
                        logger.info(f"All buffers full. Updating all agents at step {step}")
                        logger.info(f"""=============================================================================================================================================
                                                                                        Training Started
                             =============================================================================================================================================)       """)
                        for cid, agent in self.imarl.agents.items():
                            assert len(agent.buffer.rewards) == len(agent.buffer.actions), \
        f"[BUG] Buffer size mismatch: rewards={len(agent.buffer.rewards)}, states={len(agent.buffer.states)} for agent {cid}"

                            agent.update()
                        self.imarl.save_model()
                        logger.info(f"model saved at {self.model_dir}")


                except NoForecastAvailable as e:
                    logger.error(f"Grid2OpException encountered at step {step} in episode {episode}: {e}")
                    self.env.set_id(episode)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(step-1)
                    continue

                except Grid2OpException as e:
                    logger.error(f"Grid2OpException encountered at step {step} in episode {episode}: {e}")
                    self.env.set_id(episode)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(step-1)
                    continue 

            #if (episode + 1) % self.config['save_model_freq'] == 0:
            
        
            self.episode_rewards.append(episode_reward_tot)
            episode_steps.append(total_steps)

        self.save_rewards()
        self.plot_dn_pies(dn_steps_list=episode_steps, total_steps_per_episode=self.env.max_episode_duration(),show=False)
        logger.info("Training completed")
        logger.info(f"Total training time: {datetime.now().replace(microsecond=0) - start_time}")
        

    def save_rewards(self):
        np.save(os.path.join(self.reward_dir, f"ippo_cluster_episode_rewards.npy"), np.array(self.episode_rewards))
        for cid in self.imarl.agents:    
            np.save(os.path.join(self.reward_dir, f"ppo_cluster{cid}_step_rewards.npy"), np.array(self.step_rewards[cid]))
        logger.info("Saved reward logs for all agents.")

    
    def plot_dn_pies(self, dn_steps_list, total_steps_per_episode, filename="HMARL\\experiment\\dn_pie_grid.png", show=True):
        """
        Plots and saves a grid of pie charts for each episode, arranged in a nearly square grid.

        Args:
            dn_steps_list: List[int] - Do-nothing steps per episode.
            total_steps_per_episode: int - Total steps per episode.
            filename: str - Output filename.
            show: bool - If True, display the plot.

        Returns:
            fig: matplotlib.figure.Figure
        """
        n = len(dn_steps_list)
        grid_size = math.ceil(math.sqrt(n))
        rows = cols = grid_size
        if (rows-1) * cols >= n:
            rows -= 1

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if n > 1 else [axes]

        for i, dn in enumerate(dn_steps_list):
            other = total_steps_per_episode - dn
            sizes = [dn, other]
            labels = ['Do Nothing', 'Other']
            colors = ['#3CB371', '#FFA07A']
            axes[i].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, explode=(0.1, 0))
            axes[i].set_title(f'Episode {i+1}\n{dn}/{total_steps_per_episode}')
        
        # Hide any unused axes
        for j in range(len(dn_steps_list), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        fig.savefig(filename, dpi=150)
        print(f"Saved pie chart grid to {filename}")
        if show:
            plt.show()
        plt.close(fig)



    def actor_critic_train(self, min_episode, max_episodes):
        for episode_id in range(min_episode, max_episodes):
            total_steps = 0            
            self.env.set_id(episode_id)
            logger.info(f"Episode ID : {episode_id} --- Episode name : {self.env.chronics_handler.get_name()}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0


            for i in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode_id}", leave=True):
                
                try:
                    is_safe = self.imarl.is_safe(obs)

                    if not is_safe:
                        action, grid_action, logprob, state_value, cid = self.imarl.agent_action(obs, sample=True) 
                    else:
                        total_steps += 1
                        grid_action = self.env.action_space()
                        

                    obs_, reward, done, _ = self.env.step(grid_action)
                    episode_total_reward += reward
                    

                    if not is_safe:
                        agent = self.imarl.agents[cid]
                        agent.step_counter += 1
                        agent.rewards.append(reward)
                        agent.logprobs.append(logprob)
                        agent.state_values.append(state_value)
                    obs = obs_

                    if done:
                        self.env.set_id(episode_id)
                        
                        obs = self.env.reset()
                        done = False
                        reward = self.env.reward_range[0]

                        self.env.fast_forward_chronics(i - 1)
                        is_safe = self.imarl.is_safe(obs)

                        if not is_safe:
                            action, grid_action, logprob, state_value, cid = self.imarl.agent_action(obs, sample=True) 
                            obs_, reward, done, _ = self.env.step(grid_action)
                            agent = self.imarl.agents[cid]
                            agent.step_counter += 1
                            agent.rewards.append(reward)
                            agent.logprobs.append(logprob)
                            agent.state_values.append(state_value)

                    #if self.step_counter % self.update_freq == 0:
                    if all(agent.step_counter % self.config['update_freq'] == 0 and agent.step_counter > 0 for agent in self.imarl.agents.values()):
                        logger.info(f"""=============================================================================================================================================
                                                                                        Training Started
                             =============================================================================================================================================)       """)
                        
                        for cid, agent in self.imarl.agents.items():
                            assert len(agent.rewards) == len(agent.state_values), \
                            f"[BUG] Buffer size mismatch: rewards={len(agent.rewards)}, states={len(agent.state_values)} for agent {cid}"

                            agent.optimizer.zero_grad()
                            loss = agent.calculateLoss()
                            logger.info(f"Loss for agent {cid} at episode {episode_id}, step {i}: {loss.item()}")
                            loss.backward()
                            agent.optimizer.step()
                            agent.clearMemory()
                        self.imarl.save_model()
                        logger.info(f"model saved at {self.model_dir}")

                except NoForecastAvailable as e:
                    logger.error(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue

                except Grid2OpException as e:
                    logger.error(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue 



    def evaluate_hmarl(self, num_episodes=3, save_path="eval_results.png"):
        max_steps = self.env.max_episode_duration()
        episode_steps = []
        paths = self.env.chronics_handler.subpaths

        # Randomly select 3 episodes, avoid duplicates
        test_paths = random.sample(list(paths)[900:], num_episodes)

        for ep_idx, test_path in enumerate(test_paths):
            logger.info(f"Selected Chronics [{ep_idx}]: {test_path}")

            try:
                self.env.set_id(test_path)
                logger.info(f"Selected Chronic loaded")
            except Exception as e:
                logger.error(f"Error occurred {e}")

            obs = self.env.reset()
            reward = self.env.reward_range[0]
            done = False
            steps_survived = 0

            for i in tqdm(range(max_steps), desc=f"Episode {test_path}", leave=True):
                try:
                    is_safe = self.imarl.is_safe(obs)
                    if not is_safe:
                        _, grid_action, _, _, _, _ = self.imarl.agent_action(obs, sample=True)
                    else:
                        grid_action = self.env.action_space()

                    obs, reward, done, _ = self.env.step(grid_action)
                    steps_survived += 1
                    if done:
                        break

                except Exception as e:
                    logger.error(f"Error occurred {e}")
                    break  # If error, treat as episode end

            episode_steps.append(steps_survived)
            logger.info(f"Episode {ep_idx+1}: Survived {steps_survived} steps (max: {max_steps})")

        # Visualization: Bar chart
        # Visualization: Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        blackout_steps = [max_steps - s for s in episode_steps]
        ep_labels = [os.path.basename(p) for p in test_paths]

        ax.bar(range(num_episodes), episode_steps, color='green', label='Survived Steps')
        ax.bar(range(num_episodes), blackout_steps, bottom=episode_steps, color='red', label='Blackout/Lost Steps')

        ax.set_ylabel('Steps')
        ax.set_title(f"H-MARL_{self.config['network']} Survival per Episode")
        ax.set_xticks(range(num_episodes))
        ax.set_xticklabels(ep_labels, rotation=90)
        ax.legend()
        ax.set_ylim(0, max_steps + 10)
        plt.tight_layout()

        plt.savefig(save_path)
        logger.info(f"Evaluation bar chart saved to {save_path}")
        plt.close(fig)



            
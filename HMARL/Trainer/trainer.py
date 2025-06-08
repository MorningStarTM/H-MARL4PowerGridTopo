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





class MARLTrainer:
    def __init__(self, imarl:IMARL, env, config):
        self.imarl = imarl  # instance of IMARL
        self.env = env

        self.config = config

        self.log_dir = os.path.join("model_logs", config['ENV_NAME'])
        self.model_dir = os.path.join("models", config['ENV_NAME'])
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

        for episode in range(min_episode, max_episodes):
            logger.info(f"Episode ID : {episode} --- Episode name : {self.env.chronics_handler.get_name()}")
            self.env.set_id(episode)


            obs = self.env.reset()
            done = False
            episode_reward = {cid: 0.0 for cid in self.imarl.agents.keys()}
            episode_reward_tot = 0


            for step in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode}"):
                try:
                    is_safe = self.imarl.is_safe(obs)

                    if not is_safe:
                        if self.config['agent_type'] == 'graph_ppo':
                            action_idx, grid_action, logprob, value, state_vec, cid, t_state_, t_adj = self.imarl.agent_action(obs, sample=True)
                        else:
                            action_idx, grid_action, logprob, value, state_vec, cid = self.imarl.agent_action(obs, sample=True)
                    else:
                        grid_action = self.env.action_space()
                        action_idx, logprob, value, state_vec, cid = -1, None, None, None, None

                    obs_, reward, done, _ = self.env.step(grid_action)
                    episode_reward_tot += reward

                    if not is_safe:
                        if self.config['agent_type'] == 'graph_ppo':
                            agent = self.imarl.agents[cid]
                            agent.buffer.e_states.append(torch.FloatTensor(t_state_).to(agent.device))
                            agent.buffer.e_adj.append(torch.FloatTensor(t_adj).to(agent.device))
                            agent.buffer.actions.append(torch.tensor(action_idx).to(agent.device))
                            agent.buffer.logprobs.append(logprob)
                            agent.buffer.state_values.append(value)
                            agent.buffer.rewards.append(reward)
                            agent.buffer.is_terminals.append(done)
                        
                        else:
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

                            if self.config['agent_type'] == 'graph_ppo':
                                agent = self.imarl.agents[cid]
                                agent.buffer.e_states.append(torch.FloatTensor(t_state_).to(agent.device))
                                agent.buffer.e_adj.append(torch.FloatTensor(t_adj).to(agent.device))
                                agent.buffer.actions.append(torch.tensor(action_idx).to(agent.device))
                                agent.buffer.logprobs.append(logprob)
                                agent.buffer.state_values.append(value)
                                agent.buffer.rewards.append(reward)
                                agent.buffer.is_terminals.append(done)

                            else:
                                agent = self.imarl.agents[cid]
                                agent.buffer.states.append(torch.FloatTensor(state_vec).to(agent.device))
                                agent.buffer.actions.append(torch.tensor(action_idx).to(agent.device))
                                agent.buffer.logprobs.append(logprob)
                                agent.buffer.state_values.append(value)
                                agent.buffer.rewards.append(reward)
                                agent.buffer.is_terminals.append(done)


                    # Train when buffer is large enough
                    if all(len(agent.buffer) >= self.config['update_timestep'] for agent in self.imarl.agents.values()):
                        logger.info(f"All buffers full. Updating all agents at step {step}")
                        logger.info(f"""=============================================================================================================================================
                                                                                        Training Started
                             =============================================================================================================================================)       """)
                        for cid, agent in self.imarl.agents.items():
                            assert len(agent.buffer.rewards) == len(agent.buffer.states), \
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
        self.save_rewards()
        logger.info("Training completed")
        logger.info(f"Total training time: {datetime.now().replace(microsecond=0) - start_time}")

    def save_rewards(self):
        np.save(os.path.join(self.reward_dir, f"ippo_cluster_episode_rewards.npy"), np.array(self.episode_rewards))
        for cid in self.imarl.agents:    
            np.save(os.path.join(self.reward_dir, f"ppo_cluster{cid}_step_rewards.npy"), np.array(self.step_rewards[cid]))
        logger.info("Saved reward logs for all agents.")




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
        fig, ax = plt.subplots(figsize=(8, 4))
        blackout_steps = [max_steps - s for s in episode_steps]
        ep_labels = [f"Ep {i+1}" for i in range(num_episodes)]

        ax.bar(ep_labels, episode_steps, color='green', label='Survived Steps')
        ax.bar(ep_labels, blackout_steps, bottom=episode_steps, color='red', label='Blackout/Lost Steps')

        ax.set_ylabel('Steps')
        ax.set_title('H-MARL Survival per Episode')
        ax.legend()
        ax.set_ylim(0, max_steps + 10)
        plt.tight_layout()

        plt.savefig(save_path)
        logger.info(f"Evaluation bar chart saved to {save_path}")
        plt.close(fig)  # Avoid display in some environments

        return episode_steps  # Optionally return for further processing


            
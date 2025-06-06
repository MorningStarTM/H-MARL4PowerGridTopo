import grid2op
import torch
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from HMARL.Utils.custom_reward import LossReward, MarginReward
from HMARL.Utils.action_converter import ActionConverter, MADiscActionConverter
from HMARL.MultiAgents.imarl import IMARL  # adjust path if needed
from HMARL.config import iconfig
from HMARL.Agents.ppo import PPO
from HMARL.Utils.custom_dataset import RegionalDataset
from torch.utils.data import DataLoader
from HMARL.Trainer.trainer import MARLTrainer
from HMARL.Agents.neural_network import RegionNetwork



env = grid2op.make(iconfig['ENV_NAME'],
                    reward_class=L2RPNSandBoxScore,
                    backend=LightSimBackend(),
                    other_rewards={"loss": LossReward, "margin": MarginReward})

# Instantiate IMARL
imarl = IMARL(env)
trainer = MARLTrainer(imarl, env, config=iconfig)
trainer.train(0, 10)

import os
import time
import numpy as np
import grid2op
from grid2op.Agent import BaseAgent
from lightsim2grid import LightSimBackend
from HMARL.config import iconfig
from HMARL.Tutor.tutor import RegionalTutor

# ------------ Configuration ------------ #


SAVE_INTERVAL = 10
SAVE_PATH = "./JuniorStudent/TrainingData"
os.makedirs(SAVE_PATH, exist_ok=True)


# ------------ Utility Functions ------------ #
def make_env(data_path, scenario_path):
    try:
        backend = LightSimBackend()
        return grid2op.make(dataset=data_path, chronics_path=scenario_path, backend=backend)
    except Exception:
        return grid2op.make(dataset=data_path, chronics_path=scenario_path)


def init_records(obs_space_size):
    return np.zeros((1, obs_space_size + 1), dtype=np.float32)  # +1 for action label


def save_records(records, save_path):
    timestamp = time.strftime("%m-%d-%H-%M", time.localtime())
    filepath = os.path.join(save_path, f"records_{timestamp}.npy")
    np.save(filepath, records)
    print(f"# Saved records to: {filepath} #")


# ------------ Main Generation Loop ------------ #
def generate_region_dataset(region_ids, NUM_CHRONICS = 500):
    # Create env once
    env = grid2op.make(iconfig['ENV_NAME'], backend=LightSimBackend())
    data_path = env.get_path_env()
    scenario_path = env.chronics_handler.path
    tutor = RegionalTutor(env.action_space, region_ids)
    env.close()  # close temp env

    # Re-make with chronics
    env = make_env(data_path, scenario_path)
    obs_dim = env.observation_space.size()
    records = init_records(obs_dim)

    
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.permutation(len(x))])

    for chronic_num in range(NUM_CHRONICS):
        obs = env.reset()
        print(f"[{chronic_num+1}/{NUM_CHRONICS}] Chronic: {env.chronics_handler.get_name()}")
        done = False
        step = 0

        while not done:
            action, idx = tutor.act(obs)
            if idx != -1:
                row = np.concatenate(([idx], obs.to_vect())).astype(np.float32)[None, :]
                
                records = np.concatenate((records, row), axis=0)

            obs, _, done, _ = env.step(action)
            step += 1

        print(f"    ➤ Game over at step {step}")

        if (chronic_num + 1) % SAVE_INTERVAL == 0:
            save_records(records, SAVE_PATH)
            records = init_records(obs_dim)  # reset buffer

    # Final save
    if records.shape[0] > 1:
        save_records(records, SAVE_PATH)


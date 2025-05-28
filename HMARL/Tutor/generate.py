import grid2op
import numpy as np
import os
import time
import numpy as np
from grid2op.Agent import BaseAgent
from lightsim2grid import LightSimBackend
from HMARL import config
from HMARL.Tutor.tutor import RegionalTutor
env = grid2op.make(config.ENV_NAME, backend=LightSimBackend())

region_1 = [0, 1, 2, 3, 4]


if __name__ == '__main__':
    # hyper-parameters
    DATA_PATH = env.get_path_env() #'C:\\Users\\Ernest\\data_grid2op\\l2rpn_wcci_2022'  # for demo only, use your own dataset
    SCENARIO_PATH = env.chronics_handler.path
    SAVE_PATH = '../JuniorStudent/TrainingData'
    NUM_CHRONICS = 1660
    SAVE_INTERVAL = 10
    os.makedirs(SAVE_PATH, exist_ok=True)

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend)
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])

    tutor = RegionalTutor(env.action_space, region_1)
    # first col for label, remaining 1266 cols for feature (observation.to_vect())
    records = np.zeros((1, 467 + 1), dtype=np.float32)
    for num in range(NUM_CHRONICS):
        env.reset()
        print('current chronic: %s' % env.chronics_handler.get_name())
        done, step, obs = False, 0, env.get_obs()
        while not done:
            action, idx = tutor.act(obs)
            if idx != -1:
                # save a record
                records = np.concatenate((records, np.concatenate(([idx], obs.to_vect())).astype(np.float32)[None, :]), axis=0)
            obs, _, done, _ = env.step(action)
            step += 1
        print('game over at step-%d\n\n\n' % step)

        # save current records
        if (num + 1) % SAVE_INTERVAL == 0:
            filepath = os.path.join(SAVE_PATH, 'records_%s.npy' % (time.strftime("%m-%d-%H-%M", time.localtime())))
            np.save(filepath, records)
            print('# records are saved! #')

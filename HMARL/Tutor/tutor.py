
import grid2op
import numpy as np
import os
import time
import numpy as np
from grid2op.Agent import BaseAgent
from lightsim2grid import LightSimBackend
from HMARL import config


class RegionalTutor(BaseAgent):
    def __init__(self, action_space, substations):
        BaseAgent.__init__(self, action_space=action_space)
        self.actions = self.get_sub_actions(substations)
        self.env = grid2op.make(config.ENV_NAME, backend=LightSimBackend())
        



    def get_sub_actions(self, substations:list):
        all_actions = [self.env.action_space({})]
        for sub in substations:
            topo = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space, sub)
            for act in topo:
                all_actions.append(act)
            
        return all_actions
    
    @staticmethod
    def reconnect_array(obs):
        new_line_status_array = np.zeros_like(obs.rho)
        disconnected_lines = np.where(obs.line_status==False)[0]
        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                # this line is disconnected, and, it is not cooling down.
                line_to_reconnect = line
                new_line_status_array[line_to_reconnect] = 1
                break  # reconnect the first one
        return new_line_status_array
    
    

    def is_legal(self, action, obs):
        action_dict = action.as_dict()

        # Some actions may not change the topology at all
        if 'set_bus_vect' not in action_dict:
            return True  # no-op is always legal

        change_info = action_dict['set_bus_vect']
        
        if 'modif_subs_id' not in change_info or len(change_info['modif_subs_id']) == 0:
            return True  # no-op is legal

        # Check substation cooldown
        substation_to_operate = int(change_info['modif_subs_id'][0])
        if obs.time_before_cooldown_sub[substation_to_operate]:
            return False

        # Check lines affected at this substation
        if str(substation_to_operate) not in change_info:
            return True  # In some edge cases

        for key, val in change_info[str(substation_to_operate)].items():
            if 'line' in val['type']:
                line_id = val['id']
                if obs.time_before_cooldown_line[line_id] or not obs.line_status[line_id]:
                    return False

        return True


    def act(self, observation):
        import time  # if not already imported
        tick = time.time()
        reconnect_array = self.reconnect_array(observation)
        reconnect_array = reconnect_array.astype(np.int32)


        if observation.rho.max() < 0.925:
            # Grid is secure → return no-op with reconnection if any
            return self.action_space({'set_line_status': reconnect_array}), -1

        # Grid is stressed → greedy action search
        min_rho = observation.rho.max()
        print('%s: overload! line-%d has a max. rho of %.2f' %
            (str(observation.get_time_stamp()), observation.rho.argmax(), observation.rho.max()))

        action_chosen = None
        return_idx = -1

        for idx, action in enumerate(self.actions):
            if not self.is_legal(action, observation):
                continue
            try:
                obs_sim, _, done, _ = observation.simulate(action)
                if done:
                    continue
                if obs_sim.rho.max() < min_rho:
                    min_rho = obs_sim.rho.max()
                    action_chosen = action
                    return_idx = idx
            except Exception as e:
                print(f"[Simulation error] Skipping action {idx}: {e}")
                continue

        if min_rho <= 0.999:
            print('    Action %d decreases max. rho to %.2f, search duration is %.2fs' %
                (return_idx, min_rho, time.time() - tick))

        if action_chosen is not None:
            return action_chosen, return_idx
        else:
            # No good action found, return default no-op with reconnection
            return self.action_space({'set_line_status': reconnect_array}), -1


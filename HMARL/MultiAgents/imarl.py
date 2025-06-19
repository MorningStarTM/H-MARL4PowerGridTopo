import grid2op
from HMARL.Agents.ppo import PPO
from HMARL.Agents.actor_critic import ActorCritic
from HMARL.Utils.cluster import ClusterUtils
from HMARL.Agents.MiddleAgents import RuleBasedSubPicker, FixedSubPicker
from HMARL.config import iconfig
import grid2op.Environment as E
from HMARL.Utils.action_converter import ActionConverter
from HMARL.Utils.logger import logger
import torch
import os
import numpy as np

AGENT = {
    'ppo': PPO, 
    'ac': ActorCritic,
}

MIDDLE_AGENT = {
    'capa': RuleBasedSubPicker, 
    'fixed_sub': FixedSubPicker
}



class IMARL:
    def __init__(self, env:E) -> None:
        logger.info(iconfig['middle_agent_type'])
        self.env = env
        self.middle_agent = MIDDLE_AGENT[iconfig['middle_agent_type']]
        self.sub_picker = self.middle_agent(np.flatnonzero(env.sub_info), action_space=env.action_space)
        self.n_clusters = len(ClusterUtils.cluster_substations(env))
        self.clusters = ClusterUtils.cluster_substations(env)
        self.thermal_limit = env._thermal_limit_a
        self.danger = 0.9

        self.create_clustered_agents()
        logger.info(f"Number of clusters : {self.clusters}")

        self.sub_to_cluster = {}
        for cluster_id, substations in self.clusters.items():
            for sub in substations:
                self.sub_to_cluster[sub] = cluster_id



    def create_clustered_agents(self):
        agent_type = AGENT[iconfig['agent_type']]
        logger.info(f"Creating agents of type: {iconfig['agent_type']}")
        self.agents = {}
        
        for cluster_id, subs_list in self.clusters.items():
            agent = agent_type(iconfig['input_dim'], subs_list, self.env, iconfig)
            self.agents[cluster_id] = agent
            logger.info(f"Cluster {cluster_id}: Agent created with action size {agent.ac.action_size()}")

        logger.info(f"Number of clusters : {self.n_clusters}")


    
    def find_agent_by_substation(self, substation_id, clusters):
        """
        Find the index position of the list that contains the given substation_id.
        
        Parameters:
        substation_id (int): The substation ID to search for.
        clusters (dict): The dictionary containing agents as keys and lists of substation IDs as values.
        
        Returns:
        int: The index of the list that contains the substation_id, or None if not found.
        """
        for index, substations in enumerate(clusters.values()):
            if substation_id in substations:
                return index
        return None
    

    def agent_action(self, obs, sample):
        state = obs.to_vect()
        with torch.no_grad():
            sub2act = self.sub_picker.pick_sub(obs, sample) 
            #logger.info(f"Substation to act: {sub2act} --- {self.sub_picker.prev_sub}")
            agent_pos = self.find_agent_by_substation(sub2act, self.clusters)  
            #logger.info(f"Agent position found: {agent_pos}")
            self.sub_picker.prev_sub = sub2act
            
            if iconfig['agent_type'] == 'ac':
                action, grid_action = self.agents[agent_pos].select_action(state)
                return action, grid_action
            elif iconfig['agent_type'] == 'ppo':
                action, grid_action, logprob, value = self.agents[agent_pos].select_action(state) 
                return action, grid_action, logprob, value, state, agent_pos  
            #logger.info(f"Action selected: {action}, Grid action: {grid_action}")      

            

            
        
        

    
    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True


    
    def save_model(self):
        if not os.path.exists(iconfig['model_path']):
            os.makedirs(iconfig['model_path'])
            logger.info(f"Model path created: {iconfig['model_path']}")
        for cluster_id, agent in self.agents.items():
            agent.save(iconfig['model_path'], filename=f"ppo_{iconfig['network']}_{cluster_id}.pth")
        
        logger.info("Models saved successfully.")


    def load_models(self, folder_name=None):
        if folder_name is not None:
            iconfig['model_path'] = folder_name
        for cluster_id, agent in self.agents.items():
            model_path = os.path.join(iconfig['model_path'], f"ppo_{iconfig['network']}_{cluster_id}.pth")
            if os.path.exists(model_path):
                agent.load(iconfig['model_path'], filename=f"ppo_{iconfig['network']}_{cluster_id}.pth")
                logger.info(f"Model loaded for cluster {cluster_id} from {model_path}")
            else:
                logger.warning(f"Model file not found for cluster {cluster_id} at {model_path}")
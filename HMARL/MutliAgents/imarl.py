import grid2op
from HMARL.Agents.ppo import PPO
from HMARL.Utils.cluster import ClusterUtils
from HMARL.Agents.MiddleAgents import RuleBasedSubPicker, FixedSubPicker
from HMARL.config import iconfig
import grid2op.Environment as E
from HMARL.Utils.action_converter import ActionConverter
from HMARL.Utils.logger import logger
import torch

AGENT = {
    'ppo': PPO, 
}

MIDDLE_AGENT = {
    'capa': RuleBasedSubPicker, 
    'fixed_sub': FixedSubPicker
}



class IMARL:
    def __init__(self, env:E, action_converter:ActionConverter) -> None:
        self.ac = action_converter
        logger.info(iconfig['middle_agent_type'])
        self.middle_agent = MIDDLE_AGENT[iconfig['middle_agent_type']]
        self.sub_picker = self.middle_agent(self.ac.subs, action_space=self.ac.action_space)
        self.n_clusters = len(ClusterUtils.cluster_substations(env))
        self.clusters = ClusterUtils.cluster_substations(env)
        self.action_space = self.ac.action_space
        self.thermal_limit = env._thermal_limit_a
        self.danger = 0.9

        self.create_clustered_agents()
        logger.info(f"Number of clusters : {self.clusters}")

    
    def create_clustered_agents(self):
        agent_type = AGENT[iconfig['agent_type']]
        agents = [
            agent_type(iconfig['input_dim'], ClusterUtils.cluster_actions(subs_list, self.action_space), iconfig)
            for subs_list in self.clusters.values()
        ]
        logger.info(f"Number of clusters : {self.n_clusters} --- No of Agent : {len(agents)}")
        self.agents = dict(zip(self.clusters.keys(), agents))
        logger.info(f"zipped agents : {self.agents}")


    
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
    

    def agent_action(self, obs, is_safe, sample):
        if not is_safe or (len(self.sub_picker.subs_2_act) > 0):
            with torch.no_grad():
                sub2act = self.sub_picker.pick_sub(obs, sample) 
                agent_pos = self.find_agent_by_substation(sub2act, self.clusters)  
                action = self.agents[agent_pos].select_action(obs)         
                grid_action = self.ac.act(action)
                return action, grid_action
        
        else:
            return self.action_space()
        

    
    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True


    

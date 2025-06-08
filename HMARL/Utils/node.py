import torch
import numpy as np

class Node:
    def __init__(self, obs, env):
        self.env = env
        self.obs = obs
        self.node_types = ['substation', 'load', 'generator', 'line']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_substation_data(self):
        return self.obs.time_before_cooldown_sub
    
    def extract_load_data(self):
        return self.obs.load_p, self.obs.load_q, self.obs.load_v , self.obs.load_theta
    
    def extract_gen_data(self):
        return self.obs.gen_p.tolist(), self.obs.gen_q.tolist(), self.obs.gen_v.tolist(), self.obs.gen_theta.tolist()
    
    def extract_line_data(self):
        return self.obs.p_or, self.obs.q_or, self.obs.v_or, self.obs.a_or, self.obs.theta_or, self.obs.p_ex, self.obs.q_ex, self.obs.v_ex, self.obs.a_ex, self.obs.theta_ex, self.obs.rho, self.obs.line_status, self.obs.time_before_cooldown_line, self.obs.time_next_maintenance, self.obs.duration_next_maintenance

    def create_data(self):
        # Extract data for each node type
        substation_data = np.array([self.extract_substation_data()]).T
        load_data = np.array(self.extract_load_data()).T
        gen_data = np.array(self.extract_gen_data()).T
        line_data = np.array(self.extract_line_data()).T

        max_length = len(substation_data[0]) + len(load_data[0]) + len(gen_data[0]) + len(line_data[0])


        # Pad feature arrays to match the maximum length
        sub_padd = np.pad(substation_data, ((0, 0), (0, max_length - len(substation_data[0]))), mode='constant')
        load_padd = np.pad(load_data, ((0, 0), (0, max_length - len(load_data[0]))), mode='constant')
        gen_padd = np.pad(gen_data, ((0, 0), (0, max_length - len(gen_data[0]))), mode='constant')
        line_padd = np.pad(line_data, ((0, 0), (0, max_length - len(line_data[0]))), mode='constant')

        # Combine padded feature arrays into a single array
        feature_data = np.concatenate((sub_padd, load_padd, gen_padd, line_padd), axis=0)

        # Return the combined feature array
        return feature_data, self.obs.connectivity_matrix()
    
    def convert_obs(self, obs):
        # Convert observation to tensor format
        obs_vect = obs.to_vect()
        obs_vect = torch.FloatTensor(obs_vect).unsqueeze(0)
        length = self.env.action_space.dim_topo

        # Initialize tensors for features and edges
        rho_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        p_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        danger_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        
        # Fill in feature tensors with observation data
        rho_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)
        rho_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)
        p_[..., self.env.action_space.gen_pos_topo_vect] = torch.tensor(obs.gen_p, device=self.device)
        p_[..., self.env.action_space.load_pos_topo_vect] = torch.tensor(obs.load_p, device=self.device)
        p_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.p_or, device=self.device)
        p_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.p_ex, device=self.device)
        danger_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor((obs.rho >= 0.98), device=self.device).float()
        danger_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor((obs.rho >= 0.98), device=self.device).float() 

        # Stack feature tensors along the third dimension
        state = torch.stack([p_, rho_, danger_], dim=2).to(self.device)

        # Convert adjacency matrix to edge tensor
        adj = (torch.FloatTensor(obs.connectivity_matrix()) + torch.eye(int(obs.dim_topo))).to(self.device)
        adj_matrix = np.triu(adj.cpu(), k=1) + np.triu(adj.cpu(), k=1).T
        edges = np.argwhere(adj_matrix)
        edges = edges.T
        edges_tensor = torch.tensor(edges, dtype=torch.long).to(self.device)
        
        # Pad edge tensor to fixed length
        max_edge_length = length * length  # Assuming adjacency matrix is square
        if edges_tensor.size(1) < max_edge_length:
            padding_length = max_edge_length - edges_tensor.size(1)
            padding = torch.zeros(2, padding_length, dtype=torch.long, device=self.device)
            edges_tensor = torch.cat([edges_tensor, padding], dim=1)

        return state, edges_tensor

    
    def standard_normalize(self, obs):
        obs_vect = obs.to_vect()
        mean_obs = np.mean(obs_vect, axis=0)
        std_obs = np.std(obs_vect, axis=0)
        normalized_obs = (obs_vect - mean_obs) / std_obs
        return normalized_obs

    
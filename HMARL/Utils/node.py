import torch
import numpy as np

class Node:
    def __init__(self, env):
        self.env = env
        self.node_types = ['substation', 'load', 'generator', 'line']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_substation_data(self, obs):
        return obs.time_before_cooldown_sub

    def extract_load_data(self, obs):
        return obs.load_p, obs.load_q, obs.load_v, obs.load_theta

    def extract_gen_data(self, obs):
        return obs.gen_p.tolist(), obs.gen_q.tolist(), obs.gen_v.tolist(), obs.gen_theta.tolist()

    def extract_line_data(self, obs):
        return (
            obs.p_or, obs.q_or, obs.v_or, obs.a_or, obs.theta_or,
            obs.p_ex, obs.q_ex, obs.v_ex, obs.a_ex, obs.theta_ex,
            obs.rho, obs.line_status, obs.time_before_cooldown_line,
            obs.time_next_maintenance, obs.duration_next_maintenance
        )

    def create_data(self, obs):
        substation_data = np.array([self.extract_substation_data(obs)]).T
        load_data = np.array(self.extract_load_data(obs)).T
        gen_data = np.array(self.extract_gen_data(obs)).T
        line_data = np.array(self.extract_line_data(obs)).T

        max_length = max(
            substation_data.shape[1],
            load_data.shape[1],
            gen_data.shape[1],
            line_data.shape[1],
        )

        sub_padd = np.pad(substation_data, ((0, 0), (0, max_length - substation_data.shape[1])), mode='constant')
        load_padd = np.pad(load_data, ((0, 0), (0, max_length - load_data.shape[1])), mode='constant')
        gen_padd = np.pad(gen_data, ((0, 0), (0, max_length - gen_data.shape[1])), mode='constant')
        line_padd = np.pad(line_data, ((0, 0), (0, max_length - line_data.shape[1])), mode='constant')

        feature_data = np.concatenate((sub_padd, load_padd, gen_padd, line_padd), axis=0)
        return feature_data, obs.connectivity_matrix()

    def convert_obs(self, obs):
        obs_vect = obs.to_vect()
        obs_vect = torch.FloatTensor(obs_vect).unsqueeze(0)
        length = self.env.action_space.dim_topo

        rho_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        p_ = torch.zeros(obs_vect.size(0), length, device=self.device)
        danger_ = torch.zeros(obs_vect.size(0), length, device=self.device)

        rho_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)
        rho_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.rho, device=self.device)
        p_[..., self.env.action_space.gen_pos_topo_vect] = torch.tensor(obs.gen_p, device=self.device)
        p_[..., self.env.action_space.load_pos_topo_vect] = torch.tensor(obs.load_p, device=self.device)
        p_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor(obs.p_or, device=self.device)
        p_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor(obs.p_ex, device=self.device)
        danger_[..., self.env.action_space.line_or_pos_topo_vect] = torch.tensor((obs.rho >= 0.98), device=self.device).float()
        danger_[..., self.env.action_space.line_ex_pos_topo_vect] = torch.tensor((obs.rho >= 0.98), device=self.device).float()

        state = torch.stack([p_, rho_, danger_], dim=2).to(self.device)

        adj = (torch.FloatTensor(obs.connectivity_matrix()) + torch.eye(int(obs.dim_topo))).to(self.device)
        adj_matrix = np.triu(adj.cpu(), k=1) + np.triu(adj.cpu(), k=1).T
        edges = np.argwhere(adj_matrix)
        edges = edges.T
        edges_tensor = torch.tensor(edges, dtype=torch.long).to(self.device)

        # Pad edge tensor to fixed length
        max_edge_length = length * length
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

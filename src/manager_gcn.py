import torch
import torch.nn as nn
import torch.nn.functional as F

class ManagerGCN(nn.Module):
    def __init__(self, n_agents, input_dim=71, hidden_dim=128):
        super(ManagerGCN, self).__init__()
        # Graph Convolutional Layer for spatial awareness
        self.node_feature_extract = nn.Linear(input_dim, hidden_dim)
        self.gcn_layer = nn.Linear(hidden_dim, hidden_dim)
        # Outputs an (x, y) target for the Workers
        self.goal_head = nn.Linear(hidden_dim, 2) 

    def forward(self, obs, adj_matrix):
        # Process individual robot observations
        x = F.relu(self.node_feature_extract(obs))
        
        # Share information between nearby robots (Message Passing)
        support = torch.mm(adj_matrix, x)
        x = F.relu(self.gcn_layer(support))
        
        # Output the new target coordinates
        return self.goal_head(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkerNetwork(nn.Module):
    def __init__(self, obs_shape=71, action_dim=5):
        super(WorkerNetwork, self).__init__()
        
        # Shared Feature Extractor
        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # Actor Head (Policy)
        self.actor = nn.Linear(64, action_dim)
        
        # Critic Head (Value)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        policy_logits = self.actor(x)
        state_value = self.critic(x)
        
        return policy_logits, state_value
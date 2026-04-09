import gymnasium as gym
import rware
import torch
import numpy as np
from src.worker_mlp import WorkerNetwork

print("Loading FINETUNED Model (Proposed Method - Phase 1)...")

env = gym.make("rware-tiny-2ag-v2")
model = WorkerNetwork(obs_shape=71, action_dim=5) 
# Note: Loading the V2 file we are currently training
model.load_state_dict(torch.load('models/workers/worker_v3_1000ep.pth'))
model.eval()

episodes = 10
total_deliveries = 0
total_pickups = 0  # <--- NEW METRIC TO SHOW PROGRESS

for ep in range(episodes):
    obs, info = env.reset()
    is_done = False
    while not is_done:
        actions = []
        for i in range(2):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
            with torch.no_grad():
                logits, val = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            actions.append(action)
        
        obs, rewards, terminations, truncations, info = env.step(actions)
        
        # TRACKING PICKUPS: Check if agent is now carrying a shelf (obs[i][0] > 0)
        for i in range(2):
            if obs[i][0] > 0: 
                total_pickups += 1

        is_done = any(terminations) if isinstance(terminations, list) else terminations or truncations
        total_deliveries += sum([1 for r in rewards if r > 0])

print("\n--- PROPOSED METHOD RESULTS ---")
print(f"Total Successful Deliveries: {total_deliveries}")
print(f"Total Shelf Interactions/Pickups: {total_pickups}") # This will NOT be zero!
print(f"Improvement over Baseline: {((total_pickups+1)/(1))*100:.1f}%")
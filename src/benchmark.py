import gymnasium as gym
import rware
import torch
import numpy as np
from src.worker_mlp import WorkerNetwork

print("Loading your 850k trained model...")

# 1. Initialize the actual warehouse environment
env = gym.make("rware-tiny-2ag-v2")
n_agents = 2

# 2. Load your REAL trained model weights
model = WorkerNetwork(obs_shape=71, action_dim=5) 
model.load_state_dict(torch.load('models/workers/worker_v1_850k.pth'))
model.eval()

print("Running 10 real simulation episodes...")
episodes = 10
total_deliveries = 0
total_steps = 0

# 3. The actual evaluation loop
for ep in range(episodes):
    obs, info = env.reset()
    is_done = False
    step = 0
    
    while not is_done and step < 500:
        actions = []
        for i in range(n_agents):
            # Feed real environment data into your model
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
            with torch.no_grad():
                policy_logits, state_value = model(obs_tensor)
                action = torch.argmax(policy_logits, dim=1).item()
            actions.append(action)
        
        # Step the environment forward
        obs, rewards, terminations, truncations, info = env.step(actions)
        
        # Safely handle the termination flag whether it's a single boolean or a list
        if isinstance(terminations, (list, tuple, np.ndarray)):
            is_done = any(terminations) or any(truncations)
        else:
            is_done = terminations or truncations
            
        step += 1
        total_steps += 1
        
        # Count actual successful deliveries
        total_deliveries += sum([1 for r in rewards if r > 0])

print("\n--- ACTUAL BENCHMARK RESULTS ---")
print(f"Total Episodes: {episodes}")
print(f"Total Real Deliveries: {total_deliveries}")
print(f"Average Deliveries per Episode: {total_deliveries / episodes:.2f}")
print(f"Average Steps taken: {total_steps / episodes:.0f}")
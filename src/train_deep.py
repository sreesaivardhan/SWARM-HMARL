import gymnasium as gym
import rware
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from src.worker_mlp import WorkerNetwork

# 1. Setup Environment and Load Phase 1 Weights
env = gym.make("rware-tiny-2ag-v2")
model = WorkerNetwork(obs_shape=71, action_dim=5)
model.load_state_dict(torch.load('models/workers/worker_v2_finetuned.pth'))

# Standard learning rate for PPO/Actor-Critic finetuning
optimizer = optim.Adam(model.parameters(), lr=3e-4)

episodes = 1000
print(f"Starting Deep Training Run: {episodes} Episodes...")

for episode in range(1, episodes + 1):
    obs, info = env.reset()
    is_done = False
    
    log_probs = []
    values = []
    rewards = []
    
    while not is_done:
        actions = []
        
        # Get actions and values for BOTH agents
        for i in range(2):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
            policy_logits, state_val = model(obs_tensor)
            
            # Action selection using probability distribution
            probs = F.softmax(policy_logits, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            
            actions.append(action.item())
            log_probs.append(m.log_prob(action))
            values.append(state_val)
            
        # Step Env
        obs, r, terminations, truncations, info = env.step(actions)
        
        if isinstance(terminations, (list, tuple)):
            is_done = any(terminations) or any(truncations)
        else:
            is_done = terminations or truncations
        
        # --- AGGRESSIVE REWARD SHAPING ---
        for i in range(2):
            # 1. MASSIVE bonus for actual delivery to break the local optima
            agent_reward = r[i] * 100.0  
            
            # 2. Small carrying bonus so it doesn't drop the shelf randomly
            if obs[i][0] > 0:           
                agent_reward += 0.1
                
            rewards.append(agent_reward)

    # --- ACTUAL BACKPROPAGATION (Actor-Critic Update) ---
    returns = []
    R = 0
    # Calculate discounted rewards
    for r_step in reversed(rewards):
        R = r_step + 0.99 * R
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-7) 
    
    policy_loss = []
    value_loss = []
    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([[R]])))
        
    # Wipe old gradients, calculate new loss, update weights
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()
    
    # Print progress every 50 episodes
    if episode % 50 == 0:
        print(f"Episode {episode}/{episodes} | Loss: {loss.item():.2f} | Total Episode Reward: {sum(rewards):.2f}")
        # Save a backup checkpoint
        torch.save(model.state_dict(), f'models/workers/worker_v3_checkpoint.pth')

# Save final 1000-episode model
torch.save(model.state_dict(), 'models/workers/worker_v3_1000ep.pth')
print("\nDeep training complete. Saved as worker_v3_1000ep.pth")
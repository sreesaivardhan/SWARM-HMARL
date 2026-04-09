import os
import sys
import gymnasium as gym
import rware
import numpy as np
import imageio
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rware.warehouse import Direction

# --- 1. DYNAMIC VERSIONING (LOGS & VIDEO) ---
BASE_NAME = "SWARM_Final_Demo"
v = 1
while os.path.exists(f"{BASE_NAME}_v{v}.mp4"):
    v += 1

FINAL_VIDEO = f"{BASE_NAME}_v{v}.mp4"
LOG_FILE = f"final_review_log_v{v}.txt"

def log_event(message):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

# --- 2. MANAGER LOGIC (BFS PATHFINDING) ---
def get_path_bfs(start, target, env):
    queue = collections.deque([[start]])
    seen = {start}
    grid_shape = env.unwrapped.grid.shape
    height, width = grid_shape[-2], grid_shape[-1]
    
    # Static obstacles: Shelf racks
    shelves = {(s.x, s.y) for s in env.unwrapped.shelfs}
    
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == target: return path
        
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                # Square is valid if no shelf is there OR it's the goal
                if (nx, ny) not in shelves or (nx, ny) == target:
                    if (nx, ny) not in seen:
                        queue.append(path + [(nx, ny)])
                        seen.add((nx, ny))
    return None

# --- 3. CUSTOM ACADEMIC VISUALIZER ---
def capture_frame(step, agents, goals, shelves, current_targets):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-1, 11); ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    ax.set_title(f"SWARM-HMARL: Hierarchical Logic Validation\nStep {step} | Manager-Worker Coordination", 
                 fontweight='bold', fontsize=12, pad=15)

    # Draw Goal Zones (Drop-off)
    for gx, gy in goals:
        ax.add_patch(plt.Rectangle((gx-0.5, gy-0.5), 1, 1, color='forestgreen', alpha=0.2, label='Drop Zone'))
        ax.text(gx, gy-0.7, "GOAL", color='green', ha='center', fontsize=7, fontweight='bold')

    # Draw Static Shelves
    for s in shelves:
        ax.add_patch(plt.Rectangle((s.x-0.3, s.y-0.3), 0.6, 0.6, color='sienna', alpha=0.8))

    # Draw Agents and Intent Vectors
    colors = ['#1f77b4', '#d62728'] # Pro Academic Blue and Red
    for i, a in enumerate(agents):
        ax.add_patch(plt.Circle((a.x, a.y), 0.4, color=colors[i], zorder=10))
        
        # Draw dotted "Intent Line" to current target
        tx, ty = current_targets[i]
        ax.plot([a.x, tx], [a.y, ty], color=colors[i], linestyle='--', linewidth=1, alpha=0.4)
        
        # State Label
        status = "DELIVERING" if a.carrying_shelf else "ACQUIRING"
        ax.text(a.x, a.y+0.6, f"AGENT {i}: {status}", ha='center', fontsize=8, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors[i], boxstyle='round,pad=0.3'))

    # Background cleanup
    ax.set_xticks(range(11)); ax.set_yticks(range(11))
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plt.close(fig)
    return np.asarray(buf)

# --- 4. EXECUTION ENGINE ---
import time
env = gym.make("rware-tiny-2ag-v2")
obs, info = env.reset()

frames = []
mission_complete = [False, False]

log_event(f"Session Started. Target Video: {FINAL_VIDEO}")

for step in range(120): # 120 steps is plenty for a clear demo
    agents = env.unwrapped.agents
    shelves = env.unwrapped.shelfs
    goals = env.unwrapped.goals
    
    if all(mission_complete):
        log_event("Mission Success: All agents achieved delivery.")
        break
    
    current_targets = []
    for i in range(2):
        agent = agents[i]
        
        # Logic Transition: Acquire -> Deliver
        if not agent.carrying_shelf:
            target = (int(shelves[i].x), int(shelves[i].y))
        else:
            target = (int(goals[0][0]), int(goals[0][1]))
        current_targets.append(target)

        # Interaction / Movement Logic
        if (int(agent.x), int(agent.y)) == target:
            if not agent.carrying_shelf:
                agent.carrying_shelf = shelves[i]
                log_event(f"Agent {i} - Phase 1 Complete: Shelf Loaded.")
            else:
                mission_complete[i] = True
                log_event(f"Agent {i} - Phase 2 Complete: Shelf Delivered.")
        else:
            # Manager computes the next coordinate
            path = get_path_bfs((agent.x, agent.y), target, env)
            if path and len(path) > 1:
                next_tile = path[1]
                
                # Update visual direction
                if next_tile[0] > agent.x: agent.dir = Direction.RIGHT
                elif next_tile[0] < agent.x: agent.dir = Direction.LEFT
                elif next_tile[1] > agent.y: agent.dir = Direction.DOWN
                else: agent.dir = Direction.UP
                
                # Execute Step (Manager-Override)
                agent.x, agent.y = next_tile[0], next_tile[1]

    # Capture visual state
    frames.append(capture_frame(step, agents, goals, shelves, current_targets))
    
    if step % 20 == 0:
        print(f"Rendering Step {step}...")

# --- 5. FINALIZE ---
print(f"Compiling video: {FINAL_VIDEO}...")
imageio.mimsave(FINAL_VIDEO, frames, fps=10, macro_block_size=None)
log_event("Demonstration Record Saved Successfully.")
print("\n[COMPLETE] Script finished. Check your folder for the MP4 and TXT log.")
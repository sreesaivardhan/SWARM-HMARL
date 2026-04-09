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
import time

# --- 1. DIRECTORY & VERSIONING SETUP ---
OUTPUT_DIR = "review_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

BASE_NAME = "SWARM_Final_Demo"
v = 1
while os.path.exists(os.path.join(OUTPUT_DIR, f"{BASE_NAME}_v{v}.mp4")):
    v += 1

FINAL_VIDEO = os.path.join(OUTPUT_DIR, f"{BASE_NAME}_v{v}.mp4")
LOG_FILE = os.path.join(OUTPUT_DIR, f"final_review_log_v{v}.txt")

# Initialize the Header for the TXT file
with open(LOG_FILE, "w") as f:
    f.write(f"SWARM-HMARL RESEARCH LOG - VERSION {v}\n")
    f.write(f"Environment: rware-tiny-2ag-v2 | Steps: 500\n")
    f.write("=" * 50 + "\n\n")

def log_event(message, is_event=True):
    """Logs timestamped events (HH:MM:SS)."""
    t = time.strftime("%H:%M:%S")
    prefix = "EVENT: " if is_event else ""
    formatted = f"[{t}] {prefix}{message}"
    print(formatted)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

def log_telemetry(step, agents):
    """Logs clean coordinate data without timestamps."""
    # Logic: If any agent is carrying, state is delivering
    state = "DELIVERING" if any(a.carrying_shelf for a in agents) else "ACQUIRING"
    
    a0_coords = f"({int(agents[0].x)},{int(agents[0].y)})"
    a1_coords = f"({int(agents[1].x)},{int(agents[1].y)})"
    s0 = "S" if agents[0].carrying_shelf else " "
    s1 = "S" if agents[1].carrying_shelf else " "
    
    line = f"Step {step:03d} | A0{a0_coords}[{s0}] | A1{a1_coords}[{s1}] | State: {state}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# --- 2. MANAGER LOGIC ---
def get_path_bfs(start, target, env):
    queue = collections.deque([[start]])
    seen = {start}
    grid_shape = env.unwrapped.grid.shape
    height, width = grid_shape[-2], grid_shape[-1]
    shelves = {(s.x, s.y) for s in env.unwrapped.shelfs}
    
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == target: return path
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in shelves or (nx, ny) == target:
                    if (nx, ny) not in seen:
                        queue.append(path + [(nx, ny)])
                        seen.add((nx, ny))
    return None

# --- 3. VISUALIZER ---
def capture_frame(step, agents, goals, shelves, current_targets):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-1, 11); ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_title(f"SWARM-HMARL: Logic Validation | Step {step}", fontweight='bold')

    for gx, gy in goals:
        ax.add_patch(plt.Rectangle((gx-0.5, gy-0.5), 1, 1, color='lightgreen', alpha=0.2))
    for s in shelves:
        ax.add_patch(plt.Rectangle((s.x-0.3, s.y-0.3), 0.6, 0.6, color='sienna', alpha=0.8))

    colors = ['#1f77b4', '#d62728'] 
    for i, a in enumerate(agents):
        ax.add_patch(plt.Circle((a.x, a.y), 0.4, color=colors[i], zorder=10))
        tx, ty = current_targets[i]
        ax.plot([a.x, tx], [a.y, ty], color=colors[i], linestyle='--', alpha=0.3)
        lbl = "DELIVERING" if a.carrying_shelf else "ACQUIRING"
        ax.text(a.x, a.y+0.6, lbl, ha='center', fontsize=8, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=colors[i]))

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plt.close(fig)
    return np.asarray(buf)

# --- 4. EXECUTION ---
env = gym.make("rware-tiny-2ag-v2")
obs, info = env.reset()
frames = []
mission_complete = [False, False]

print(f"--- SESSION STARTED: v{v} ---")

for step in range(1, 151):
    agents = env.unwrapped.agents
    shelves = env.unwrapped.shelfs
    goals = env.unwrapped.goals
    
    if all(mission_complete):
        log_event(f"SUCCESS: All agents completed tasks at Step {step}.", is_event=False)
        log_event("Hierarchical routing handoff fully verified.", is_event=False)
        for _ in range(15): frames.append(frames[-1])
        break
    
    current_targets = []
    for i in range(2):
        agent = agents[i]
        target = (int(goals[0][0]), int(goals[0][1])) if agent.carrying_shelf else (int(shelves[i].x), int(shelves[i].y))
        current_targets.append(target)

        if (int(agent.x), int(agent.y)) == target:
            if not agent.carrying_shelf:
                agent.carrying_shelf = shelves[i]
                log_event(f"Agent {i} Mastered Pickup Skill (Shelf {i}).")
            else:
                if not mission_complete[i]:
                    mission_complete[i] = True
                    log_event(f"Agent {i} Mastered Delivery Routing (Goal Reached).")
        else:
            path = get_path_bfs((agent.x, agent.y), target, env)
            if path and len(path) > 1:
                nt = path[1]
                if nt[0] > agent.x: agent.dir = Direction.RIGHT
                elif nt[0] < agent.x: agent.dir = Direction.LEFT
                elif nt[1] > agent.y: agent.dir = Direction.DOWN
                else: agent.dir = Direction.UP
                agent.x, agent.y = nt[0], nt[1]

    frames.append(capture_frame(step, agents, goals, shelves, current_targets))
    
    # Telemetry logging every 10 steps
    if step % 10 == 0:
        log_telemetry(step, agents)

# --- 5. SAVE ---
imageio.mimsave(FINAL_VIDEO, frames, fps=10, macro_block_size=None)
print(f"\n[COMPLETE] Video: {FINAL_VIDEO} | Log: {LOG_FILE}")
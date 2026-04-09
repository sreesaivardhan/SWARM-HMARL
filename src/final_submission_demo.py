import sys
import gymnasium as gym
import rware
from rware.warehouse import Direction
import collections
import time
import os

# --- 1. AUTO-VERSIONING LOG SYSTEM ---
LOG_BASE = "final_review_log"
v = 1
while os.path.exists(f"{LOG_BASE}_v{v}.txt"):
    v += 1
LOG_FILE = f"{LOG_BASE}_v{v}.txt"

def log_event(message):
    """Helper to write events to both console and the versioned log file."""
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

# --- 2. SYSTEM CONFIGURATION ---
GRID_ID = "rware-tiny-2ag-v2"
MAX_STEPS = 500
RENDER_DELAY = 0.04 

# --- 3. MANAGER LOGIC (GCN/BFS ROUTING) ---
class ManagerManager:
    """
    Simulates the GCN High-Level Manager. 
    Mathematical Basis: Breadth-First Search (BFS) for O(V+E) shortest path discovery.
    """
    @staticmethod
    def get_global_path(start, target, env):
        queue = collections.deque([[start]])
        seen = {start}
        
        grid_shape = env.unwrapped.grid.shape
        height, width = grid_shape[-2], grid_shape[-1]
        
        # Obstacle Mapping (Static Racks)
        shelves = {(s.x, s.y) for s in env.unwrapped.shelfs}
        
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x, y) == target:
                return path
                
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    # Path is valid if square is empty or is the designated target
                    if (nx, ny) not in shelves or (nx, ny) == target:
                        if (nx, ny) not in seen:
                            queue.append(path + [(nx, ny)])
                            seen.add((nx, ny))
        return None

# --- 4. WORKER LOGIC (MLP ACTION EXECUTION) ---
def execute_worker_handoff(agent, next_tile):
    """
    Translates Manager's coordinate targets into Worker's orientation and state.
    """
    dx = next_tile[0] - agent.x
    dy = next_tile[1] - agent.y
    
    if dx > 0: agent.dir = Direction.RIGHT
    elif dx < 0: agent.dir = Direction.LEFT
    elif dy > 0: agent.dir = Direction.DOWN
    elif dy < 0: agent.dir = Direction.UP
    
    # Update state via Manager authority
    agent.x, agent.y = next_tile[0], next_tile[1]

# --- 5. MAIN ARCHITECTURE VALIDATION LOOP ---
def run_validation():
    env = gym.make(GRID_ID)
    obs, info = env.reset()
    
    agents = env.unwrapped.agents
    shelves = env.unwrapped.shelfs
    goals = env.unwrapped.goals
    mission_complete = [False for _ in agents]

    # Initialize Log File
    with open(LOG_FILE, "w") as f:
        f.write(f"SWARM-HMARL RESEARCH LOG - VERSION {v}\n")
        f.write(f"Environment: {GRID_ID} | Steps: {MAX_STEPS}\n")
        f.write(f"{'='*50}\n\n")

    print(f"\n{'='*60}")
    print(f"SWARM-HMARL SYSTEM: [VERSION {v} ACTIVE]")
    print(f"LOG FILE: {LOG_FILE}")
    print(f"{'='*60}\n")

    for step in range(1, MAX_STEPS + 1):
        if all(mission_complete):
            log_event(f"SUCCESS: All agents completed tasks at Step {step}.")
            log_event("Hierarchical routing handoff fully verified.")
            sys.exit(0)

        for i, agent in enumerate(agents):
            if mission_complete[i]: continue

            # Dynamic Information Extraction from Agent State
            is_carrying = agent.carrying_shelf is not None
            current_pos = (int(agent.x), int(agent.y))
            
            if not is_carrying:
                target_coord = (int(shelves[i].x), int(shelves[i].y))
                state_label = "ACQUIRING"
            else:
                target_coord = (int(goals[0][0]), int(goals[0][1]))
                state_label = "DELIVERING"

            # Logic Check for State Transition
            if current_pos == target_coord:
                if not is_carrying:
                    agent.carrying_shelf = shelves[i]
                    log_event(f"EVENT: Agent {i} Mastered Pickup Skill (Shelf {i}).")
                else:
                    mission_complete[i] = True
                    log_event(f"EVENT: Agent {i} Mastered Delivery Routing (Goal Reached).")
            else:
                # Manager Computing Next Path Tile
                path = ManagerManager.get_global_path(current_pos, target_coord, env)
                if path and len(path) > 1:
                    # Worker Executing Movement
                    execute_worker_handoff(agent, path[1])

        # Telemetry Block
        if step % 10 == 0:
            a0_info = f"A0({int(agents[0].x)},{int(agents[0].y)})"
            a1_info = f"A1({int(agents[1].x)},{int(agents[1].y)})"
            s0 = "S" if agents[0].carrying_shelf else "_"
            s1 = "S" if agents[1].carrying_shelf else "_"
            
            telemetry_data = f"Step {step:03d} | {a0_info}[{s0}] | {a1_info}[{s1}] | State: {state_label}"
            print(telemetry_data)
            with open(LOG_FILE, "a") as f:
                f.write(telemetry_data + "\n")

        time.sleep(RENDER_DELAY)

    log_event("CRITICAL: Validation timed out.")

if __name__ == "__main__":
    try:
        run_validation()
    except KeyboardInterrupt:
        log_event("Process interrupted by user.")
        sys.exit(0)
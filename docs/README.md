## SWARM: Hierarchical Multi-Agent Reinforcement Learning for Autonomous Warehousing

SWARM is a research-focused implementation of a **Hierarchical Multi-Agent Reinforcement Learning (H-MARL)** system designed to optimize throughput and safety in large-scale robotic warehouses. 

### Key Technical Features:
* **Hierarchical Architecture:** Decouples high-level task allocation (**GCN-based Manager**) from low-level navigation (**PPO-based MLP Workers**).
* **Guaranteed Safety:** Integrated **Control Barrier Functions (CBF)** layer to ensure zero-collision navigation even in high-congestion scenarios.
* **Industrial Benchmarking:** Comparative analysis against industry standards including **A* Pathfinding** and the **Hungarian Algorithm**.
* **Scalable Simulation:** Built on the `rware` (Robotic Warehouse) Gymnasium environment, optimized for 100+ agent coordination.

### Target Metrics:
- **Throughput:** >5,200 packages/hour (targeting a 15% improvement over rule-based baselines).
- **Safety:** <2% collision rate using formal safety constraints.

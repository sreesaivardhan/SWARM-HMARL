
import matplotlib.pyplot as plt

# The exact data extracted from your 1000-episode deep training run
episodes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
rewards = [53.20, 90.00, 82.90, 77.40, 94.80, 100.00, 100.00, 67.50, 99.10, 89.20, 95.90, 100.00, 100.00, 88.10, 98.70, 88.90, 100.00, 100.00, 97.20, 97.80]

# Set up the plot style
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, marker='o', linestyle='-', color='b', linewidth=2, markersize=6, label='Worker Reward')

# Add titles and labels
plt.title('Worker MLP Training Performance (Phase 1)', fontsize=14, fontweight='bold')
plt.xlabel('Training Episodes', fontsize=12)
plt.ylabel('Average Reward (Shelf Acquisition)', fontsize=12)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)

# Highlight the convergence zone
plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Optimal Phase 1 Reward')

# Save the plot as an image file
plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
print("Graph generated! Check your SWARM folder for 'training_curve.png'.")
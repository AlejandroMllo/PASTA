# demo_pasta.py
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pasta import PASTA

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Standalone 3-Objective MORL Environment
# ==========================================
class MORLEnv:
    """
    A minimal 2D continuous navigation environment with 3 competing objectives:
        1. Goal Seeking: Reach the origin (0,0) (Minimize distance).
        2. Energy Efficiency: Minimize control effort (action magnitude).
        3. Smoothness: Minimize sudden changes in action (action delta).
    """
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 2
        self.max_steps = 64
        self.state = np.zeros(self.state_dim)
        self.prev_action = np.zeros(self.action_dim)
        self.step_count = 0

    def reset(self):
        # Start at a random position away from the origin
        self.state = np.random.uniform(-5.0, 5.0, size=(self.state_dim,))
        self.prev_action = np.zeros(self.action_dim)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        # Scale and clip action to simulate physical limits
        action = np.clip(2.0 * action - 1.0, -1.0, 1.0)  
        self.state += action
        self.step_count += 1

        # Objective 1: Goal Seeking (Negative distance to origin)
        dist_to_goal = np.linalg.norm(self.state)
        reward_goal = -dist_to_goal

        # Objective 2: Energy Efficiency (Negative action magnitude)
        reward_energy = -np.linalg.norm(action)

        # Objective 3: Smoothness (Negative change from previous action)
        reward_smoothness = -np.linalg.norm(action - self.prev_action)
        
        self.prev_action = action.copy()
        rewards = np.array([reward_goal, reward_energy, reward_smoothness], dtype=np.float32)
        
        terminated = dist_to_goal < 0.1
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        return self.state.copy(), rewards, done, {}

# ==========================================
# 2. Visualization Helpers
# ==========================================
def save_performance_plot(history, filename="pasta_learning_curve.png"):
    print(f"\n📊 Saving performance plot to {filename}...")
    steps = history['steps']
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot Returns on primary Y axis
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Average Return', color='black')
    ax1.plot(steps, history['goal'], label='Goal Return', color='tab:blue', linewidth=2)
    ax1.plot(steps, history['energy'], label='Energy Return', color='tab:orange', linewidth=2)
    ax1.plot(steps, history['smooth'], label='Smoothness Return', color='tab:green', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Smoothness parameter (mu) on secondary Y axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Smoothness parameter (μ)', color='tab:red')
    ax2.plot(steps, history['mu'], label='μ (Adaptive)', color='tab:red', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Combine legends cleanly
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
    
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print("✅ Plot saved successfully.")

def save_eval_video(agent, env, filename="pasta_eval.gif"):
    print(f"🎥 Recording evaluation episode to {filename}...")
    obs, _ = env.reset()
    states = [obs.copy()]
    done = False
    
    # Run one deterministic episode
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        states.append(obs.copy())
        
    states = np.array(states)
    
    # Set up the animation plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_title("PASTA Agent: Evaluation Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Draw origin (goal)
    ax.plot(0, 0, 'rx', markersize=10, markeredgewidth=2, label="Goal (0,0)")
    
    line, = ax.plot([], [], 'b-', linewidth=2, label="Trajectory")
    point, = ax.plot([], [], 'bo', markersize=8, label="Agent")
    ax.legend(loc="upper right")
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
        
    def update(frame):
        line.set_data(states[:frame+1, 0], states[:frame+1, 1])
        point.set_data([states[frame, 0]], [states[frame, 1]])
        return line, point
        
    ani = animation.FuncAnimation(fig, update, frames=len(states),
                                  init_func=init, blit=True, interval=100)
    
    # Save using Pillow (built-in with standard matplotlib installations)
    ani.save(filename, writer='pillow', fps=10)
    plt.close(fig)
    print("✅ Video saved successfully.")

# ==========================================
# 3. Training Demonstration
# ==========================================
def main():
    print("🍝 Starting PASTA 3-Objective Demonstration...")

    set_seed(42)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_OBJECTIVES = 3
    STEPS_PER_ROLLOUT = 4096
    TOTAL_TIMESTEPS = STEPS_PER_ROLLOUT * 25
    
    env = MORLEnv()
    preferences = np.array([0.6, 0.2, 0.2], dtype=np.float32)
    print(f"🎯 Target Preferences: {preferences}")

    agent = PASTA(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_objectives=N_OBJECTIVES,
        n_steps=STEPS_PER_ROLLOUT,
        batch_size=32,
        lr=2e-3, 
        device=DEVICE,
        total_train_steps=TOTAL_TIMESTEPS
    )
    agent.set_preference(preferences)

    obs, _ = env.reset()
    global_step = 0

    # Dictionaries to hold data for plotting
    history = {'steps': [], 'goal': [], 'energy': [], 'smooth': [], 'mu': []}

    print("\n🚀 Training Started...")
    while global_step < TOTAL_TIMESTEPS:
        
        agent.buffer.reset()
        ep_rewards_accum = np.zeros(N_OBJECTIVES, dtype=np.float32)
        episodes_completed = 0
        
        # --- Rollout Phase ---
        for _ in range(STEPS_PER_ROLLOUT):
            action, log_prob, val = agent.select_action(obs)
            next_obs, rewards, done, _ = env.step(action)
            
            agent.buffer.store(obs, action, log_prob, rewards, val, float(done))
            ep_rewards_accum += rewards
            
            obs = next_obs
            global_step += 1
            
            if done:
                obs, _ = env.reset()
                episodes_completed += 1

        # --- Update Phase ---
        _, _, next_val = agent.select_action(obs)
        agent.buffer.compute_advantages_and_returns(next_val, float(done))
        metrics = agent.update_parameters()
        
        # --- Logging & Data Collection ---
        avg_ep_reward = ep_rewards_accum / max(1, episodes_completed)
        mu_val = metrics.get('mu', 0.0)
        
        history['steps'].append(global_step)
        history['goal'].append(avg_ep_reward[0])
        history['energy'].append(avg_ep_reward[1])
        history['smooth'].append(avg_ep_reward[2])
        history['mu'].append(mu_val)
        
        print(f"Step: {global_step:04d}/{TOTAL_TIMESTEPS} | "
              f"Returns [Goal, Energy, Smooth]: [{avg_ep_reward[0]:.1f}, {avg_ep_reward[1]:.1f}, {avg_ep_reward[2]:.1f}] | "
              f"μ: {mu_val:.3f}")

    print("\n✅ Training Complete. Generating visualizations...")
    os.makedirs("results", exist_ok=True)

    save_performance_plot(history, filename="results/pasta_learning_curve.png")
    save_eval_video(agent, env, filename="results/pasta_eval.gif")

    # ==========================================
    # 4. Save and Load Demonstration
    # ==========================================
    print("\n💾 Demonstrating Model Save & Load...")
    save_path = "results/pasta_model.pt"
    
    # Save the entire agent object
    torch.save(agent, save_path)
    print(f"✅ Model saved to {save_path}")

    # Load the agent from disk
    print("🔄 Loading the model...")
    loaded_agent = torch.load(save_path, weights_only=False)
    loaded_agent.device = DEVICE  # Ensure device mapping is maintained
    
    # Verify the loaded model works
    save_eval_video(loaded_agent, env, filename="results/pasta_loaded_eval.gif")
    print("🎉 Demo complete! Check your folder for the plots, GIFs, and saved model.")

if __name__ == "__main__":
    main()

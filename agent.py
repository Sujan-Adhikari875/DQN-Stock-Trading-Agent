import gymnasium as gym
import gym_anytrading
import torch
import torch.nn as nn
import yaml
import random
import torch.optim as optim
import argparse
import os
import numpy as np
from experience_replay import ReplayMemory
from dqn import DQN
from itertools import count

# Device Configuration
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        try:
            with open("parameters.yaml", "r") as f:
                all_params = yaml.safe_load(f)
                param = all_params[param_set]
        except FileNotFoundError:
            raise FileNotFoundError("parameters.yaml not found. Please create it with your hyperparameters.")
        except KeyError:
            raise KeyError(f"Parameter set '{param_set}' not found in parameters.yaml")

        self.epsilon_init = param["epsilon_init"]
        self.epsilon_min = param["epsilon_min"]
        self.epsilon_decay = param["epsilon_decay"]
        self.alpha = param["alpha"]
        self.gamma = param["gamma"]
        self.reward_threshold = param.get("reward_threshold", float("inf"))
        self.replay_memory_size = param["replay_memory_size"]
        self.mini_batch_size = param["mini_batch_size"]
        self.network_sync_rate = param["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOGFILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def runs(self, is_training=True, render=False):
        # Initialize Environment
        # Note: Ensure 'stocks-v0' is registered in gym_anytrading. 
        # If you get an error, check if you need to register it manually.
        try:
            env = gym.make("stocks-v0",
                           frame_bound=(50, 500),
                           window_size=10,
                           render_mode="human" if render else None)
        except gym.error.NameNotFound:
            print("Error: 'stocks-v0' not found. Make sure gym_anytrading is installed and registered.")
            return

        if render:
            env.unwrapped.metadata["render_fps"] = 10

        num_states = env.observation_space.shape[0] * env.observation_space.shape[1]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()
            
            steps = 0
            best_rewards = float("-inf")
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
        else:
            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(f"Model file {self.MODEL_FILE} not found for testing.")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device, weights_only=True))
            policy_dqn.eval()

        for episode in count():
            state, info = env.reset()
            
            # Handle potential tuple return from older gym versions or wrappers
            if isinstance(state, tuple):
                state = state[0]
            
            # Flatten and convert to tensor
            state_np = np.asarray(state, dtype=np.float32).flatten()
            state = torch.as_tensor(state_np, device=device)

            done = False
            total_reward = 0

            while not done:
                # Epsilon-greedy strategy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # Add batch dimension for inference
                        state_batch = state.unsqueeze(0)
                        q_values = policy_dqn(state_batch)
                        action = q_values.argmax().item()

                # Step environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Handle next_state shape
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                
                next_state_np = np.asarray(next_state, dtype=np.float32).flatten()
                next_state_tensor = torch.as_tensor(next_state_np, device=device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
                done_tensor = torch.tensor(done, dtype=torch.bool, device=device)

                if is_training:
                    # Store raw numpy arrays or simple tensors to avoid device transfer overhead during append
                    # Storing as numpy is generally safer for CPU memory usage
                    memory.append(state_np, action, next_state_np, reward, done)
                    
                    steps += 1

                    # Train if enough samples
                    if len(memory) >= self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Sync target network
                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

                state = next_state_tensor
                total_reward += float(reward)

            print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

            # Save best model
            if is_training and total_reward > best_rewards:
                best_rewards = total_reward
                log_msg = f"Best Reward: {best_rewards:.2f} at Episode: {episode + 1}"
                
                with open(self.LOGFILE, "a") as f:
                    f.write(log_msg + "\n")
                
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                print(f"Model saved with reward: {best_rewards:.2f}")

            # Decay epsilon
            if is_training:
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            
            # Optional: Stop if threshold reached
            if is_training and total_reward >= self.reward_threshold:
                print(f"Threshold {self.reward_threshold} reached. Stopping.")
                break

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unpack batch
        states_np, actions, next_states_np, rewards, dones = zip(*mini_batch)

        # Convert to tensors on the correct device
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=device)

        # Current Q values: Q(s, a)
        # policy_dqn expects shape (batch, state_dim)
        current_q_values = policy_dqn(states)
        # Gather the Q value for the action taken
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values: r + gamma * max Q(s', a')
        with torch.no_grad():
            next_q_values = target_dqn(next_states)
            next_q = next_q_values.max(dim=1).values
            # Double DQN style: only add future value if episode didn't end
            target_q = rewards + self.gamma * next_q * (~dones).float()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test DQN agent for stock trading.')
    parser.add_argument('hyperparameter', type=str, help='Key name from parameters.yaml')
    parser.add_argument('--train', action='store_true', help='Enable training mode (default is test if flag missing)')
    
    args = parser.parse_args()
    
    agent = Agent(param_set=args.hyperparameter)
    
    if args.train:
        agent.runs(is_training=True)
    else:
        agent.runs(is_training=False, render=True)
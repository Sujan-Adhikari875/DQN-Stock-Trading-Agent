import gymnasium as gym
import gym_anytrading
import torch
import torch.nn as nn
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch.optim as optim
import argparse
import os
import numpy as np

if torch.backends.mps.is_available():
    device="mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device="cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent():
    def __init__(self, param_set):
        self.param_set = param_set

        with open ("parameters.yaml", "r") as f:
            all_params = yaml.safe_load(f)
            param = all_params[param_set]
        self.epsilon_init = param["epsilon_init"]
        self.epsilon_min = param["epsilon_min"]
        self.epsilon_decay = param["epsilon_decay"]
        self.alpha = param["alpha"]
        self.gamma = param["gamma"]
        self.reward_threshold = param["reward_threshold"]
        self.replay_memory_size = param["replay_memory_size"]
        self.mini_batch_size = param["mini_batch_size"]
        self.network_sync_rate = param["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOGFILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def runs(self, is_training=True, render=False):
        env = gym.make("stocks-v0",
                    frame_bound=(50, 500),
                    window_size=10, render_mode="human" if render else None)
        if render:
            env.unwrapped.metadata["render_fps"] = 5
        
        num_states =  env.observation_space.shape[0] * env.observation_space.shape[1]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)

            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            best_rewards = float("-inf")

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, info = env.reset()

# Handle nested observation returned by some wrapper/version
            if isinstance(state, tuple):
                state = state[0]

            state = torch.as_tensor(np.asarray(state, dtype=np.float32)).flatten().to(device)  

            done = False 
            total_reward = 0

            while (not done and total_reward < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, truncated, info = env.step(action.item())

                done = terminated or truncated
                if isinstance(next_state, tuple):
                    next_state = next_state[0]

                next_state_tensor = torch.as_tensor(np.asarray(next_state, dtype=np.float32)).flatten().to(device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    memory.append((state, action, next_state_tensor, reward_tensor, done))

                    steps += 1

                    # Network Sync 

                    if  len(memory) >= self.mini_batch_size:
                        # get sample
                        mini_batch =memory.sample(self.mini_batch_size)

                        self.optimize(mini_batch, policy_dqn, target_dqn)

                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

                state = next_state_tensor

                total_reward += float(reward)

            print(f"for episode={episode+1} with total_reward ={total_reward}")

            if is_training and total_reward > best_rewards:
                log_msg = f"best rewards = {total_reward} and episodes = {episode + 1}"

                with open(self.LOGFILE, "a") as f:
                    f.write(log_msg + "\n")

                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                best_rewards = total_reward

            if is_training:
                    epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        # env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, done = zip(*mini_batch)

        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)

        actions = torch.tensor([action.item() if torch.is_tensor(action) else action for action in actions],
                               dtype=torch.long,device=device)

        rewards = torch.stack(rewards).to(device)

        done = torch.tensor(done, dtype=torch.bool, device=device)

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = target_dqn(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (~done).float()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
          

if __name__ == "__main__":
    parser =argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameter', help=' ')
    parser.add_argument('--train', help='Training mode', action='store_true')

    args = parser.parse_args()
    dql = Agent(param_set=args.hyperparameter)
    if args.train:
        dql.runs(is_training=True)

    else:
        dql.runs(is_training=False, render=True)
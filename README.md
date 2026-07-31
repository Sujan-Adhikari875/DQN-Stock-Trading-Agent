# DQN Stock Trading Agent

This project implements a **Deep Q-Network (DQN)** agent for stock trading using **PyTorch**, **Gymnasium**, and **Gym-AnyTrading**.

The agent interacts with a stock-trading environment and learns whether to perform a **Buy** or **Sell** action. It improves its decisions through rewards received from the environment.

The project uses important DQN techniques such as:

* Experience Replay
* Target Network
* Epsilon-Greedy Exploration
* Bellman Equation
* Model Checkpointing

## Project Structure

```text
├── agent.py
├── dqn.py
├── experience_replay.py
├── parameters.yaml
├── runs/
│   ├── stocks1.pt
│   └── stocks1.log
└── README.md
```

* `agent.py` contains the training and testing process.
* `dqn.py` contains the neural-network model.
* `experience_replay.py` stores previous experiences.
* `parameters.yaml` contains the training hyperparameters.
* `runs/` stores trained models and log files.

## Installation

Install the required libraries:

```bash
pip install torch gymnasium gym-anytrading numpy pyyaml
```

## Stock Environment

The stock environment is created using:

```python
env = gym.make(
    "stocks-v0",
    frame_bound=(50, 500),
    window_size=10
)
```

`frame_bound` defines the section of stock data used by the environment.

`window_size=10` means the agent observes information from the previous 10 time steps before selecting an action.

The environment normally provides two actions:

```text
0 = Sell
1 = Buy
```

## State Preparation

The observation returned by the environment is usually a two-dimensional array. It is flattened before being passed into the neural network:

```python
state = torch.as_tensor(
    np.asarray(state, dtype=np.float32)
).flatten().to(device)
```

Flattening converts the observation into a one-dimensional input vector.

The number of input states is calculated using:

```python
num_states = (
    env.observation_space.shape[0]
    * env.observation_space.shape[1]
)
```

## Device Selection

The code automatically selects the best available device:

```python
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

* `mps` is used for supported Apple devices.
* `cuda` is used for NVIDIA GPUs.
* `cpu` is used when GPU acceleration is unavailable.

## Policy Network

The policy network predicts a Q-value for every possible action:

```python
policy_dqn = DQN(
    num_states,
    num_actions
).to(device)
```

For example, the network may produce:

```text
[Q-value for Sell, Q-value for Buy]
```

The action with the highest Q-value is selected:

```python
with torch.no_grad():
    action = policy_dqn(
        state.unsqueeze(0)
    ).squeeze().argmax()
```

`unsqueeze(0)` adds the batch dimension required by the neural network.

## Epsilon-Greedy Exploration

During training, the agent must balance exploration and exploitation.

```python
if is_training and random.random() < epsilon:
    action = env.action_space.sample()
else:
    with torch.no_grad():
        action = policy_dqn(
            state.unsqueeze(0)
        ).squeeze().argmax()
```

* Exploration selects a random action.
* Exploitation selects the action with the highest predicted Q-value.

At the beginning, epsilon is high, so the agent explores more. After every episode, epsilon decreases:

```python
epsilon = max(
    self.epsilon_min,
    epsilon * self.epsilon_decay
)
```

This allows the agent to gradually use more of its learned knowledge.

## Experience Replay

Each interaction is stored as an experience:

```python
memory.append(
    (
        state,
        action,
        next_state_tensor,
        reward_tensor,
        done
    )
)
```

Each experience contains:

```text
Current state
Selected action
Next state
Reward
Episode completion status
```

When enough experiences are available, the agent randomly samples a mini-batch:

```python
mini_batch = memory.sample(
    self.mini_batch_size
)
```

Random sampling reduces the correlation between consecutive stock observations and makes training more stable.

## Target Network

DQN uses two neural networks:

* Policy network
* Target network

The target network starts with the same weights as the policy network:

```python
target_dqn.load_state_dict(
    policy_dqn.state_dict()
)
```

After a fixed number of steps, the target network is updated:

```python
if steps >= self.network_sync_rate:
    target_dqn.load_state_dict(
        policy_dqn.state_dict()
    )
    steps = 0
```

The target network provides more stable target Q-values because it is not updated after every training step.

## Q-Value Calculation

The Q-values for the selected actions are obtained using:

```python
current_q = policy_dqn(states).gather(
    1,
    actions.unsqueeze(1)
).squeeze(1)
```

The target Q-value is calculated using the Bellman equation:

```python
with torch.no_grad():
    next_q = target_dqn(
        next_states
    ).max(dim=1).values

    target_q = rewards + (
        self.gamma
        * next_q
        * (~dones).float()
    )
```

The main equation is:

```text
Target Q = Reward + Gamma × Maximum Next Q
```

When the episode is finished, the future reward is removed using:

```python
(~dones).float()
```

## Network Optimization

The difference between the predicted Q-value and target Q-value is calculated using Mean Squared Error:

```python
loss = self.loss_fn(
    current_q,
    target_q
)
```

The policy network is updated using backpropagation:

```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

The Adam optimizer is created using:

```python
self.optimizer = optim.Adam(
    policy_dqn.parameters(),
    lr=self.alpha
)
```

## Model Saving

The model is saved whenever the agent achieves a better total reward:

```python
if total_reward > best_rewards:
    torch.save(
        policy_dqn.state_dict(),
        self.MODEL_FILE
    )

    best_rewards = total_reward
```

The saved model is stored inside the `runs/` directory.

The training result is also written to a log file:

```python
with open(self.LOGFILE, "a") as f:
    f.write(log_msg + "\n")
```

## Training

Run the following command:

```bash
python agent.py stocks1 --train
```

`stocks1` must match a configuration name inside `parameters.yaml`.

Example:

```yaml
stocks1:
  epsilon_init: 1.0
  epsilon_min: 0.05
  epsilon_decay: 0.995
  alpha: 0.001
  gamma: 0.99
  reward_threshold: 100000
  replay_memory_size: 10000
  mini_batch_size: 64
  network_sync_rate: 1000
```

## Testing

Run the trained model using:

```bash
python agent.py stocks1
```

During testing, the saved model is loaded:

```python
policy_dqn.load_state_dict(
    torch.load(
        self.MODEL_FILE,
        map_location=device
    )
)

policy_dqn.eval()
```

Using `map_location=device` is recommended because the model may have been trained on a different device.

## Training Flow

The complete learning process is:

1. Reset the stock environment.
2. Convert the observation into a tensor.
3. Select a random or predicted action.
4. Execute the action in the environment.
5. Receive the next state and reward.
6. Store the experience in replay memory.
7. Sample a mini-batch from memory.
8. Calculate current and target Q-values.
9. Update the policy network.
10. Periodically synchronize the target network.
11. Save the model when the reward improves.

## Important Note

The current training loop uses:

```python
for episode in itertools.count():
```

This creates an unlimited number of episodes. Training continues until the program is manually stopped.

For a fixed number of episodes, it can be replaced with:

```python
for episode in range(500):
```

This will stop training after 500 episodes.

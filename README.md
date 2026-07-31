# DQN Stock Trading Agent

This project implements a **Deep Q-Network (DQN)** agent for stock trading using **PyTorch**, **Gymnasium**, and **Gym-AnyTrading**.

The agent interacts with the `stocks-v0` environment and learns whether to perform a **Buy** or **Sell** action. It improves its decisions by receiving rewards from the environment and training on previously stored experiences.

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
│   ├── stocks-v0.pt
│   └── stocks-v0.log
└── README.md
```

* `agent.py` contains the training, testing, and optimization process.
* `dqn.py` contains the neural-network architecture.
* `experience_replay.py` contains the replay-memory implementation.
* `parameters.yaml` contains the DQN hyperparameters.
* `runs/` stores trained models and training logs.

## Installation

Install the required libraries:

```bash
pip install torch gymnasium gym-anytrading numpy pyyaml pygame
```

## Stock Trading Environment

The environment is created using:

```python
env = gym.make(
    "stocks-v0",
    frame_bound=(50, 500),
    window_size=10,
    render_mode="human" if render else None
)
```

### Environment settings

* `stocks-v0` is the stock-trading environment.
* `frame_bound=(50, 500)` defines the section of stock data used.
* `window_size=10` means the agent observes the previous 10 time steps.
* `render_mode="human"` displays the trading environment during testing.

When rendering is enabled, the frame rate is set using:

```python
env.unwrapped.metadata["render_fps"] = 3
```

The environment provides two possible actions:

```text
0 = Sell
1 = Buy
```

## State Preparation

The observation returned by the environment is converted into a NumPy array, transformed into a PyTorch tensor, and flattened:

```python
state = torch.as_tensor(
    np.asarray(state, dtype=np.float32)
).flatten().to(device)
```

Flattening converts the two-dimensional observation into a one-dimensional input vector for the neural network.

The number of input states is calculated using:

```python
num_states = (
    env.observation_space.shape[0]
    * env.observation_space.shape[1]
)
```

The code also handles cases where an environment wrapper returns a nested observation:

```python
if isinstance(state, tuple):
    state = state[0]
```

The same check is applied to `next_state`.

## Device Selection

The program automatically selects the best available device:

```python
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

* `mps` is used on supported Apple devices.
* `cuda` is used with supported NVIDIA GPUs.
* `cpu` is used when GPU acceleration is unavailable.

## Policy Network

The policy network predicts a Q-value for each possible action:

```python
policy_dqn = DQN(
    num_states,
    num_actions
).to(device)
```

The network output represents:

```text
[Q-value for Sell, Q-value for Buy]
```

During exploitation, the action with the highest Q-value is selected:

```python
with torch.no_grad():
    action = policy_dqn(
        state.unsqueeze(dim=0)
    ).squeeze().argmax()
```

`unsqueeze(dim=0)` adds the batch dimension required by the neural network.

## Epsilon-Greedy Exploration

During training, the agent balances exploration and exploitation.

```python
if is_training and random.random() < epsilon:
    action = env.action_space.sample()
    action = torch.tensor(
        action,
        dtype=torch.long,
        device=device
    )
else:
    with torch.no_grad():
        action = policy_dqn(
            state.unsqueeze(dim=0)
        ).squeeze().argmax()
```

* **Exploration:** selects a random action.
* **Exploitation:** selects the action with the highest predicted Q-value.

After every episode, epsilon is reduced:

```python
epsilon = max(
    self.epsilon_min,
    epsilon * self.epsilon_decay
)
```

This allows the agent to explore more at the beginning and gradually depend on its learned policy.

## Experience Replay

Each interaction is stored in replay memory:

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

Once the replay memory contains enough experiences, a random mini-batch is sampled:

```python
if len(memory) >= self.mini_batch_size:
    mini_batch = memory.sample(
        self.mini_batch_size
    )

    self.optimize(
        mini_batch,
        policy_dqn,
        target_dqn
    )
```

Random sampling reduces correlation between consecutive stock observations and improves training stability.

## Target Network

DQN uses two neural networks:

* Policy network
* Target network

The target network is initialized with the policy-network weights:

```python
target_dqn.load_state_dict(
    policy_dqn.state_dict()
)
```

After a fixed number of environment steps, the target network is synchronized:

```python
if steps >= self.network_sync_rate:
    target_dqn.load_state_dict(
        policy_dqn.state_dict()
    )
    steps = 0
```

The target network helps reduce instability because its parameters are not updated during every optimization step.

## Q-Value Calculation

The predicted Q-values for the selected actions are calculated using:

```python
current_q = policy_dqn(states).gather(
    1,
    actions.unsqueeze(1)
).squeeze(1)
```

The target Q-values are calculated using the target network:

```python
with torch.no_grad():
    next_q = target_dqn(
        next_states
    ).max(dim=1).values

    target_q = rewards + (
        self.gamma
        * next_q
        * (~done).float()
    )
```

The Bellman equation used is:

```text
Target Q = Reward + Gamma × Maximum Next Q
```

For terminal states, the future Q-value is removed using:

```python
(~dones).float()
```

## Network Optimization

The difference between the current Q-value and target Q-value is calculated using Mean Squared Error:

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

The Adam optimizer is initialized with:

```python
self.optimizer = optim.Adam(
    policy_dqn.parameters(),
    lr=self.alpha
)
```

## Model Saving

The model is saved whenever the agent achieves a better episode reward:

```python
if is_training and total_reward > best_rewards:
    torch.save(
        policy_dqn.state_dict(),
        self.MODEL_FILE
    )

    best_rewards = total_reward
```

The saved model is stored inside the `runs/` directory.

Training information is also written to a log file:

```python
with open(self.LOGFILE, "a") as f:
    f.write(log_msg + "\n")
```

For the `stocks-v0` parameter set, the output files are:

```text
runs/stocks-v0.pt
runs/stocks-v0.log
```

## Hyperparameters

The training configuration is loaded from `parameters.yaml`.

Example:

```yaml
stocks-v0:
  epsilon_init: 1.0
  epsilon_min: 0.05
  epsilon_decay: 0.9995
  alpha: 0.001
  gamma: 0.99
  reward_threshold: 1000
  replay_memory_size: 10000
  mini_batch_size: 32
  network_sync_rate: 10
```

### Hyperparameter descriptions

* `epsilon_init` – Initial exploration rate.
* `epsilon_min` – Minimum exploration rate.
* `epsilon_decay` – Controls how quickly exploration decreases.
* `alpha` – Learning rate for the Adam optimizer.
* `gamma` – Discount factor for future rewards.
* `reward_threshold` – Stops the current episode loop if the total reward reaches this value.
* `replay_memory_size` – Maximum number of experiences stored.
* `mini_batch_size` – Number of experiences used in one optimization step.
* `network_sync_rate` – Number of steps before updating the target network.

The `env_id` parameter is not required because the current code directly creates the `"stocks-v0"` environment.

## Training

Train the agent using:

```bash
python agent.py stocks-v0 --train
```

The value `stocks-v0` is the hyperparameter configuration name inside `parameters.yaml`.

During training, the program:

1. Creates the policy and target networks.
2. Initializes replay memory.
3. Selects actions using epsilon-greedy exploration.
4. Stores experiences in replay memory.
5. Samples mini-batches.
6. Optimizes the policy network.
7. Synchronizes the target network.
8. Saves the best-performing model.

## Testing

Run the trained model using:

```bash
python agent.py stocks-v0
```

During testing, the saved model is loaded:

```python
policy_dqn.load_state_dict(
    torch.load(self.MODEL_FILE)
)

policy_dqn.eval()
```

Testing automatically enables rendering:

```python
dql.runs(
    is_training=False,
    render=True
)
```

For better compatibility between CPU and GPU devices, the model-loading line can be changed to:

```python
policy_dqn.load_state_dict(
    torch.load(
        self.MODEL_FILE,
        map_location=device
    )
)
```

## Training Flow

The complete DQN training process is:

1. Reset the stock environment.
2. Convert and flatten the observation.
3. Select a random or predicted action.
4. Execute the action in the environment.
5. Receive the next state and reward.
6. Check whether the episode has terminated or been truncated.
7. Store the transition in replay memory.
8. Sample a mini-batch.
9. Calculate the current and target Q-values.
10. Update the policy network.
11. Periodically synchronize the target network.
12. Save the model when the reward improves.
13. Reduce epsilon after every episode.

## Important Notes

### Unlimited training episodes

The current training loop uses:

```python
for episode in itertools.count():
```

This creates an unlimited number of episodes. Training continues until the program is manually stopped.

To train for a fixed number of episodes, replace it with:

```python
for episode in range(500):
```

### Episode completion

An episode is considered complete when it is terminated or truncated:

```python
done = terminated or truncated
```

This follows the current Gymnasium API.

### Model file requirement

Testing requires an existing trained model:

```text
runs/stocks-v0.pt
```

Therefore, the training command must be run before the testing command.

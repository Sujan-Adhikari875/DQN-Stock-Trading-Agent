import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        # Simple Feed Forward Network
        self.network = nn.Sequential(
            nn.Linear(num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)
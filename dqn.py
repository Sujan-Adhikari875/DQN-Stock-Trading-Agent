import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=12, action_dim=3):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim)

        )

    def forward(self, x):
        return self.model(x)
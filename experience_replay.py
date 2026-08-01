from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        # Store as tuple of numpy/scalar types to save memory
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample indices
        batch = random.sample(self.memory, batch_size)
        return batch

    def __len__(self):
        return len(self.memory)
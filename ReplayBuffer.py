import numpy as np


class ReplayBuffer:
    def __init__(self, mem_size, batch_size, state_shape):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch = 0
        self.store_ptr = 0
        self.sample_ptr = 0
        self.rewards = np.zeros((mem_size), dtype=np.float32)
        self.actions = np.zeros_like(self.rewards)
        self.terminal = np.zeros_like(self.rewards)
        self.states = np.zeros((mem_size, *state_shape), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.buffer_size = 0

    # circular list of state transitons
    def store(self, state, action, reward, next_state, done):

        if self.store_ptr >= self.mem_size:
            self.store_ptr = 0

        self.rewards[self.store_ptr] = reward
        self.states[self.store_ptr] = state
        self.next_states[self.store_ptr] = next_state
        self.terminal[self.store_ptr] = done
        self.actions[self.store_ptr] = float(action)
        self.store_ptr += 1

        if self.buffer_size <= self.mem_size:
            self.buffer_size += 1

    # random sampling
    def sample(self):
        if self.batch_size >= self.buffer_size:
            return (None, None, None, None, None)
        else:
            indices = np.random.choice(
                self.buffer_size-1, self.batch_size, replace=False)

            states = self.states[indices, :]
            actions = self.actions[indices]
            rewards = self.rewards[indices]
            next_states = self.next_states[indices, :]
            dones = self.terminal[indices]

            return (states, actions, rewards, next_states, dones)

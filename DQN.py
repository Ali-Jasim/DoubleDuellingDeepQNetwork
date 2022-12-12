import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ReplayBuffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, input_shape, actions, hidden_layer):
        super().__init__()
        self.l1 = nn.Linear(*input_shape, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, hidden_layer)
        self.l3 = nn.Linear(hidden_layer, actions)

        self.device = "cuda" if T.cuda.is_available() else "cpu"

    # given a state, produce Q value (value of actions in a given state)
    def forward(self, state):
        state = T.tensor(state).to(self.device)
        p1 = F.relu(self.l1(state)).to(self.device)
        p2 = F.relu(self.l2(p1)).to(self.device)
        p3 = self.l3(p2).to(self.device)

        return p3.to(self.device)


class Agent():
    def __init__(self, lr, gamma, actions, input_shape, eps, hidden_layer, batch_size=256, buffer_size=500000):

        # action space
        self.actions = [i for i in range(actions)]

        # hyper parameters
        self.gamma = gamma  # discounting previous rewards
        self.eps = eps  # epsilon greedy, Explore-Expliot dilemma
        self.eps_min = 0.05  # minimum epsilon
        self.eps_dec = 0.99  # how we reduce epsilon, when we take greedy action
        self.batch_size = batch_size
        self.indices = np.arange(self.batch_size, dtype=np.int32)

        # initialize neural network, the brain of our agent
        # note we are learning the (Q)uality of taking an action in a given state
        self.Q = QNetwork(input_shape, actions, hidden_layer)
        # replay buffer to reduce correlation error
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, input_shape)

        # we compute loss of predicted q and actual q
        self.loss = nn.MSELoss().to(self.Q.device)

        # predicted q = our prediction of next state reward
        # actual q = actual reward for going to next state
        # use MSE loss because emphasizes large errors

        # use .to in pytorch to enable GPU accelerated training
        self.Q.to(self.Q.device)
        # nueral network back propogation
        self.optim = optim.SGD(self.Q.parameters(), lr)

    # we store transitions in the replay buffer then use batch normalization to reduce correlation error
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)

    # epsilon-greedy action choice, multi-armed bandit problem
    def choose_action(self, state):
        if np.random.random() < self.eps:
            return np.random.choice(self.actions)
        else:
            action = self.Q.forward(state).argmax()
            return self.actions[action]

    # sample the replay buffer and learn from experiences
    def learn(self):

        # take a random sample of transitions
        states, actions, rewards, next_states, terminal = self.replay_buffer.sample()

        if states is not None:

            rewards = T.tensor(rewards).to(
                self.Q.device)

            # feed state batch to NN, then map our predicted Q values to actions
            # this is our prediction
            q_prediction = self.Q.forward(states)[self.indices, actions]

            # feed next states batch to NN and get maximum expected return
            # for taking best action
            q_prediction_next = self.Q.forward(
                next_states)

            q_next, max_act = T.max(q_prediction_next, 1)

            # using the bellman equation to produce a loss
            q_target = rewards + (self.gamma *
                                  q_prediction_next[self.indices, max_act])

            q_target[terminal] = rewards[terminal]
            loss = self.loss(q_prediction, q_target)

            # neural network optimization
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # reduce epsilon
            self.eps = max(self.eps*self.eps_dec, self.eps_min)

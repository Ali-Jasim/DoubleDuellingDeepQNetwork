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
        self.V = nn.Linear(hidden_layer, 1)
        self.A = nn.Linear(hidden_layer, actions)

        self.device = "cuda" if T.cuda.is_available() else "cpu"

    # given a state, produce the value of being in that state
    # and the action quality values for that state
    def forward(self, state):
        state = T.tensor(state).to(self.device)
        p1 = F.relu(self.l1(state)).to(self.device)
        p2 = F.relu(self.l2(p1)).to(self.device)
        V = self.V(p2).to(self.device)
        A = self.A(p2).to(self.device)

        return V, A


class Agent():
    def __init__(self, lr, gamma, eps, actions, input_shape, hidden_layer,
                 batch_size=256, buffer_size=5000000, replace_thresh=100,
                 eps_min=0.05, eps_dec=0.99):

        # action space
        self.actions = [i for i in range(actions)]

        # hyper parameters
        self.gamma = gamma  # discounting previous rewards
        self.eps = eps  # epsilon greedy, Explore-Expliot dilemma
        self.eps_min = eps_min  # minimum epsilon
        self.eps_dec = eps_dec  # how we reduce epsilon, when we take greedy action
        # learn count
        self.learn_count = 0
        self.replace_thresh = replace_thresh

        # batch info
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.indices = np.arange(batch_size, dtype=np.int32)

        # initialize neural network, the brain of our agent
        # note we are learning the (Q)uality of taking an action in a given state
        self.Q = QNetwork(input_shape, actions, hidden_layer)
        self.Qt = QNetwork(input_shape, actions, hidden_layer)
        # replay buffer to reduce correlation error
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, input_shape)

        # we compute loss of predicted q and actual q
        self.loss = nn.MSELoss().to(self.Q.device)
        # predicted q = our prediction of next state reward
        # actual q = actual reward for going to next state
        # use MSE loss because emphasizes large errors

        # use .to in pytorch to enable GPU accelerated training
        self.Q.to(self.Q.device)
        self.Qt.to(self.Q.device)
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
            action = self.Q.forward(state)[1].argmax()
            return self.actions[action]

    # sample the replay buffer and learn from experiences
    def learn(self):

        # take a random sample of transitions
        states, actions, rewards, next_states, terminal = self.replay_buffer.sample()

        if states is not None:
            if self.learn_count % 100 == 0:
                self.Qt.load_state_dict(self.Q.state_dict())

            rewards = T.tensor(rewards).to(self.Q.device)

            # feed state batch to NN, then map our predicted Q values to actions
            # this is our prediction
            V, A = self.Q.forward(states)

            # feed next states batch to NN and get quality prediction for each state
            with T.no_grad():
                V_next, A_next = self.Qt.forward(next_states)

            q_prediction = V + (A-T.mean(A, 1, True))

            q_prediction = q_prediction[self.indices, actions]

            q_next = V_next + (A_next-T.mean(A_next, 1, True))

            max_next = T.argmax(q_next, 1)

            # using the bellman equation to produce a loss
            y = rewards + self.gamma*q_next[self.indices, max_next]
            y[terminal] = rewards[terminal]
            loss = self.loss(q_prediction, y)

            # neural network optimization
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # reduce epsilon
            self.eps = max(self.eps*self.eps_dec, self.eps_min)
            self.learn_count += 1

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
    def __init__(self, lr, gamma, actions, input_shape, eps, hidden_layer, batch_size=256, buffer_size=5000000, replace_thresh=100):

        # action space
        self.actions = [i for i in range(actions)]

        # hyper parameters
        self.gamma = gamma  # discounting previous rewards
        self.eps = eps  # epsilon greedy, Explore-Expliot dilemma
        self.eps_min = 0.05  # minimum epsilon
        self.eps_dec = 0.99  # how we reduce epsilon, when we take greedy action
        self.learn_count = 0  # every 100 learning steps, update the target network
        # affects neural network training step (how loss affects the weights)
        self.lr = lr
        # how big of a batch of transitions we use for training
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # used to map
        self.indices = np.arange(self.batch_size)
        self.replace_thresh = replace_thresh

        # initialize neural network, the brain of our agent
        # note we are learning the (Q)uality of taking an action in a given state
        self.Q = QNetwork(input_shape, actions, hidden_layer)
        self.Qt = QNetwork(input_shape, actions, hidden_layer)

        # replay buffer to reduce correlation error
        # this is the agents memory, at every step, we learn from batches of experiences
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, self.batch_size, input_shape)

        # we compute loss of predicted q and actual q
        self.loss = nn.MSELoss().to(self.Q.device)
        # predicted q = our prediction of next state reward
        # actual q = actual reward for going to next state
        # use MSE loss because emphasizes large errors

        # use .to in pytorch to enable GPU accelerated training
        self.Q.to(self.Q.device)
        self.Qt.to(self.Q.device)
        # self.Qt.load_state_dict(self.Q.state_dict())

        # nueral network back propogation, SGD as suggested by paper
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

    # testing soft update rules
    # target weights = (TAU * policy weights) + ((1-TAU) * target weights)
    def soft_update(self, TAU):
        target_net_state_dict = self.Qt.state_dict()
        policy_net_state_dict = self.Q.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
            self.Qt.load_state_dict(target_net_state_dict)

    # sample the replay buffer and learn from experiences
    def learn(self):

        # take a random sample of transitions
        states, actions, rewards, next_states, terminal = self.replay_buffer.sample()

        if states is not None:

            # copy the weights to the target network every 100 iterations
            if self.learn_count % self.replace_thresh == 0:
                self.Qt.load_state_dict(self.Q.state_dict())

            rewards = T.tensor(rewards).to(
                self.Q.device).reshape(self.batch_size)

            # feed next states batch to NN and get quality prediction for each state
            # using array broadcasting to map the batchs
            q_prediction = self.Q.forward(states)[self.indices, actions]

            q_next = self.Q.forward(next_states)
            q_prediction_arg = T.argmax(q_next, 1)

            # target network should not affect training (gradient descent)
            with T.no_grad():
                q_target_next = self.Qt.forward(next_states)

            # bellman update equation
            y = rewards + self.gamma * \
                q_target_next[self.indices, q_prediction_arg]
            y[terminal] = rewards[terminal]

            # MSE loss y(Qt max) and Q max
            loss = self.loss(q_prediction, y)

            # neural network optimization
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # reduce epsilon
            self.eps = max(self.eps*self.eps_dec, self.eps_min)
            self.learn_count += 1

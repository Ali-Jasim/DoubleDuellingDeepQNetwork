import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_shape, action_shape, hidden_layer):
        super.__init__()
        self.l1 = nn.Linear(*input_shape, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, action_shape)

        self.device = "cuda" if T.cuda.is_available() else "cpu"

    def forward(self, state):

        state = T.tensor(state).to(self.device)

        x = F.relu(self.l1(state))
        x = self.l2(x)

        return x

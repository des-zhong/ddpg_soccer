import torch as T
import torch.nn as nn
import torch.optim as optim

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight,std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, fc3_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)  
        self.ln1 = nn.LayerNorm(fc1_dim) 
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim) 
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.ln3 = nn.LayerNorm(fc3_dim) 
        self.action = nn.Linear(fc3_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.ln1(self.fc1(state)))
        x = T.relu(self.ln2(self.fc2(x)))
        x = T.relu(self.ln3(self.fc3(x)))
        # print(self.action(x))
        action = 100 * T.tanh(self.action(x))
        #print(action)

        return action

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file,map_location='cpu'))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim, fc3_dim):
        super(CriticNetwork, self).__init__()
        input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim,fc3_dim)
        self.q = nn.Linear(fc3_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action):
        x = T.cat((state, action), dim=1)
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        x = T.relu(self.fc3(x))
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file,map_location='cpu'))

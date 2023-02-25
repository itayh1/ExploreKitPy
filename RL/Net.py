
from itertools import pairwise
import torch
import torch.nn as nn

dims = [100, 10]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 128
class Net(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Net, self).__init__()

        self.input_size = state_size
        self.out_size = action_size

        self.seed = torch.manual_seed(seed)
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.adv = nn.Linear(in_features=hidden_size, out_features=self.out_size)
        self.val = nn.Linear(in_features=hidden_size, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x, bsize, hidden_state, cell_state):
        # x = x.view(bsize, 1, self.input_size)

        # conv_out = self.conv_layer1(x)
        # conv_out = self.relu(conv_out)
        # conv_out = self.conv_layer2(conv_out)
        # conv_out = self.relu(conv_out)
        # conv_out = self.conv_layer3(conv_out)
        # conv_out = self.relu(conv_out)
        # conv_out = self.conv_layer4(conv_out)
        # conv_out = self.relu(conv_out)

        # x = x.view(bsize, time_step, 512)

        lstm_out = self.lstm_layer(x, (hidden_state, cell_state))
        out = lstm_out[0].data[:, - 1]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        adv_out = self.adv(out)
        val_out = self.val(out)

        qout = val_out.expand(bsize, self.out_size) + (
                    adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize, self.out_size))

        return qout, (h_n, c_n)

    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, hidden_size).float().to(device)
        c = torch.zeros(1, bsize, hidden_size).float().to(device)

        return h, c

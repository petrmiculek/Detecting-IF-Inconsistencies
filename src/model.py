import torch
import torch.nn as nn


class LSTMBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.fc_size = 256
        self.batch_size = 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, self.fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        # c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(x)  # , (h_0, c_0))
        y = self.fc1(final_hidden_state[-1])
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return y

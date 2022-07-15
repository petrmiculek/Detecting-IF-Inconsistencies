import torch
import torch.nn as nn


class LSTMBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, tokens_length=128):
        super().__init__()
        self.name = ''
        self.hidden_size = hidden_size
        self.input_size = input_size  # input embedding dim = 32
        self.output_size = output_size
        self.fc_size = 32
        self.batch_size = 1
        self.tokens_length = tokens_length

        bidi = True
        scale_if_bidi = 2 if bidi else 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=bidi, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.hidden_size * tokens_length * scale_if_bidi, self.fc_size)
        # self.dropout1 = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.test_mode = False

    def forward(self, x):
        y, (final_hidden_state, final_cell_state) = self.lstm(x)
        y = self.flatten(y)
        y = self.fc1(y)
        # y = self.dropout1(y)
        y = self.relu(y)
        y = self.fc2(y)
        # y = self.sigmoid(y)
        return y

    def predict(self, x):
        y_logits = self.forward(x)
        return self.sigmoid(y_logits)

# class Transformer(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1):
#         super().__init__()
#         self.name = ''
#         self.hidden_size = hidden_size
#         self.input_size = input_size  # input embedding dim = 32
#         self.output_size = output_size
#         self.fc_size = 256
#         self.batch_size = 1
#
        # self.model = nn.Transformer(d_model=self.input_size, nhead=4)

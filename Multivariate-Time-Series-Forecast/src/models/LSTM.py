import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        out, (hidden1, _) = self.lstm1(input)
        out, (hidden2, _) = self.lstm2(out)
        out, (hidden3, _) = self.lstm3(out)
        out = self.fc(hidden3[-1, :, :])

        return out.view(input.size(0))
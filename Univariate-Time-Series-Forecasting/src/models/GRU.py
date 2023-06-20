import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.gru3 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        out, hidden1 = self.gru1(input)
        out, hidden2 = self.gru2(out)
        out, hidden3 = self.gru3(out)
        out = self.fc(hidden3[-1, :, :])

        return out.view(-1,1)
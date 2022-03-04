import torch


class LastJobPredictor(torch.nn.Module):
    def __init__(self, hidden_size, elmo_size):
        super(LastJobPredictor, self).__init__()
        self.layer1 = torch.nn.Linear(hidden_size, hidden_size)
        self.tanh = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(hidden_size, elmo_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.tanh(out)
        out = self.layer2(out)
        return out


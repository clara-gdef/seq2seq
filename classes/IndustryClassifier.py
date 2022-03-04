import torch.nn


class IndustryClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IndustryClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.tanh = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.tanh(out)
        out = self.layer2(out)
        return out

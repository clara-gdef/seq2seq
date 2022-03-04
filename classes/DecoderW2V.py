import torch
from torch.nn import functional as F


class DecoderW2V(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.lin = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        out = self.lin(inputs)
        results = F.softmax(out)

        return results

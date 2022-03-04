import torch


class DecoderWithFT(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = torch.nn.LSTM(input_size + 1, hidden_size, num_layer,  batch_first=True)
        self.lin_out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, encoder_representation, hidden_state, token):
        inputs = torch.cat([encoder_representation, token.type(dtype=torch.FloatTensor).unsqueeze(-1).cuda()], dim=2)
        out, hidden = self.lstm(inputs, hidden_state)
        results = self.lin_out(out)

        return results, hidden

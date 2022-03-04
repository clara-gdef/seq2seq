import torch


class CareerDecoderRNN(torch.nn.Module):
    def __init__(self, vector_size, hidden_size, num_layers, dpo):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.RNN(vector_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_dropout = torch.nn.Dropout(p=dpo)
        self.lin_lstm_out = torch.nn.Linear(hidden_size, vector_size, bias=True)

    def forward(self, career_embedding, hidden_state, token):
        inputs = torch.cat([career_embedding, token], dim=2)
        out, hidden = self.lstm(inputs, hidden_state)
        results = self.lin_lstm_out(self.lstm_dropout(out))

        return results, hidden

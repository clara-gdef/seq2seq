import torch
from torch.nn import functional as F
import ipdb


class CareerEncoderLSTM(torch.nn.Module):
    def __init__(self, vector_size, hidden_size, num_layers, dpo, bidirectional):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(vector_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.lstm_dropout = torch.nn.Dropout(p=dpo)
        self.lin_lstm_out = torch.nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)

        # Vecteur contexte sur les mots
        self.context = torch.nn.Linear(hidden_size, 1, bias=False)

    def job_level(self, job_emb, x_len, enforce_sorted):
        # job_emb = torch.autograd.Variable(job_emb)
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(job_emb, x_len, batch_first=True, enforce_sorted=enforce_sorted)

        out, hidden_state = self.lstm(packed_x)
        H, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        dropped_out_seq = self.lstm_dropout(H)

        out_lstm_transformed = torch.tanh(self.lin_lstm_out(dropped_out_seq))

        last_states = torch.zeros(len(x_len), 1, self.hidden_size*2).cuda()
        for i, length in enumerate(x_len):
            last_states[i] = out_lstm_transformed[i, length-1, :].clone()

        # tmp = self.context(out_lstm_transformed).squeeze(-1)
        # self.alpha = torch.nn.functional.softmax(tmp, dim=1)
        # sentence_rep = torch.einsum("blf,bl->bf", dropped_out_seq, self.alpha)
        return last_states, hidden_state

    def forward(self, job_emb, len_seq, enforce_sorted):
        """len_seq : longueur effective des séquences """
        job_rep, hidden_state = self.job_level(job_emb, len_seq, enforce_sorted)
        # return job_rep, self.alpha, hidden_state
        return job_rep, hidden_state



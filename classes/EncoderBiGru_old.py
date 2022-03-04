import torch
from torch.nn import functional as F


class EncoderBiGru_old(torch.nn.Module):
    def __init__(self, embeddings, vector_size, hidden_size, num_layers, MAX_CAREER_LENGTH, b_size):
        super().__init__()

        self.b_size = b_size

        self.embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0)
        self.embeddings.load_state_dict({'weight': embeddings})

        self.num_layers = num_layers
        self.MAX_CAREER_LENGTH = MAX_CAREER_LENGTH

        self.hidden_size = hidden_size
        # Niveau mot
        self.bi_gru_words = torch.nn.GRU(vector_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.bi_gru_words_dropout = torch.nn.Dropout(p=0.5)

        # Vecteur contexte sur les mots
        self.context = torch.nn.Linear(hidden_size * 2, 1, bias=False)

    def word_level(self, x, x_len, hw):
        # Forward au niveau des mots d'un job
        # x : un batch de séquence // liste de liste d'indices
        # x_len : Un tenseur de la taille des séquences pour le padding

        # with ipdb.launch_ipdb_on_exception():
        # TODO sort sequences by length, and keep the indices
        sorted_indices = [i[0] for i in (sorted(enumerate(x_len),  key=lambda x_len: x_len[1], reverse=True))]
        sorted_lengths = sorted(x_len, reverse=True)
        sorted_sequences = x[sorted_indices, :]

        reverse_dict = {key: value for (key, value) in enumerate(sorted_indices)}
        x_var = torch.autograd.Variable(sorted_sequences)
        emb = self.embeddings(x_var)

        if 0 not in sorted_lengths:
            x = torch.nn.utils.rnn.pack_padded_sequence(emb, sorted_lengths, batch_first=True)
        else:
            non_zeros = [e if e != 0 else 1 for e in sorted_lengths]
            # non_z_indices = [i for i, e in enumerate(sorted_lengths) if e != 0]
            zeros = [i for i, e in enumerate(sorted_lengths) if e == 0]
            # non_zero_length = list(sorted_lengths[i] for i in non_zeros)
            emb[zeros] = -1
            x = torch.nn.utils.rnn.pack_padded_sequence(emb, non_zeros, batch_first=True)

        out, hidden_state = self.bi_gru_words(x, hw)
        H, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        H = self.bi_gru_words_dropout(H)
        M = torch.tanh(H)

        self.alpha = F.softmax(self.context(M).view(H.size(0), -1), dim=1)

        S = torch.matmul(H.permute(0, 2, 1), self.alpha.view(self.alpha.size(0), self.alpha.size(1), 1)).view(
            self.alpha.size(0), self.hidden_size * 2).cuda()
        S = torch.tanh(S)

        S_temp = S.clone()
        for k in reverse_dict.keys():
            S_temp[reverse_dict[k]] = S[k]
        return S_temp, hidden_state

    def forward(self, job_id, len_seq, hw):
        """len_seq : longueur effective des séquences """
        job_rep, hidden_state = self.word_level(job_id, len_seq.tolist(), hw)
        return job_rep, self.alpha, hidden_state

    def initHidden(self):
        return torch.zeros(self.num_layers*2, self.b_size, self.hidden_size)


import torch
import torch.nn.functional as F
import ipdb


class EncoderBiLSTM(torch.nn.Module):
    def __init__(self, embeddings, vector_size, hidden_size, num_layers, MAX_CAREER_LENGTH, b_size):
        super().__init__()

        self.b_size = b_size
        # TODO remove brackets!!!!!!
        self.embeddings = torch.nn.Embedding(embeddings.size(0), 100, padding_idx=0)
        self.embeddings.load_state_dict({'weight': embeddings[:, :100]})

        self.num_layers = num_layers
        self.MAX_CAREER_LENGTH = MAX_CAREER_LENGTH

        self.hidden_size = hidden_size
        # Niveau mot
        self.bi_lstm = torch.nn.LSTM(vector_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.lin_lstm_out = torch.nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)

        # Vecteur contexte sur les mots
        self.context = torch.nn.Linear(hidden_size * 2, 1, bias=False)

    def word_level(self, x, x_len, enforce_sorted):
        # Forward au niveau des mots d'un job
        # x : un batch de séquence // liste de liste d'indices
        # x_len : Un tenseur de la taille des séquences pour le padding

        x_var = torch.autograd.Variable(x)
        emb = self.embeddings(x_var)

        packed_x = torch.nn.utils.rnn.pack_padded_sequence(emb, x_len, batch_first=True, enforce_sorted=enforce_sorted)

        out, hidden_state = self.bi_lstm(packed_x)
        H, lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        seq = self.dropout(H)

        out_lstm_transformed = torch.tanh(self.lin_lstm_out(seq))

        mask = x != 0
        self.alpha = self.masked_softmax(self.context(out_lstm_transformed).squeeze(-1), mask)
        # sentence_rep = torch.matmul(seq.permute(0, 2, 1), self.alpha)
        sentence_rep = torch.einsum("blf,bl->bf", seq, self.alpha)

        # S2 = torch.stack([attentioned_seq[i, v-1, :self.hidden_size] for i, v in enumerate(x_len)])
        # S3 = attentioned_seq[:, 0, self.hidden_size:]

        # return torch.cat([S2, S3], dim=1), hidden_state
        return sentence_rep, hidden_state

    def forward(self, job_id, len_seq, enforce_sorted):
        """len_seq : longueur effective des séquences """
        job_rep, hidden_state = self.word_level(job_id, len_seq, enforce_sorted)
        return job_rep, self.alpha, hidden_state

    def masked_softmax(self, logits, mask):
        logits = logits - torch.min(logits, dim=1, keepdim=True)[0]
        mask = mask.type(dtype=logits.dtype)
        weigths = torch.exp(logits) * mask
        return weigths / torch.sum(weigths, dim=1, keepdim=True)


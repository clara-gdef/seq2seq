import torch
import torch.nn.functional as F
from allennlp.modules.elmo import batch_to_ids


class EncoderWithElmo(torch.nn.Module):
    def __init__(self, elmo, emb_dimension, b_size):
        super().__init__()

        self.b_size = b_size
        self.elmo = elmo
        self.context = torch.nn.Linear(emb_dimension, 1, bias=False)

    def word_level(self, sentences):
        character_ids = batch_to_ids(sentences)

        emb = self.elmo(character_ids.cuda())

        emb_tensor = emb["elmo_representations"][-1]

        # mask = emb["mask"]
        # self.alpha = self.masked_softmax(self.context(emb_tensor).squeeze(-1), mask)
        # sentence_rep = torch.einsum("blf,bl->bf", emb_tensor, self.alpha)

        return emb_tensor

    def forward(self, job_id):
        """len_seq : longueur effective des séquences """
        job_rep = self.word_level(job_id)
        return job_rep

    # def masked_softmax(self, logits, mask):
    #     logits = logits - torch.min(logits, dim=1, keepdim=True)[0]
    #     mask = mask.type(dtype=logits.dtype)
    #     weigths = torch.exp(logits) * mask
    #     return weigths / torch.sum(weigths, dim=1, keepdim=True)


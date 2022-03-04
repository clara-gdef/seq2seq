from torch.utils.data import Dataset
import ipdb


class ProfileDatasetElmoIndices(Dataset):
    def __init__(self, tuple_list):
        self.ids = [e[0] for e in tuple_list if len(e[1]) > 1]
        self.embs = [e[1] for e in tuple_list if len(e[1]) > 1]
        self.career_lengths = [len(e[1]) for e in tuple_list if len(e[1]) > 1]
        self.indices = [e[2] for e in tuple_list if len(e[2]) > 1]
        self.words = [e[3] for e in tuple_list if len(e[3]) > 1]
        self.jobs_lengths = [e[4] for e in tuple_list if len(e[2]) > 1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        identifier = self.ids[index]
        emb = self.embs[index]
        career_length = self.career_lengths[index]
        indice = self.indices[index]
        words = self.words[index]
        job_lengths = self.jobs_lengths[index]
        return identifier, emb, career_length, indice, words, job_lengths

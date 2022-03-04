from torch.utils.data import Dataset


class LinkedInDataset(Dataset):

    def __init__(self, tuple_list, trans):
        self.ids = [e[0] for e in tuple_list["data"]]
        self.data = tuple_list["data"]
        self.lengths = tuple_list["lengths"]
        self.transform = trans

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        people = self.data[index]
        lengths = self.lengths[index]
        identifier, jobs, jobs_len, lj, lj_len = self.transform(people, lengths)
        return identifier, jobs, jobs_len, lj, lj_len

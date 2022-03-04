from torch.utils.data import Dataset


class JobDataset(Dataset):
    def __init__(self, tuple_list, trans=None):
        self.ids = [e[0] for e in tuple_list["data"]]
        self.data = tuple_list["data"]
        self.lengths = tuple_list["lengths"]
        self.transform = trans

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.data[index], self.lengths[index]

from torch.utils.data import Dataset
import ipdb


class JobDatasetElmo(Dataset):
    def __init__(self, tuple_list, trans=None):
        # self.ids = range(len(tuple_list["data"]))
        self.data = tuple_list["data"]
        self.lengths = tuple_list["lengths"]
        self.indices = tuple_list["indices"]
        self.transform = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.lengths[index], self.indices[index]

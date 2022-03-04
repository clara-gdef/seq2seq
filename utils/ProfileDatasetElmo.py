from torch.utils.data import Dataset


class ProfileDatasetElmo(Dataset):
    def __init__(self, tuple_list, trans):
        self.ids = [e[0] for e in tuple_list["data"] if len(e[1]) > 1]
        self.data = [e for e in tuple_list["data"] if len(e[1]) > 1]
        self.lengths = [len(e[1]) for e in tuple_list["data"] if len(e[1]) > 1]
        self.transform = trans

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        people = self.data[index]
        lengths = self.lengths[index]
        identifier, jobs, jobs_len, last_job, last_job_len = self.transform(people, lengths)
        return identifier, jobs, jobs_len, last_job, last_job_len

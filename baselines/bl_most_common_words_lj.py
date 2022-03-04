import argparse

import ipdb
import torch

from tqdm import tqdm
import os
import pickle as pkl
from utils.JobDatasetElmo import JobDatasetElmo
from collections import Counter


def main(args):

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    dataset_train = JobDatasetElmo(datadict["train_data"])
    dataset_valid = JobDatasetElmo(datadict["valid_data"])
    del datadict

    word_count = Counter()
    avg_len = []
    last_jobs = []
    for person in tqdm(dataset_valid, desc="Parsing valid..."):
        profile = person[0][1]
        last_jobs.append(profile[0])
        for job in profile[1:]:
            avg_len.append(len(job))
            for word in job:
                word_count[word] += 1
    for person in tqdm(dataset_train, desc="Parsing train..."):
        profile = person[0][1]
        last_jobs.append(profile[0])
        for job in profile[1:]:
            avg_len.append(len(job))
            for word in job:
                word_count[word] += 1

    average_length = int(torch.mean(torch.FloatTensor(avg_len)))
    most_common_train_valid = [i[0] for i in word_count.most_common(average_length)]

    with open(os.path.join(args.DATA_DIR, "mc_lj_words.txt"), 'a+') as f:
        for iteration in range(len(last_jobs)):
            for w in most_common_train_valid:
                f.write(w + " ")
            f.write("\n")
    with open(os.path.join(args.DATA_DIR, "labels_lj.txt"), 'a+') as f:
        for lj in last_jobs:
            for w in lj:
                f.write(w + " ")
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--input_file", type=str, default="pkl/profiles_elmo.pkl")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    args = parser.parse_args()
    main(args)

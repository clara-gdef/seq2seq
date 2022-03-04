import argparse
import pickle as pkl
import os
import re

from tqdm import tqdm
from random import shuffle
import ipdb


def main(args):
    data_file_train = os.path.join(args.DATA_DIR, args.input_file)

    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    last_jobs_train = {}
    last_jobs_valid = {}
    last_jobs_test = {}

    with open(data_file_train, 'rb') as file:
        data = pkl.load(file)

    with tqdm(data["train_data"]) as pbar:
        for person in pbar:
            last_jobs_train[person[0]] = [words_to_indices(person[1][0], index), turn_dict_to_sequence(person[1][0], index)]
    with tqdm(data["valid_data"]) as pbar:
        for person in pbar:
            last_jobs_valid[person[0]] = [words_to_indices(person[1][0], index), turn_dict_to_sequence(person[1][0], index)]
    with tqdm(data["test_data"]) as pbar:
        for person in pbar:
            last_jobs_test[person[0]] = [words_to_indices(person[1][0], index), turn_dict_to_sequence(person[1][0], index)]

    labels = {"train_data": last_jobs_train, "valid_data": last_jobs_valid, "test_data": last_jobs_test}

    with open(os.path.join(args.DATA_DIR, args.output), "wb") as f:
        pkl.dump(labels, f)


def words_to_indices(sequence, vocab_index):
    job_indices = [vocab_index["SOT"]]
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    word_counter = 1
    for word in sequence["position"]:
        if word_counter < args.MAX_SEQ_LENGTH - 2:
            if word not in vocab_index.keys():
                job_indices.append(vocab_index["UNK"])
            else:
                if re.match(number_regex, word):
                    job_indices.append(vocab_index["NUM"])
                else:
                    job_indices.append(vocab_index[word])
            word_counter += 1
    job_indices.append(vocab_index["SOD"])
    word_counter += 1
    for word in sequence["description"]:
        if word_counter < args.MAX_SEQ_LENGTH - 1:
            if word not in vocab_index.keys():
                job_indices.append(vocab_index["UNK"])
            else:
                if re.match(number_regex, word):
                    job_indices.append(vocab_index["NUM"])
                else:
                    job_indices.append(vocab_index[word])
            word_counter += 1
    job_indices.append(vocab_index["EOD"])
    return job_indices


def turn_dict_to_sequence(dic, vocab_index):
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    new_tup = ["SOT"]
    word_counter = 1
    for word in dic["position"]:
        if word_counter < args.MAX_SEQ_LENGTH - 2:
            if word not in vocab_index.keys():
                new_tup.append("UNK")
            else:
                if re.match(number_regex, word):
                    new_tup.append("NUM")
                else:
                    new_tup.append(word)
            word_counter += 1
    new_tup.append("SOD")
    word_counter += 1
    for word in dic["description"]:
        if word_counter < args.MAX_SEQ_LENGTH - 1:
            if word not in vocab_index.keys():
                new_tup.append("UNK")
            else:
                if re.match(number_regex, word):
                    new_tup.append("NUM")
                else:
                    new_tup.append(word)
            word_counter += 1
    new_tup.append("EOD")
    cleaned_tup = [item for item in new_tup if item != ""]
    return cleaned_tup


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/people_edu_sk_ind.pkl")
    parser.add_argument("--output", type=str, default="pkl/lj_labels.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    args = parser.parse_args()
    main(args)

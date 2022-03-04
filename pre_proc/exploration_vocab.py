import argparse
from tqdm import tqdm
import os
import pickle as pkl
import ipdb
import re
from collections import Counter


def main(args):
    print("Loading data...")
    data_train = []
    data_valid = []
    data_test = []
    suffix = "fs"
    flag_err = False
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    rev_index = {v: k for k, v in index.items()}
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_lj_ft_" + suffix + "_train.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_train.append(data)
            except EOFError:
                flag_err = True
                continue
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_lj_ft_" + suffix + "_valid.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_valid.append(data)
            except EOFError:
                flag_err = True
                continue

    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_lj_ft_" + suffix + "_test.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_test.append(data)
            except EOFError:
                flag_err = True
                continue
    print("Data loaded.")

    with ipdb.launch_ipdb_on_exception():
        word_counter_train_valid = Counter()
        word_counter_test = Counter()

        for person in data_train:
                for indice in person[2]:
                    word_counter_train_valid[rev_index[indice]] += 1
        for person in data_valid:
                for indice in person[2]:
                    word_counter_train_valid[rev_index[indice]] += 1

        total_word_train_valid = sum(v for v in word_counter_train_valid.values())

        for person in data_test:
            for indice in person[2]:
                word_counter_test[rev_index[indice]] += 1

        total_word_test = sum(v for v in word_counter_test.values())

        ipdb.set_trace()

        with open(os.path.join(args.DATA_DIR, args.target_file), "wb") as tgt_f:
            pkl.dump({"train_valid": word_counter_train_valid, "test": word_counter_test}, tgt_f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/profiles_elmo.pkl")
    parser.add_argument("--target_file", type=str, default="pkl/voc_distrib.pkl")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    args = parser.parse_args()
    main(args)

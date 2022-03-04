import argparse
import os
import pickle as pkl
import torch
from tqdm import tqdm
import  re
import ipdb


def main(args):
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")
    datadict_new = dict()
# with ipdb.launch_ipdb_on_exception():
    seq_train, len_train = turn_words_into_indices_wo_padding(datadict["train_data"], index)
    seq_valid, len_valid = turn_words_into_indices_wo_padding(datadict["valid_data"], index)
    seq_test, len_test = turn_words_into_indices_wo_padding(datadict["test_data"], index)
    datadict_new["train"] = {"data": seq_train, "lengths": len_train}
    datadict_new["valid"] = {"data": seq_valid, "lengths": len_valid}
    datadict_new["test"] = {"data": seq_test, "lengths": len_test}
    with open(os.path.join(args.DATA_DIR, "pkl/indices_jobs.pkl"), 'wb') as f:
        pkl.dump(datadict_new, f)


def turn_words_into_indices_wo_padding(datadict, vocab_index):
    all_indices = []
    lengths = []
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    with tqdm(datadict, desc="turning to indices...") as pbar:
        for job in pbar:
            job_indices = [vocab_index["SOT"]]
            word_counter = 1
            for word in job["position"]:
                if word_counter < args.MAX_SEQ_LENGTH:
                    if word not in vocab_index.keys():
                        job_indices.append(vocab_index["UNK"])
                    else:
                        if re.match(number_regex, word):
                            job_indices.append(vocab_index["NUM"])
                        else:
                            job_indices.append(vocab_index[word])
                    word_counter += 1
            job_indices.append(vocab_index["EOT"])
            job_indices.append(vocab_index["SOD"])
            word_counter += 2
            for word in job["description"]:
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
            lengths.append(len(job_indices))
            all_indices.append(job_indices)
            pbar.update(1)
    return all_indices, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/jobs.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    args = parser.parse_args()
    main(args)

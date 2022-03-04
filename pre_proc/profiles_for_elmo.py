import argparse
from tqdm import tqdm
import os
import pickle as pkl
import ipdb
import re

def main(args):
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    new_data = dict()

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    for k in ["train_data", "valid_data", "test_data"]:
        tuples = datadict[k]
        new_data[k] = dict()
        new_data[k]["data"] = []
        new_data[k]["indices"] = []
        new_data[k]["lengths"] = []

        for tup in tqdm(tuples):
            seq, length = turn_tuple_to_sequence(tup)
            indices = turn_words_into_indices_wo_padding(tup, index)
            new_data[k]["data"].append(seq)
            new_data[k]["indices"].append(indices)
            new_data[k]["lengths"].append(length)

    with open(os.path.join(args.DATA_DIR, args.target_file), "wb") as tgt_f:
        pkl.dump(new_data, tgt_f)


def turn_tuple_to_sequence(tup):
    new_profile = []
    lengths = []
    for job in tup[1]:
        word_counter = 1
        new_tup = ["SOT"]
        for word in job["position"]:
            if word_counter < args.MAX_SEQ_LENGTH - 2:
                new_tup.append(word)
                word_counter += 1
        new_tup.append("SOD")
        for word in job["description"]:
            if word_counter < args.MAX_SEQ_LENGTH - 1:
                new_tup.append(word)
                word_counter += 1
        new_tup.append("EOD")
        cleaned_tup = [item for item in new_tup if item != ""]
        new_profile.append(cleaned_tup)
        lengths.append(len(cleaned_tup))
    return (tup[0], new_profile), (tup[0], lengths)


def turn_words_into_indices_wo_padding(tup, vocab_index):
    all_indices = []
    number_regex = re.compile(r'\d+(,\d+)?')
    for job in tup[1]:
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
        all_indices.append(job_indices)
    return (tup[0], all_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/people_edu_sk_ind.pkl")
    parser.add_argument("--target_file", type=str, default="pkl/profiles_elmo.pkl")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    args = parser.parse_args()
    main(args)

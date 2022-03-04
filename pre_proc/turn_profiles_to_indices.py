import argparse
import os
import pickle as pkl
import ipdb
from tqdm import tqdm
import re


def main(args):
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    data_file_train = os.path.join(args.DATA_DIR, args.input_file_train)
    with open(data_file_train, 'rb') as file:
        datadict_train = pkl.load(file)
    data_file_test = os.path.join(args.DATA_DIR, args.input_file_test)
    with open(data_file_test, 'rb') as file:
        datadict_test = pkl.load(file)
    print("Data loaded.")
    datadict_new = dict()
    with ipdb.launch_ipdb_on_exception():
        ipdb.set_trace()
        seq_train, len_train = turn_profiles_into_indices(datadict_train["train"], index)
        seq_valid, len_valid = turn_profiles_into_indices(datadict_train["valid"], index)
        seq_test, len_test = turn_profiles_into_indices(datadict_test["test"], index)
        datadict_new["train"] = {"data": seq_train, "lengths": len_train}
        datadict_new["valid"] = {"data": seq_valid, "lengths": len_valid}
        datadict_new["test"] = {"data": seq_test, "lengths": len_test}
    with open(os.path.join(args.DATA_DIR, "pkl/ppl_indices_num.pkl"), 'wb') as f:
        pkl.dump(datadict_new, f)


def turn_profiles_into_indices(datadict, vocab_index):
    people = []
    lengths_list = []
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    with tqdm(datadict, desc="turning to indices...") as pbar:
        for person in tqdm(pbar):
            jobs = []
            id_person = person[0]
            lengths = []
            for i, job in enumerate(person[1:][0]):
                indices = [vocab_index["SOT"]]
                word_counter = 1
                for word in job["position"]:
                    if word_counter < args.MAX_SEQ_LENGTH:
                        if word not in vocab_index.keys():
                            indices.append(vocab_index["UNK"])
                        else:
                            if re.match(number_regex, word):
                                indices.append(vocab_index["NUM"])
                            else:
                                indices.append(vocab_index[word])
                        word_counter += 1
                indices.append(vocab_index["EOT"])
                indices.append(vocab_index["SOD"])
                word_counter += 2
                for word in job["description"]:
                    if word_counter < args.MAX_SEQ_LENGTH - 1:
                        if word not in vocab_index.keys():
                            indices.append(vocab_index["UNK"])
                        else:
                            if re.match(number_regex, word):
                                indices.append(vocab_index["NUM"])
                            else:
                                indices.append(vocab_index[word])
                        word_counter += 1
                indices.append(vocab_index["EOD"])
                lengths.append(word_counter + 1)
                jobs.append(indices)

            people.append((id_person, jobs))
            lengths_list.append((id_person, lengths))
    return people, lengths_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_train", type=str, default="pkl/people_index.pkl")
    parser.add_argument("--input_file_test", type=str, default="pkl/people_index_test.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    args = parser.parse_args()
    main(args)

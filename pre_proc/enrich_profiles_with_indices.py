import argparse
from tqdm import tqdm
import os
import pickle as pkl
import ipdb
import re


def main(args):

    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(args.DATA_DIR, args.index_file), 'rb') as f:
            vocab = pkl.load(f)

        with open(os.path.join(args.DATA_DIR, args.indices_file), 'rb') as f:
            indices = pkl.load(f)

        train_file = "pkl/prof_rep_elmo_train_cpu.pkl"
        valid_file = "pkl/prof_rep_elmo_valid_cpu.pkl"
        test_file = "pkl/prof_rep_elmo_test_cpu.pkl"

        # loading data
        print("Loading data...")
        train_file = os.path.join(args.DATA_DIR, train_file)
        datadict_train = {"data": []}
        flag_err = False
        with open(train_file, "rb") as f:
            while not flag_err:
                try:
                    datadict_train["data"].append(pkl.load(f))
                except EOFError:
                    flag_err = True
                    continue
        print("Train file loaded.")

        map_indices_and_profiles(datadict_train, indices['train_data'], "train", vocab)

        del datadict_train

        valid_file = os.path.join(args.DATA_DIR, valid_file)
        datadict_valid = {"data": []}
        flag_err = False
        with open(valid_file, "rb") as f:
            while not flag_err:
                try:
                    datadict_valid["data"].append(pkl.load(f))
                except EOFError:
                    flag_err = True
                    continue
        print("Valid file loaded.")

        map_indices_and_profiles(datadict_valid, indices['valid_data'], "valid", vocab)

        del datadict_valid

        test_file = os.path.join(args.DATA_DIR, test_file)
        datadict_test = {"data": []}
        flag_err = False
        with open(test_file, "rb") as f:
            while not flag_err:
                try:
                    datadict_test["data"].append(pkl.load(f))
                except EOFError:
                    flag_err = True
                    continue
        print("Test file loaded.")

        map_indices_and_profiles(datadict_test, indices['test_data'], "test", vocab)

        del datadict_test


def map_indices_and_profiles(emb, indices, suffix, vocab):
    emb_dict = {}
    for i in tqdm(emb["data"], desc="Building emb dictionary for " + suffix + "..."):
        emb_dict[i[0]] = i[1]

    indices_dict = {}
    for i in tqdm(indices, desc="Building indices dictionary for " + suffix + "..."):
        indices_dict[i[0]] = i[1]

    enriched_data = {"data": []}
    for k in tqdm(emb_dict.keys(), desc="Enriching data for " + suffix + "..."):
        indices, lengths = turn_profiles_into_indices(indices_dict[k][1:], vocab)
        words = turn_list_to_word_seq(indices_dict[k][1:], vocab)
        ipdb.set_trace()
        enriched_data["data"].append((k, emb_dict[k], indices, words, lengths))

    with open(os.path.join(args.DATA_DIR, "pkl/prof_ind_elmo_" + suffix + "_cpu.pkl"), "wb") as f:
        pkl.dump(enriched_data, f)


def turn_list_to_word_seq(l, vocab_index):
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    profile = []
    for dic in l:
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
        profile.append(cleaned_tup)
    return profile


def turn_profiles_into_indices(liste, vocab_index):
    person = []
    lengths_list = []
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    for i, job in enumerate(liste):
        indices = [vocab_index["SOT"]]
        word_counter = 1
        for word in job["position"]:
            if word_counter < args.MAX_SEQ_LENGTH - 2:
                if word not in vocab_index.keys():
                    indices.append(vocab_index["UNK"])
                else:
                    if re.match(number_regex, word):
                        indices.append(vocab_index["NUM"])
                    else:
                        indices.append(vocab_index[word])
                word_counter += 1
        indices.append(vocab_index["SOD"])
        word_counter += 1
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
        person.append(indices)
        lengths_list.append(len(indices))
    return person, lengths_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices_file", type=str, default="pkl/people_edu_sk_ind.pkl")
    parser.add_argument("--embeddings_file", type=str, default="pkl")
    parser.add_argument("--target_file", type=str, default="pkl/profiles_elmo_.pkl")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    args = parser.parse_args()
    main(args)

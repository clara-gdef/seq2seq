import argparse
import pickle as pkl
import re

import ipdb
import torch
import os
import fastText
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes import EncoderWithElmo
from utils import ProfileDatasetElmo
from utils.Utils import transform_for_elmo_lj, collate_for_profiles_elmo


def main(args):
    with ipdb.launch_ipdb_on_exception():
        print("Loading word vectors...")
        ft_model = fastText.load_model(os.path.join(args.model_dir, args.ft_model))
        print("Word vectors loaded.")
        print("Loading data...")
        data_file = os.path.join(args.DATA_DIR, args.input_file)
        with open(data_file, 'rb') as file:
            datadict = pkl.load(file)
        with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
            index = pkl.load(f)
        print("Data loaded")

        dataset_train = ProfileDatasetElmo(datadict["train_data"], transform_for_elmo_lj)
        dataset_valid = ProfileDatasetElmo(datadict["valid_data"], transform_for_elmo_lj)
        dataset_test = ProfileDatasetElmo(datadict["test_data"], transform_for_elmo_lj)
        del datadict

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                      collate_fn=collate_for_profiles_elmo,
                                      shuffle=True, num_workers=0, drop_last=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                      collate_fn=collate_for_profiles_elmo,
                                      shuffle=True, num_workers=0, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                     collate_fn=collate_for_profiles_elmo,
                                     shuffle=True, num_workers=0, drop_last=True)
        dictionary = get_representations(ft_model, dataloader_train, dataloader_valid, dataloader_test, index)


def get_representations(enc, dataloader_train, dataloader_valid, dataloader_test, index):
    lj = "_lj" if args.lj else ""

    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep2" + lj + "_ft_" + args.ft_type + "_train.pkl"), 'ab') as v2_F:
        with tqdm(dataloader_train, desc="Computing ft representation for train...") as pbar:
            for identifiers, profiles, _ in pbar:
                reps = []
                profile = profiles[0]
                if len(profile) > 0:
                    for job in profile[1:]:
                        avg_job_rep = sentence_to_avg_emb(job, enc)
                        reps.append(avg_job_rep)
                    if lj:
                        lj_ind = word_seq_to_indices(profile[0], index)
                        pkl.dump((identifiers[0], torch.mean(torch.stack(reps), dim=0), lj_ind, len(reps)), v2_F)
                    else:
                        pkl.dump((identifiers[0], torch.mean(torch.stack(reps), dim=0), len(reps)), v2_F)
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep2" + lj + "_ft_" + args.ft_type + "_valid.pkl"), 'ab') as v2_F:
        with tqdm(dataloader_valid, desc="Computing ft representation for valid...") as pbar:
            for identifiers, profiles, _ in pbar:
                reps = []
                profile = profiles[0]
                if len(profile) > 0:
                    for job in profile:
                        avg_job_rep = sentence_to_avg_emb(job, enc)
                        reps.append(avg_job_rep)
                    if lj:
                        lj_ind = word_seq_to_indices(profile[0], index)
                        pkl.dump((identifiers[0], torch.mean(torch.stack(reps), dim=0), lj_ind, len(reps)), v2_F)
                    else:
                        pkl.dump((identifiers[0], torch.mean(torch.stack(reps), dim=0), len(reps)), v2_F)
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep2" + lj + "_ft_" + args.ft_type + "_test.pkl"), 'ab') as v2_F:
        with tqdm(dataloader_test, desc="Computing ft representation for test...") as pbar:
            counter = 0
            for identifiers, profiles, _ in pbar:
                counter += 1
                reps = []
                profile = profiles[0]
                if len(profile) > 0:
                    for job in profile:
                        avg_job_rep = sentence_to_avg_emb(job, enc)
                        reps.append(avg_job_rep)
                    if lj:
                        lj_ind = word_seq_to_indices(profile[0], index)
                        pkl.dump((identifiers[0], torch.mean(torch.stack(reps), dim=0), lj_ind, len(reps)), v2_F)
                    else:
                        pkl.dump((identifiers[0], torch.mean(torch.stack(reps), dim=0), len(reps)), v2_F)


def sentence_to_avg_emb(sentence, ft_model):
    tmp_tensor = []
    for word in sentence:
        try:
            word_emb = torch.from_numpy(ft_model.get_word_vector(word))
            tmp_tensor.append(word_emb)
        except KeyError:
            pass
    return torch.mean(torch.stack(tmp_tensor), dim=0)


def word_seq_to_indices(word_list, vocab_index):
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    job_indices = []
    word_counter = 0
    for word in word_list:
        if word_counter < args.MAX_SEQ_LENGTH:
            if word not in vocab_index.keys():
                job_indices.append(vocab_index["UNK"])
            else:
                if re.match(number_regex, word):
                    job_indices.append(vocab_index["NUM"])
                else:
                    job_indices.append(vocab_index[word])
            word_counter += 1
    return job_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--input_file", type=str, default="pkl/profiles_elmo.pkl")
    parser.add_argument("--ft_model", type=str, default='ft_from_scratch.bin')
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--lj", type=bool, default=True)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20')

    args = parser.parse_args()
    main(args)

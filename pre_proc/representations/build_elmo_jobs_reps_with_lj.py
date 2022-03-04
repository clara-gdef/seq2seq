import argparse
import pickle as pkl
import torch
import os
import ipdb
from allennlp.modules import Elmo
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes import EncoderWithElmo
from utils import ProfileDatasetElmo
from utils.Utils import transform_for_elmo_lj, collate_profiles_lj_elmo


def main(args):
    elmo_dimension = 1024
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    print("Initializing ELMo...")
    elmo = Elmo(options_file, weight_file, 2, requires_grad=False, dropout=0)
    encoder_job = EncoderWithElmo(elmo, elmo_dimension, args.batch_size)
    print("ELMo ready.")

    # loading data
    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    with ipdb.launch_ipdb_on_exception():

        dataset_train = ProfileDatasetElmo(datadict["train_data"], transform_for_elmo_lj)
        dataset_valid = ProfileDatasetElmo(datadict["valid_data"], transform_for_elmo_lj)
        dataset_test = ProfileDatasetElmo(datadict["test_data"], transform_for_elmo_lj)
        del datadict

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                      collate_fn=collate_profiles_lj_elmo,
                                      shuffle=True, num_workers=0, drop_last=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                      collate_fn=collate_profiles_lj_elmo,
                                      shuffle=True, num_workers=0, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                      collate_fn=collate_profiles_lj_elmo,
                                      shuffle=True, num_workers=0, drop_last=True)
        enc_job = encoder_job.cuda()

        dictionary = get_representations(enc_job, dataloader_train, dataloader_valid, dataloader_test)

        with open(os.path.join(args.DATA_DIR, "pkl/job_rep_elmo.pkl"), 'wb') as v2_F:
            pkl.dump(dictionary, v2_F)


def get_representations(enc, train, valid, test):
    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_elmo_train_lj_cpu.pkl"), 'ab') as v2_F:
            with tqdm(train, desc="Computing elmo representation for train...") as pbar:
                for identifiers, profiles, prof_len, last_job, last_job_len in pbar:
                    reps = []
                    profile = profiles[0]
                    lj = last_job[0]
                    if len(profile) > 0:
                        for job in profile:
                            enc_output = enc([job])
                            reps.append(torch.mean(enc_output, dim=1).cpu())
                        pkl.dump((identifiers[0], reps, len(reps), lj, last_job_len[0]), v2_F)
                    # if counter > 100:
                    #     break
        with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_elmo_valid_lj_cpu.pkl"), 'ab') as v2_F:
            with tqdm(valid, desc="Computing elmo representation for valid...") as pbar:
                for identifiers, profiles, prof_len, last_job, last_job_len in pbar:
                    reps = []
                    profile = profiles[0]
                    lj = last_job[0]
                    if len(profile) > 0:
                        for job in profile:
                            enc_output = enc([job])
                            reps.append(torch.mean(enc_output, dim=1).cpu())
                        pkl.dump((identifiers[0], reps, len(reps), lj, last_job_len[0]), v2_F)
        with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_elmo_test_lj_cpu.pkl"), 'ab') as v2_F:
            with tqdm(test, desc="Computing elmo representation for test...") as pbar:
                counter = 0
                for identifiers, profiles, prof_len, last_job, last_job_len in pbar:
                    counter += 1
                    reps = []
                    profile = profiles[0]
                    lj = last_job[0]
                    if len(profile) > 0:
                        for job in profile:
                            enc_output = enc([job])
                            reps.append(torch.mean(enc_output, dim=1).cpu())
                        pkl.dump((identifiers[0], reps, len(reps), lj, last_job_len[0]), v2_F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/profiles_elmo.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo')
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)




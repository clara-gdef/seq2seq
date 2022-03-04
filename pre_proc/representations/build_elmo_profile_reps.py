import argparse
import pickle as pkl
import torch
import os
import ipdb
from allennlp.modules import Elmo
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes import CareerEncoderLSTM, CareerEncoderRNN
from utils import ProfileDatasetElmo
from utils.Utils import transform_for_elmo_lj, collate_profiles_lj_elmo


def main(args):
    print("Loading data...")
    train_file = os.path.join(args.DATA_DIR, args.train_file)
    datadict_train = {"data": []}
    flag_err = False

    with open(train_file, "rb") as f:
        while not flag_err:
            try:
                datadict_train["data"].append(pkl.load(f))
            except EOFError:
                flag_err = True
                continue

    valid_file = os.path.join(args.DATA_DIR, args.valid_file)
    datadict_valid = {"data": []}
    flag_err = False
    with open(valid_file, "rb") as f:
        while not flag_err:
            try:
                datadict_valid["data"].append(pkl.load(f))
            except EOFError:
                flag_err = True
                continue

    test_file = os.path.join(args.DATA_DIR, args.test_file)
    datadict_test = {"data": []}
    flag_err = False
    with open(test_file, "rb") as f:
        while not flag_err:
            try:
                datadict_test["data"].append(pkl.load(f))
            except EOFError:
                flag_err = True
                continue

    print("Data loaded.")
    with ipdb.launch_ipdb_on_exception():

        enc_hs = str.split(args.enc_model, sep="_")[7]
        enc_lr = str.split(args.enc_model, sep="_")[3]
        enc_type = str.split(args.enc_model, sep="_")[1]
        enc_ep = str.split(args.enc_model, sep="_")[-2]
        elmo_size = 1024

        if enc_type == "w2v":
            if str.split(args.enc_model, sep="_")[2] == "ce":
                encoder_career = CareerEncoderLSTM(elmo_size, int(enc_hs), 1, args.dpo, args.bidirectional).cuda()
            elif str.split(args.enc_model, sep="_")[2] == "rnn":
                encoder_career = CareerEncoderRNN(elmo_size, int(enc_hs), 1, args.dpo, args.bidirectional).cuda()
            else:
                encoder_career = CareerEncoderLSTM(elmo_size, int(enc_hs), 1, args.dpo, args.bidirectional).cuda()

        enc_weights = os.path.join(args.model_dir, args.enc_model)
        encoder_career.load_state_dict(torch.load(enc_weights))

        dataset_train = ProfileDatasetElmo(datadict_train, transform_for_elmo_lj)
        dataset_valid = ProfileDatasetElmo(datadict_valid, transform_for_elmo_lj)
        dataset_test = ProfileDatasetElmo(datadict_test, transform_for_elmo_lj)
        del datadict_train, datadict_valid, datadict_test

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                      collate_fn=collate_profiles_lj_elmo,
                                      shuffle=True, num_workers=0, drop_last=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                      collate_fn=collate_profiles_lj_elmo,
                                      shuffle=True, num_workers=0, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                     collate_fn=collate_profiles_lj_elmo,
                                     shuffle=True, num_workers=0, drop_last=True)
        encoder_career = encoder_career.cuda()

        get_representations(encoder_career, dataloader_train, dataloader_valid, dataloader_test, enc_ep, enc_hs, enc_lr,
                            enc_type)


def get_representations(enc, train, valid, test, enc_ep, enc_hs, enc_lr, enc_type):
    suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)
    with open(os.path.join(args.DATA_DIR, "pkl/career_train_cpu" + suffix + ".pkl"), 'ab') as v2_F:
        with tqdm(train, desc="Computing elmo representation for train...") as pbar:
            for ids, jobs, career_len, lj, lj_len in pbar:
                profile_tensor = torch.cat(jobs[0]).unsqueeze(0).cuda()
                z_people, hidden_state = enc(profile_tensor, [len(jobs[0])], enforce_sorted=False)
                pkl.dump((ids[0], z_people.cpu()), v2_F)

    with open(os.path.join(args.DATA_DIR, "pkl/career_valid_cpu" + suffix + ".pkl"), 'ab') as v2_F:
        with tqdm(valid, desc="Computing elmo representation for valid...") as pbar:
            for ids, jobs, career_len, lj, lj_len in pbar:
                profile_tensor = torch.cat(jobs[0]).unsqueeze(0).cuda()
                z_people, hidden_state = enc(profile_tensor, [len(jobs[0])], enforce_sorted=False)
                pkl.dump((ids[0], z_people.cpu()), v2_F)

    with open(os.path.join(args.DATA_DIR, "pkl/career_test_cpu" + suffix + ".pkl"), 'ab') as v2_F:
        with tqdm(test, desc="Computing elmo representation for test...") as pbar:
            for ids, jobs, career_len, lj, lj_len in pbar:
                profile_tensor = torch.cat(jobs[0]).unsqueeze(0).cuda()
                z_people, hidden_state = enc(profile_tensor, [len(jobs[0])], enforce_sorted=False)
                pkl.dump((ids[0], z_people.cpu()), v2_F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="pkl/prof_rep_elmo_train_cpu.pkl")
    parser.add_argument("--valid_file", type=str, default="pkl/prof_rep_elmo_valid_cpu.pkl")
    parser.add_argument("--test_file", type=str, default="pkl/prof_rep_elmo_test_cpu.pkl")
    parser.add_argument("--dpo", type=float, default=.0)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--model_dir", type=str,
                        default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/elmo_w2v')
    parser.add_argument("--enc_model", type=str,
                        default='elmo_w2v_rnn_bs64_lr0.001_tf0_hs_512_max_ep_300_encCareer_best_ep_98_savec')
    parser.add_argument("--rnn_model", type=str,
                        default='elmo_agg_bs64_lr0.0001_dpo0.5_max_ep_300_dechs_256_declr_lr0.001_decep_19_40k_rnn_best_ep_60')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    args = parser.parse_args()
    main(args)

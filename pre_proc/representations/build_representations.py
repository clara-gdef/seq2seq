import argparse
import pickle as pkl
import torch
import os
import ipdb
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes import DecoderLSTM, EncoderBiLSTM, CareerEncoderLSTM, MLP
from utils.Utils import transform_indices, collate_profiles_lj

from utils.LinkedInDataset import LinkedInDataset


def main(args):
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)
    print("Loading data...")

    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(args.DATA_DIR, "pkl/ppl_indices_100.pkl"), "rb") as f:
            datadict = pkl.load(f)

        dataset_train = LinkedInDataset(datadict["train"], transform_indices)
        dataset_valid = LinkedInDataset(datadict["valid"], transform_indices)
        dataset_test = LinkedInDataset(datadict["test"], transform_indices)

        dataloader_train = DataLoader(dataset_train, batch_size=1, collate_fn=collate_profiles_lj, shuffle=True, num_workers=0, drop_last=False)
        dataloader_valid = DataLoader(dataset_valid, batch_size=1,
                                     collate_fn=collate_profiles_lj,
                                     shuffle=True, num_workers=0, drop_last=False)
        dataloader_test = DataLoader(dataset_test, batch_size=1,
                                     collate_fn=collate_profiles_lj,
                                     shuffle=True, num_workers=0, drop_last=False)

        dimension = 100
        hidden_size = 16
        num_layers = 1

        encoder_job = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, 64, 1)
        enc_weights = os.path.join(args.model_dir, args.enc_model)
        encoder_job.load_state_dict(torch.load(enc_weights))

        enc_job = encoder_job.cuda()

        ## profile RNN
        rnn = CareerEncoderLSTM(hidden_size * 2, hidden_size * 2, 1, 0.5, False)
        rnn = rnn.cuda()

        ## career dynamic mlp
        mlp = MLP(hidden_size * 2, hidden_size * 2)
        mlp = mlp.cuda()
        dico = get_representations_for_all(enc_job, mlp, rnn, dataloader_train, dataloader_valid, dataloader_test)
        with open(os.path.join(args.DATA_DIR, "pkl/prof_rep.pkl"), 'wb') as v_F:
            pkl.dump(dico, v_F)


def get_tuples(data, indices):
    new_data_dict = {"train": {"data": [], "lengths": []}, "valid": {"data": [], "lengths": []}, "test": {"data": [], "lengths": []}}
    for i, tup in enumerate(tqdm(data["train"]["data"], desc="finding indices for train...")):
        if tup[0] in indices:
            indices.remove(tup[0])
            new_data_dict["train"]["data"].append(data["train"]["data"][i])
            new_data_dict["train"]["lengths"].append(data["train"]["lengths"][i])
    for i, tup in enumerate(tqdm(data["valid"]["data"], desc="finding indices for valid...")):
        if tup[0] in indices:
            indices.remove(tup[0])
            new_data_dict["valid"]["data"].append(data["valid"]["data"][i])
            new_data_dict["valid"]["lengths"].append(data["valid"]["lengths"][i])
    for i, tup in enumerate(tqdm(data["test"]["data"], desc="finding indices for test...")):
        if tup[0] in indices:
            indices.remove(tup[0])
            new_data_dict["test"]["data"].append(data["test"]["data"][i])
            new_data_dict["test"]["lengths"].append(data["test"]["lengths"][i])

    return new_data_dict


def get_representations_for_all(enc, mlp, rnn, train, valid, test):
    # reps = {"train": {}, "valid": {}, "test": {}}
    with ipdb.launch_ipdb_on_exception():
        # get_representations(enc, mlp, rnn, train, "train")
        # get_representations(enc, mlp, rnn, valid, "valid")
        # get_representations(enc, mlp, rnn, test, "test")
        get_job_representations(enc, train, "train")
        get_job_representations(enc, valid, "valid")
        get_job_representations(enc, test, "test")


def get_representations(enc, mlp, rnn, dataloader, name):
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_" + name + ".pkl"), 'ab') as v_F:
        with tqdm(dataloader, desc="Computing representation for " + name + "...") as pbar:
            for ids, profiles, profiles_len, last_jobs, last_jobs_len in pbar:
                profile = profiles[0]
                if len(profile) > 0 :
                    num_lines = sum([len(e) for e in profiles])
                    num_col = max([e for i in profiles_len for e in i])
                    prof_tensor = torch.zeros(num_lines, num_col, dtype=torch.int64).cuda()
                    count = 0
                    for i in range(len(profiles)):
                        for job in profiles[i]:
                            prof_tensor[count, :len(job)] = torch.LongTensor(job).cuda()
                            count += 1
                    profile_len_flat = [e for i in profiles_len for e in i]
                    enc_output, att, enc_hidden_out = enc(prof_tensor, profile_len_flat, enforce_sorted=False)

                    jobs_reps = []
                    start = 0
                    for seq in profiles_len:
                        end = start + len(seq)
                        jobs_reps.append(enc_output[start:end].unsqueeze(0))
                        start = end

                    job_reps = torch.zeros(1, 1, 32).cuda()
                    for i in range(len(jobs_reps)):
                        target = jobs_reps[i]
                        seq_len = len(profiles_len[i])
                        enc_output, attention, enc_hidden_out = rnn(target, [seq_len], enforce_sorted=False)
                        job_reps[i, :, :] = enc_output

                    lj_app = mlp(job_reps)
                    pkl.dump((ids[0], lj_app), v_F)


def get_job_representations(enc, dataloader, name):
    with open(os.path.join(args.DATA_DIR, "pkl/job_mean_rep_" + name + ".pkl"), 'ab') as mf:
        with open(os.path.join(args.DATA_DIR, "pkl/last_job_rep_" + name + ".pkl"), 'ab') as lf:
            with tqdm(dataloader, desc="Computing representation for " + name + "...") as pbar:
                for ids, profiles, profiles_len, last_jobs, last_jobs_len in pbar:
                    profile = profiles[0]
                    if len(profile) > 0:
                        num_lines = sum([len(e) for e in profiles])
                        num_col = max([e for i in profiles_len for e in i])
                        prof_tensor = torch.zeros(num_lines, num_col, dtype=torch.int64).cuda()
                        count = 0
                        for i in range(len(profiles)):
                            for job in profiles[i]:
                                prof_tensor[count, :len(job)] = torch.LongTensor(job).cuda()
                                count += 1
                        profile_len_flat = [e for i in profiles_len for e in i]
                        job_encoded, _, _ = enc(prof_tensor, profile_len_flat, enforce_sorted=False)

                        lj_encoded, _, _ = enc(torch.LongTensor(last_jobs[0]).unsqueeze(0).cuda(), [last_jobs_len[0]], enforce_sorted=False)

                        pkl.dump((ids[0], torch.mean(job_encoded, dim=0)), mf)
                        pkl.dump((ids[0], lj_encoded), lf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--indices_file", type=str, default="degree.p")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--source_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--enc_model", type=str,
                        default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_enc_best_ep_99")
    parser.add_argument("--v2_model", type=str,
                        default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_mlp_best_ep_99")
    parser.add_argument("--v3_model", type=str,
                        default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_rnn_best_ep_99")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--output_file", type=str, default="reps.pkl")
    args = parser.parse_args()
    main(args)




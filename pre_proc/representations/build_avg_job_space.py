import argparse
import pickle as pkl
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes import EncoderBiLSTM
from utils.JobDataset import JobDataset
from utils.Utils import collate_for_jobs
import ipdb


def main(args):
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    dimension = embeddings.shape[1]

    hidden_size = 64
    num_layers = 1

    print("Loading splits")
    with open(os.path.join(args.DATA_DIR, "pkl/jobs_s" + str(args.split) + "_indices.pkl"), 'rb') as file:
        indices = pkl.load(file)

    all_indices = indices["train"][:]
    all_indices.extend(indices["valid"])
    all_indices.extend(indices["test"])
    dataset = JobDataset(datadict, all_indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_for_jobs,
                            shuffle=True, num_workers=3, drop_last=True)

    encoder_job = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)

    enc_weights = os.path.join(args.model_dir + "/split" + str(args.split), args.enc_model)
    encoder_job.load_state_dict(torch.load(enc_weights))

    enc_job = encoder_job.cuda()

    build_avg_space_job_rep(enc_job, dataloader, args)


def build_avg_space_job_rep(enc, data_loader, args):
    #vtensors = list()
    job_index = dict()
    counter = 1
    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(args.DATA_DIR, "encoded_jobs_tensor.pkl"), 'ab') as f:
            with tqdm(data_loader, desc="Building avg space for job reps...") as pbar:
                for profile, seq_length in pbar:
                    b_size = len(profile)

                    profile_tensor = torch.stack(profile, dim=0).cuda()
                    enc_output, attention, encoder_hidden = enc(profile_tensor, seq_length)
                    tnr = enc_output.detach().cpu().type(torch.float32)
                    #tensors.append(tnr)
                    pkl.dump(tnr, f)
                    for ind in range(b_size):
                    #ipdb.set_trace()
                        # pkl.dump(tnr, f)
                        job_index[str(profile[ind])] = counter
                        counter += 1
        #with open(os.path.join(args.DATA_DIR, "encoded_jobs_tensors.pkl"), "wb") as f:
        #    pkl.dump(tensors, f)

        with open(os.path.join(args.DATA_DIR, "encoded_jobs_index.pkl"), 'wb') as f:
            pkl.dump(job_index, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/indices_all.pkl")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_vocab_3j.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_vocab_3j.pkl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="s2s")
    parser.add_argument("--enc_model", type=str, default="s2s_sep_bs200_enc_best_model_ep_99")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq')
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

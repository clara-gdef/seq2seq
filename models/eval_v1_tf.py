import argparse
import os
import pickle as pkl
from datetime import datetime

import ipdb
import quiviz
import torch
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import DecoderLSTM, EncoderBiLSTM
from pre_proc.from_indices_to_words import decode_indices
from utils.LinkedInDataset import LinkedInDataset
from utils.Utils import transform_indices, collate_profiles_lj, compute_crossentropy


def main(args):
    xp_title = "eval m_v1 bs256 mxlen64"
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    dimension = 100

    hidden_size = 16
    num_layers = 1

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        data = pkl.load(file)
    print("Data loaded.")

    dataset = LinkedInDataset(data["test"], transform_indices)

    decoder_job = DecoderLSTM(embeddings, hidden_size, num_layers, dimension, embeddings.size(0), args.MAX_CAREER_LENGTH)
    encoder_job = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)

    enc_weights = os.path.join(args.model_dir + "/splitless", args.enc_model)
    dec_weights = os.path.join(args.model_dir + "/splitless", args.dec_model)
    encoder_job.load_state_dict(torch.load(enc_weights))
    decoder_job.load_state_dict(torch.load(dec_weights))

    enc_job = encoder_job.cuda()
    dec_job = decoder_job.cuda()

    # dictionary = main_for_one_split(args, enc_job, dec_job, data_tl, testit, index, faiss_index, encoded_jobs_index)
    dictionary = main_for_one_split(args, enc_job, dec_job, dataset, index)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    print(dictionary)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/splitless/" + args.model_type + "_bs256_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder_job, decoder_job, dataset, vocab_index):

    dataloader_test = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_profiles_lj,
                                  shuffle=True, num_workers=0, drop_last=True)

    eval_perplexity = evaluate_perp(args, encoder_job, decoder_job, dataloader_test, vocab_index)
    dictionary = {"perplexity": eval_perplexity}
    return dictionary


def evaluate_perp(args, encoder, decoder, dataloader_test, vocab_index):
    cross_entropy_overall = []
    encoder.eval()
    decoder.eval()
    nb_tokens = 0
    rev_index = {v: k for k, v in vocab_index.items()}
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_test, desc="Evaluating perplexity...") as pbar:
            for ids, profiles, profile_len, last_jobs, last_jobs_len in pbar:
                b_size = 1
                if len(cross_entropy_overall) % 100 == 1:
                    print(torch.exp(torch.sum(torch.FloatTensor(cross_entropy_overall) / float(nb_tokens))).item())
                if len(profile_len[0]) > 0:
                    profile = profiles[0]
                    profile_tensor = torch.zeros(len(profile), max(profile_len[0]), dtype=torch.int64)
                    for i in range(len(profile)):
                        profile_tensor[i][:profile_len[0][i]] = torch.LongTensor(profile[i])
                    profile_tensor = profile_tensor.cuda()

                    enc_output, attention, enc_hidden_out = encoder(profile_tensor, profile_len[0], enforce_sorted=False)

                    job_rep_tensor = torch.mean(enc_output, dim=0).unsqueeze(0)
                    decoder_hidden = (torch.zeros(1, b_size, 32).cuda(), torch.zeros(1, b_size, 32).cuda())

                    lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
                    for i in range(len(last_jobs)):
                        lj_tensor[i, :len(last_jobs[i])] = torch.LongTensor(last_jobs[i]).cuda()
                    lj_app = job_rep_tensor.expand(b_size, max(last_jobs_len), 32)
                    decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, lj_tensor)

                    tmp = compute_crossentropy(decoder_output.transpose(2, 1), lj_tensor)
                    # print(torch.exp(tmp.item()))
                    cross_entropy_overall.append(tmp.item())
                    nb_tokens += last_jobs_len[0]

                    pred_file_overall = os.path.join(args.DATA_DIR, "results/eval_output_v1_tf.txt")
                    pred_text_overall = "SOT " + decode_indices(decoder_output[0].argmax(-1), rev_index)
                    pred_text_overall += "\n"
                    with open(pred_file_overall, "a") as pf:
                        pf.write(pred_text_overall)
                    label_file_overall = os.path.join(args.DATA_DIR, "results/eval_label_v1_tf.txt")
                    label_text_overall = decode_indices(lj_tensor[0], rev_index)
                    label_text_overall += "\n"
                    with open(label_file_overall, "a") as lf:
                        lf.write(label_text_overall)

    return {"perplexity_overall": torch.exp(torch.sum(torch.FloatTensor(cross_entropy_overall) / float(nb_tokens))).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--encoded_jobs", type=str, default="")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="s2s")
    parser.add_argument("--record_outputs", type=bool, default=True)
    parser.add_argument("--enc_model", type=str, default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_enc_best_ep_92")
    parser.add_argument("--dec_model", type=str, default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_dec_best_ep_92")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq')
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

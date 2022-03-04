import argparse
import math
import os
import pickle as pkl
from datetime import datetime

import ipdb
import quiviz
import torch
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import DecoderLSTM, EncoderBiLSTM, MLP
from pre_proc.from_indices_to_words import decode_indices
from utils.LinkedInDataset import LinkedInDataset
from utils.Utils import transform_indices, collate_profiles_lj, compute_crossentropy


def main(args):
    xp_title = "eval m_v1 bs256 mxlen32"
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    with open(os.path.join(args.DATA_DIR, args.input_file), 'rb') as file:
        data = pkl.load(file)
    print("Data loaded.")

    dataset_test = LinkedInDataset(data["test"], transform_indices)

    dimension = 100

    hidden_size = 16
    num_layers = 1

    decoder_job = DecoderLSTM(embeddings, hidden_size, num_layers, dimension, embeddings.size(0), args.MAX_CAREER_LENGTH)
    encoder_job = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)
    mlp = MLP(hidden_size*2, hidden_size*2)

    enc_weights = os.path.join(args.model_dir, args.enc_model)
    dec_weights = os.path.join(args.model_dir, args.dec_model)
    mlp_weights = os.path.join(args.model_dir, args.mlp_model)
    encoder_job.load_state_dict(torch.load(enc_weights))
    decoder_job.load_state_dict(torch.load(dec_weights))
    mlp.load_state_dict(torch.load(mlp_weights))

    enc_job = encoder_job.cuda()
    dec_job = decoder_job.cuda()
    mlp = mlp.cuda()

    dictionary = main_for_one_split(args, enc_job, dec_job, mlp, index, dataset_test)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    print(dictionary)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/"+ str(args.model_type) +"/splitless/" + args.model_type + "_" + args.mlp_model + "_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder_job, decoder_job, mlp, vocab_index, dataset_test):

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj,
                                  shuffle=True, num_workers=0, drop_last=False)

    eval_perplexity = evaluate_perp(args, encoder_job, decoder_job, mlp, dataloader_test, vocab_index)
    eval_BLEU = None
    dictionary = {"perplexity": eval_perplexity, "BLEU": eval_BLEU}
    return dictionary


def evaluate_perp(args, encoder, decoder, mlp, dataloader_test, vocab_index):
    cross_entropy_overall = []
    nb_tokens = 0
    encoder.eval()
    decoder.eval()
    mlp.eval()
    rev_index = {v: k for k, v in vocab_index.items()}
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_test, desc="Evaluating perplexity for v2...") as pbar:
            for ids, profiles, profile_len, last_jobs, last_jobs_len in pbar:
                if len(profile_len[0]) > 0:
                    if len(cross_entropy_overall) % 100 == 1:
                        print(math.exp(torch.sum(torch.FloatTensor(cross_entropy_overall))/float(nb_tokens)))

                    b_size = 1
                    profile = profiles[0]
                    profile_tensor = torch.zeros(len(profile), max(profile_len[0]), dtype=torch.int64)
                    for i in range(len(profile)):
                        profile_tensor[i][:profile_len[0][i]] = torch.LongTensor(profile[i])
                    profile_tensor = profile_tensor.cuda()

                    enc_output, attention, enc_hidden_out = encoder(profile_tensor, profile_len[0], enforce_sorted=False)

                    job_rep_tensor = torch.mean(enc_output, dim=0).unsqueeze(0)
                    lj_approximation = mlp(job_rep_tensor)

                    decoder_hidden = (torch.zeros(1, b_size, 32).cuda(), torch.zeros(1, b_size, 32).cuda())

                    lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
                    for i in range(len(last_jobs)):
                        lj_tensor[i, :len(last_jobs[i])] = torch.LongTensor(last_jobs[i]).cuda()
                    lj_app = lj_approximation.expand(b_size, max(last_jobs_len), 32)
                    decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, lj_tensor)
                    # decoder_outputs.append(decoder_output)

                    cross_entropy_overall.append(compute_crossentropy(decoder_output.transpose(2, 1), lj_tensor).item())

                    nb_tokens += last_jobs_len[0]

                    pred_file_overall = os.path.join(args.DATA_DIR, "results/eval_output_v2_tf.txt")
                    pred_text_overall = "SOT " + decode_indices(decoder_output[0].argmax(-1), rev_index)
                    pred_text_overall += "\n"
                    with open(pred_file_overall, "a") as pf:
                        pf.write(pred_text_overall)
                    label_file_overall = os.path.join(args.DATA_DIR, "results/eval_label_v2_tf.txt")
                    label_text_overall = decode_indices(lj_tensor[0], rev_index)
                    label_text_overall += "\n"
                    with open(label_file_overall, "a") as lf:
                        lf.write(label_text_overall)
                        # out_file = os.path.join(args.DATA_DIR, "results/split" + str(args.split) + '/eval_output_v2.pkl')
                        # out_data = {"id": ids,
                        #             "last_job": last_jobs[0][0],
                        #             "pred": out_tokens}
                        # with open(out_file, "ab") as f:
                           # pkl.dump(out_data, f)
        # out_file = os.path.join(args.DATA_DIR, args.out_file)
        # with open(out_file, "wb") as f:
        #     pkl.dump(recorded_results, f)

    return {
        # "perplexity_title": 2**(torch.sum(torch.FloatTensor(cross_entropy_title) / float(nb_token_title)).item()),
        #     "perplexity_desc": 2**(torch.sum(torch.FloatTensor(cross_entropy_desc) / float(nb_token_desc)).item()),
            "perplexity_overall": math.exp(torch.sum(torch.FloatTensor(cross_entropy_overall)) / float(nb_tokens))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=True)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--encoded_jobs", type=str, default="")
    parser.add_argument("--record_outputs", type=bool, default=True)
    parser.add_argument("--out_file", type=str, default="out_file_v2_full.pkl")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="s2s_v2")
    parser.add_argument("--enc_model", type=str, default="s2s_v2_bs256_lr0.001_tf1_hs_16_max_ep_100_40k_enc_last_ep_100")
    parser.add_argument("--dec_model", type=str, default="s2s_v2_bs256_lr0.001_tf1_hs_16_max_ep_100_40k_dec_last_ep_100")
    parser.add_argument("--mlp_model", type=str, default="s2s_v2_bs256_lr0.001_tf1_hs_16_max_ep_100_40k_mlp_last_ep_100")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

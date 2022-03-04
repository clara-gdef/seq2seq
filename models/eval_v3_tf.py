import argparse
import os
import pickle as pkl
from datetime import datetime

import ipdb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import DecoderLSTM, EncoderBiLSTM, CareerEncoderLSTM, MLP
from pre_proc.from_indices_to_words import decode_indices
from utils.LinkedInDataset import LinkedInDataset
from utils.Utils import transform_indices, collate_profiles_lj, compute_crossentropy


def main(args):

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

    ## Career RNN
    rnn = CareerEncoderLSTM(hidden_size * 2, hidden_size * 2, 1, 0.5, False)

    ## career dynamic mlp
    mlp = MLP(hidden_size*2, hidden_size*2)

    enc_weights = os.path.join(args.model_dir, args.enc_model)
    dec_weights = os.path.join(args.model_dir, args.dec_model)
    rnn_weights = os.path.join(args.model_dir, args.rnn_model)
    mlp_weights = os.path.join(args.model_dir, args.mlp_model)
    encoder_job.load_state_dict(torch.load(enc_weights))
    decoder_job.load_state_dict(torch.load(dec_weights))
    rnn.load_state_dict(torch.load(rnn_weights))
    mlp.load_state_dict(torch.load(mlp_weights))

    enc_job = encoder_job.cuda()
    dec_job = decoder_job.cuda()
    rnn = rnn.cuda()
    mlp = mlp.cuda()

    dictionary = main_for_one_split(args, enc_job, dec_job, rnn, mlp, index, dataset_test, hidden_size*2)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    print(dictionary)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/" + str(args.model_type) + "/splitless/" + args.model_type + "_" + args.rnn_model + "_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder_job, decoder_job, rnn, mlp, vocab_index, dataset_test, hidden_size):

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                 collate_fn=collate_profiles_lj,
                                 shuffle=True, num_workers=0, drop_last=False)

    eval_perplexity = evaluate_perp(args, encoder_job, decoder_job, rnn, mlp, dataloader_test, vocab_index, hidden_size)
    eval_BLEU = None
    dictionary = {"perplexity": eval_perplexity, "BLEU": eval_BLEU}
    return dictionary


def evaluate_perp(args, encoder, decoder, rnn, mlp, dataloader_test, vocab_index, hidden_size):
    cross_entropy_overall = []
    nb_tokens = 0
    encoder.eval()
    decoder.eval()
    highest_perp = -1
    wrongest_tuple = None
    right_answer = None
    rnn.eval()
    mlp.eval()
    b_size = 1
    flag_has_changed = False
    rev_index = {v: k for k, v in vocab_index.items()}
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_test, desc="Evaluating perplexity for v3...") as pbar:
            for ids, profiles, profiles_len, last_jobs, last_jobs_len in pbar:
                #try:
                if len(profiles_len[0]) > 0:
                    if len(cross_entropy_overall) % 100 == 1:
                        print(2 ** (torch.sum(
                            torch.FloatTensor(cross_entropy_overall) / float(nb_tokens))).item())
                        print("HIGHEST PERP: " + str(highest_perp))
                        print("OBTAINED FOR PRED: " + str(wrongest_tuple))
                        print("IN LIEU OF: " + str(right_answer))
                    num_lines = sum([len(e) for e in profiles])
                    num_col = max([e for i in profiles_len for e in i])
                    prof_tensor = torch.zeros(num_lines,  num_col, dtype=torch.int64).cuda()
                    count = 0
                    for i in range(len(profiles)):
                        for job in profiles[i]:
                            prof_tensor[count, :len(job)] = torch.LongTensor(job).cuda()
                            count += 1
                    profile_len_flat = [e for i in profiles_len for e in i]
                    enc_output, att, enc_hidden_out = encoder(prof_tensor, profile_len_flat, enforce_sorted=False)

                    jobs_reps = []
                    start = 0
                    for seq in profiles_len:
                        end = start + len(seq)
                        jobs_reps.append(enc_output[start:end].unsqueeze(0))
                        start = end

                    job_reps = torch.zeros(b_size, 1, hidden_size).cuda()
                    for i in range(len(jobs_reps)):
                        target = jobs_reps[i]
                        seq_len = len(profiles_len[i])
                        enc_output, attention, enc_hidden_out = rnn(target, [seq_len], enforce_sorted=False)
                        job_reps[i, :, :] = enc_output

                    tmp = mlp(job_reps)
                    lj_app = tmp.expand(b_size, max(last_jobs_len), 32)

                    decoder_hidden = (torch.zeros(1, b_size, 32).cuda(), torch.zeros(1, b_size, 32).cuda())

                    lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
                    for i in range(len(last_jobs)):
                        lj_tensor[i, :len(last_jobs[i])] = torch.LongTensor(last_jobs[i]).cuda()
                    decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, lj_tensor)
                    # decoder_outputs.append(decoder_output)

                    loss = compute_crossentropy(decoder_output.transpose(2, 1), lj_tensor).item()
                    cross_entropy_overall.append(loss)
                    nb_tokens += last_jobs_len[0]

                    if loss > highest_perp:
                        highest_perp = loss
                        wrongest_tuple = decode_indices(decoder_output[0].argmax(-1), rev_index)
                        right_answer = decode_indices(lj_tensor[0], rev_index)
                        flag_has_changed = True

                    pred_file_overall = os.path.join(args.DATA_DIR, "results/eval_output_agg_splitless_tf.txt")
                    pred_text_overall = decode_indices(decoder_output[0].argmax(-1), rev_index)
                    pred_text_overall += "\n"
                    with open(pred_file_overall, "a") as pf:
                        pf.write(pred_text_overall)
                    label_file_overall = os.path.join(args.DATA_DIR, "results/eval_label_agg_splitless_tf.txt")
                    label_text_overall = decode_indices(lj_tensor[0], rev_index)
                    label_text_overall += "\n"
                    with open(label_file_overall, "a") as lf:
                        lf.write(label_text_overall)
                    debug_file = os.path.join(args.DATA_DIR, "results/eval_wrongest_agg.txt")
                    if flag_has_changed:
                        with open(debug_file, "a") as f:
                            f.write(str({"perp": highest_perp,
                                     "pred": wrongest_tuple,
                                     "label": right_answer}))
                            f.write("\n")
                        flag_has_changed = False

    return {"perplexity_overall":  torch.exp(torch.sum(torch.FloatTensor(cross_entropy_overall)) / float(nb_tokens)).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=True)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--encoded_jobs", type=str, default="")
    parser.add_argument("--record_outputs", type=str, default="True")
    parser.add_argument("--out_file", type=str, default="out_file_v2_full.pkl")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="s2s_v3")
    parser.add_argument("--enc_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_enc_best_ep_80")
    parser.add_argument("--dec_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_dec_best_ep_80")
    parser.add_argument("--rnn_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_rnn_best_ep_80")
    parser.add_argument("--mlp_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_mlp_best_ep_80")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

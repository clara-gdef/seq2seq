import argparse
import os
import pickle as pkl
import torch
from tqdm import tqdm
import ipdb
import json
import spacy
import re

from classes import DecoderLSTM, EncoderBiLSTM, CareerEncoderLSTM, MLP
from from_indices_to_words import decode_indices


def main(args):
    with ipdb.launch_ipdb_on_exception():
        nlp = spacy.load("fr", create_pipeline=custom_pipeline)
        file = os.path.join(args.DATA_DIR, args.input_file)
        with open(file, 'r') as f:
            data = f.read()
        index_file = os.path.join(args.DATA_DIR, args.index_file)
        with open(index_file, 'rb') as f:
            index = pkl.load(f)
        person = pre_process_file(args, data, index, nlp)
        print("Loading models...")
        enc, dec, rnn, mlp = load_models(args)
        print("Models Loaded!")
        predict(person, enc, dec, rnn, mlp, index)


def predict(person, encoder, decoder, rnn, mlp, vocab_index):
    rev_index = {v: k for k, v in vocab_index.items()}
    num_lines = len(person["data"])
    num_col = max(person["lengths"])
    prof_tensor = torch.zeros(num_lines, num_col, dtype=torch.int64).cuda()
    count = 0
    # for i in range(num_lines):
    for job in person["data"]:
        prof_tensor[count, :len(job)] = torch.LongTensor(job).cuda()
        count += 1
    profile_len_flat = [e for e in person["lengths"]]
    enc_output, att, enc_hidden_out = encoder(prof_tensor, profile_len_flat, enforce_sorted=False)

    jobs_reps = []
    start = 0
    for seq in person["lengths"]:
        end = start + seq
        jobs_reps.append(enc_output[start:end].unsqueeze(0))
        start = end

    # job_reps = torch.zeros(1, 1, 32).cuda()
    # for i in range(len(jobs_reps)):
    #     target = jobs_reps[i]
    #     seq_len = person["lengths"][i]
    #     enc_output, attention, enc_hidden_out = rnn(target, [num_lines], enforce_sorted=False)
    #     job_reps[i, :, :] = enc_output

    job_reps, attention, enc_hidden_out = rnn(enc_output.unsqueeze(0), [num_lines], enforce_sorted=False)

    tmp = mlp(job_reps.unsqueeze(1))
    lj_app = tmp.cuda()

    decoder_hidden = (torch.zeros(1, 1, 32).cuda(), torch.zeros(1, 1, 32).cuda())

    tmp = torch.zeros(1, dtype=torch.int64) + vocab_index["SOT"]
    token = tmp.unsqueeze(1).cuda()
    decoder_outputs = []
    while token != vocab_index["EOD"] and len(decoder_outputs) < args.MAX_SEQ_LENGTH:
        decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, token)
        output = decoder_output.argmax(-1)
        token = output.type(torch.LongTensor).cuda()
        decoder_outputs.append(token)

    decoded_future = decode_indices(decoder_outputs, rev_index)
    out_file = os.path.join(args.DATA_DIR, args.name + "_future.txt")
    with open(out_file, "w") as f:
        f.write(decoded_future)
    print(decoded_future)


def load_models(args):
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)

    dimension = 100
    hidden_size = 16
    num_layers = 1

    decoder_job = DecoderLSTM(embeddings, hidden_size, num_layers, dimension, embeddings.size(0), args.MAX_CAREER_LENGTH)
    encoder_job = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)

    del embeddings

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

    return enc_job, dec_job, rnn, mlp


def pre_process_file(args, file, index, nlp):
    line = json.loads(file)
    tmp_jobs = []
    for job in tqdm(line["positions"], desc="Parsing and tokenizing person..."):
        tmp = {"position": [], "description": []}
        for i in nlp.tokenizer.pipe((j for j in job["position"]), batch_size=64, n_threads=2):
            for word in i:
                tmp["position"].append(str(word).lower())
        for i in nlp.tokenizer.pipe((j for j in job["description"]), batch_size=64, n_threads=2):
            for word in i:
                tmp["position"].append(str(word).lower())
        tmp_jobs.append(tmp)
    indices, lengths = turn_jobs_to_indices(tmp_jobs, index, args.MAX_SEQ_LENGTH)
    formed_person = {"id": line["id"], "data": indices, "lengths": lengths}
    return formed_person


def turn_jobs_to_indices(profile, vocab_index, MAX_SEQ_LENGTH):
    lengths = []
    person = []
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    for job in tqdm(profile, desc="Turning profile to indices..."):
        indices = [vocab_index["SOT"]]
        word_counter = 1
        for word in job["position"]:
            if word_counter < MAX_SEQ_LENGTH:
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
            if word_counter < MAX_SEQ_LENGTH - 1:
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
        person.append(indices)

    return person, lengths


def to_array_comp(doc):
        return [w.orth_ for w in doc]


def custom_pipeline(nlp):
    return (nlp.tagger, to_array_comp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="etienne.txt")
    parser.add_argument("--name", type=str, default="etienne")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tf", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=.5)
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="s2s_v3")
    parser.add_argument("--enc_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_enc_best_ep_92")
    parser.add_argument("--dec_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_dec_best_ep_92")
    parser.add_argument("--rnn_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_rnn_best_ep_92")
    parser.add_argument("--mlp_model", type=str, default="s2s_agg_bs128_lr0.001_dpo0.5_hs_16_max_ep_100_40k_mlp_best_ep_92")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

import argparse

import itertools
import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import DecoderLSTM, EncoderBiLSTM, MLP
import os
import pickle as pkl
import math
import torch
from utils.Utils import collate_profiles_lj, transform_indices, save_best_model_v2, model_checkpoint_v2
from datetime import datetime
from torch.utils.data import DataLoader
from utils.LinkedInDataset import LinkedInDataset
import random
import ipdb


def main(args):
    xp_title = "V2 allParams bs" + str(args.batch_size) + " tf" + str(args.tf) + " lr" + str(args.lr)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)

    print("Loading data...")
    with open(os.path.join(args.DATA_DIR, args.input_file), 'rb') as file:
        data = pkl.load(file)
    print("Data loaded.")

    dataset_train = LinkedInDataset(data["train"], transform_indices)
    dataset_valid = LinkedInDataset(data["valid"], transform_indices)

    dimension = 100

    hidden_size = args.hidden_size
    num_layers = 1

    decoder = DecoderLSTM(embeddings, hidden_size, num_layers, dimension, embeddings.size(0), args.MAX_CAREER_LENGTH)
    encoder = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)
    # # for efficient memory save
    del embeddings

    enc_weights = os.path.join(args.model_dir, args.enc_model)
    dec_weights = os.path.join(args.model_dir, args.dec_model)
    encoder.load_state_dict(torch.load(enc_weights))
    decoder.load_state_dict(torch.load(dec_weights))

    enc = encoder.cuda()
    dec = decoder.cuda()

    ## MLP
    model = MLP(hidden_size*2, hidden_size*2)
    model = model.cuda()

    dictionary = main_for_one_split(args, enc, dec, model, dataset_train, dataset_valid)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/splitless/" + args.model_type + "_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder, decoder, model, dataset_train, dataset_valid):
    res_epoch = {}
    optim_mlp = torch.optim.Adam(itertools.chain(encoder.parameters(), model.parameters(), decoder.parameters()), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    best_val_loss = 1e+300

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj,
                                  shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, encoder, decoder, model, criterion,
                                  args, best_val_loss, optim_mlp, dataloader_train,
                                  dataloader_valid)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']
    return res_epoch


@quiviz.log
def main_for_one_epoch(epoch, encoder, decoder, model, criterion,
                       args, best_val_loss, optim, dataloader_train, dataloader_valid):
    print("Training and validating for epoch " + str(epoch))

    if epoch == 1:
        print("INITIAL LOSS: " + str(valid(encoder, decoder, model, dataloader_valid, criterion)))

    train_loss = train(encoder, decoder, model, dataloader_train, criterion, optim)

    valid_loss = valid(encoder, decoder, model, dataloader_valid, criterion)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_model_v2(args, epoch, target_dir, encoder, decoder, model, optim)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_v2(args, epoch, target_dir, encoder, decoder, model, optim)

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(encoder, decoder, model, dataloader_train, criterion, optim_mlp):
    loss_list = []
    encoder.train()
    decoder.train()
    model.train()
    nb_tokens = 0
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_train, desc="Training...") as pbar:
            for ids, profiles, profile_len, last_jobs, last_jobs_len in pbar:
                optim_mlp.zero_grad()
                b_size = len(ids)

                loss = 0

                faulty_tuples = []
                for i in range(b_size):
                    if not profiles[i]:
                        faulty_tuples.append(i)
                if faulty_tuples:
                    ids = list(ids)
                    profiles = list(profiles)
                    profile_len = list(profile_len)
                    last_jobs = list(last_jobs)
                    last_jobs_len = list(last_jobs_len)
                    for index in faulty_tuples:
                        del ids[index]
                        del profiles[index]
                        del profile_len[index]
                        del last_jobs[index]
                        del last_jobs_len[index]

                b_size = len(ids)

                num_lines = sum([len(e) for e in profiles])
                num_col = max([e for i in profile_len for e in i])
                prof_tensor = torch.zeros(num_lines, num_col, dtype=torch.int64).cuda()
                count = 0
                for i in range(len(profiles)):
                    for job in profiles[i]:
                        prof_tensor[count, :len(job)] = torch.LongTensor(job).cuda()
                        count += 1
                profile_len_flat = [e for i in profile_len for e in i]
                enc_output, att, enc_hidden_out = encoder(prof_tensor, profile_len_flat, enforce_sorted=False)

                mean_job_reps = []
                start = 0
                for seq in profile_len:
                    end = start + len(seq)
                    mean_job_reps.append(torch.mean(enc_output[start:end], dim=0).unsqueeze(0))
                    start = end

                job_rep_tensor = torch.stack(mean_job_reps)
                lj_approximation = model(job_rep_tensor)

                decoder_hidden = (torch.zeros(1, b_size, 32).cuda(), torch.zeros(1, b_size, 32).cuda())

                lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
                for i in range(len(last_jobs)):
                    lj_tensor[i, :len(last_jobs[i])] = torch.LongTensor(last_jobs[i]).cuda()

                lj_app = lj_approximation.expand(b_size, max(last_jobs_len), 32)
                decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, lj_tensor)
                loss += criterion(decoder_output.transpose(2, 1), lj_tensor)

                # lj_app = lj_approximation
                # for tok in range(1, lj_tensor.shape[1]):
                #     use_teacher_forcing = True if random.random() < args.tf else False
                #     targets = lj_tensor[:, tok]
                #     decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, tokens)
                #     loss += criterion(decoder_output.transpose(2, 1), targets.unsqueeze(1))
                #     if use_teacher_forcing:
                #         tokens = targets.unsqueeze(1)
                #     else:
                #         output = decoder_output.argmax(-1)
                #         tokens = output.type(torch.LongTensor).cuda()

                nb_tokens += sum([i for i in last_jobs_len])

                # loss = criterion(lj_approximation, enc_lj.unsqueeze(1))
                loss_list.append(loss.item())
                loss.backward()
                optim_mlp.step()

    return {"train_loss": torch.sum(torch.FloatTensor(loss_list) / nb_tokens).item(),
            "train_perplexity": math.exp((sum(loss_list) / nb_tokens))}


def valid(encoder, decoder, model, dataloader_valid, criterion):
    loss_list = []
    encoder.eval()
    decoder.eval()
    model.eval()
    nb_tokens = 0
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_valid, desc="Validating...") as pbar:
            for ids, profiles, profile_len, last_jobs, last_jobs_len in pbar:
                b_size = len(ids)
                loss = 0
                faulty_tuples = []
                for i in range(b_size):
                    if not profiles[i]:
                        faulty_tuples.append(i)
                if faulty_tuples:
                    ids = list(ids)
                    profiles = list(profiles)
                    profile_len = list(profile_len)
                    last_jobs = list(last_jobs)
                    last_jobs_len = list(last_jobs_len)
                    for index in faulty_tuples:
                        del ids[index]
                        del profiles[index]
                        del profile_len[index]
                        del last_jobs[index]
                        del last_jobs_len[index]

                b_size = len(ids)

                num_lines = sum([len(e) for e in profiles])
                num_col = max([e for i in profile_len for e in i])
                prof_tensor = torch.zeros(num_lines, num_col, dtype=torch.int64).cuda()
                count = 0
                for i in range(len(profiles)):
                    for job in profiles[i]:
                        prof_tensor[count, :len(job)] = torch.LongTensor(job).cuda()
                        count += 1
                profile_len_flat = [e for i in profile_len for e in i]
                enc_output, att, enc_hidden_out = encoder(prof_tensor, profile_len_flat, enforce_sorted=False)

                mean_job_reps = []
                start = 0
                for seq in profile_len:
                    end = start + len(seq)
                    mean_job_reps.append(torch.mean(enc_output[start:end], dim=0).unsqueeze(0))
                    start = end

                job_rep_tensor = torch.stack(mean_job_reps)

                lj_approximation = model(job_rep_tensor)
                decoder_hidden = (torch.zeros(1, b_size, 32).cuda(), torch.zeros(1, b_size, 32).cuda())

                lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
                for i in range(len(last_jobs)):
                    lj_tensor[i, :len(last_jobs[i])] = torch.LongTensor(last_jobs[i]).cuda()

                lj_app = lj_approximation.expand(b_size, max(last_jobs_len), 32)
                decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, lj_tensor)
                loss += criterion(decoder_output.transpose(2, 1), lj_tensor)

                # lj_app = lj_approximation.expand(b_size, max(last_jobs_len), 32)
                # lj_app = lj_approximation
                # for tok in range(1, lj_tensor.shape[1]):
                #     use_teacher_forcing = True if random.random() < args.tf else False
                #     targets = lj_tensor[:, tok]
                #     decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, tokens)
                #     loss += criterion(decoder_output.transpose(2, 1), targets.unsqueeze(1))
                #     if use_teacher_forcing:
                #         tokens = targets.unsqueeze(1)
                #     else:
                #         output = decoder_output.argmax(-1)
                #         tokens = output.type(torch.LongTensor).cuda()

                nb_tokens += sum([i for i in last_jobs_len])

                loss_list.append(loss.item())

    return {"valid_loss": torch.sum(torch.FloatTensor(loss_list) / nb_tokens).item(),
            "valid_perplexity": math.exp((sum(loss_list) / nb_tokens))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="s2s_v2_allParams")
    parser.add_argument("--enc_model", type=str, default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_enc_best_ep_56")
    parser.add_argument("--dec_model", type=str, default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_dec_best_ep_56")
    parser.add_argument("--tf", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

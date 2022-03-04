import argparse
import itertools

import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import DecoderLSTM, EncoderBiLSTM, CareerEncoderLSTM, DecoderW2V
import os
import pickle as pkl
import torch
from utils.Utils import collate_profiles_lj, transform_indices, save_best_model_v3, model_checkpoint_v3
from datetime import datetime
import glob
from torch.utils.data import DataLoader, Subset
from utils.LinkedInDataset import LinkedInDataset
import ipdb


def main(args):
    xp_title = "w2v bs" + str(args.batch_size) + " dpo" + str(args.dpo) + " hs" + str(args.hidden_size) + " lr" + str(args.lr)
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

    decoder_job = DecoderLSTM(embeddings, hidden_size, num_layers, dimension, embeddings.size(0),
                              args.MAX_CAREER_LENGTH)
    encoder_job = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)
    # # for efficient memory save
    del embeddings

    enc_weights = os.path.join(args.model_dir, args.enc_model)
    dec_weights = os.path.join(args.model_dir, args.dec_model)
    encoder_job.load_state_dict(torch.load(enc_weights))
    decoder_job.load_state_dict(torch.load(dec_weights))

    enc_job = encoder_job.cuda()
    dec_job = decoder_job.cuda()

    ## profile RNN
    enc_career = CareerEncoderLSTM(hidden_size * 2, hidden_size * 2, 1, args.dpo, args.bidirectional)
    dec_career = DecoderW2V(hidden_size*2, embeddings.size(0))

    enc_career = enc_career.cuda()
    dec_career = dec_career.cuda()

    dictionary = main_for_one_split(args, enc_job, dec_job, enc_career, dec_career, dataset_train, dataset_valid, hidden_size * 2)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/splitless/" + args.model_type + "_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder_job, decoder_job, rnn, mlp, dataset_train, dataset_valid, hidden_size):
    res_epoch = {}
    optim = torch.optim.Adam(itertools.chain(rnn.parameters(), mlp.parameters()), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    best_val_loss = 1e+300

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj,
                                  shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, encoder_job, decoder_job, rnn, mlp, criterion,
                                  args, best_val_loss, optim, dataloader_train,
                                  dataloader_valid, hidden_size)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']
    return res_epoch


@quiviz.log
def main_for_one_epoch(epoch, encoder, decoder, rnn, mlp, criterion,
                       args, best_val_loss, optim, dataloader_train, dataloader_valid, hidden_size):
    print("Training and validating for epoch " + str(epoch))

    train_loss = train(encoder, decoder, rnn, mlp, dataloader_train, criterion, optim, args.MAX_SEQ_LENGTH, hidden_size,
                       epoch)

    valid_loss = valid(encoder, decoder, rnn, mlp, dataloader_valid, criterion, args.MAX_SEQ_LENGTH, hidden_size)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_model_v3(args, epoch, target_dir, encoder, decoder, rnn, mlp, optim)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_v3(args, epoch, target_dir, encoder, decoder, rnn, mlp, optim)
        best_val_loss = valid_loss['valid_loss']

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(encoder, decoder, rnn, mlp, dataloader_train, criterion, optim, MAX_SEQ_LENGTH, hidden_size, ep):
    loss_list = []
    encoder.train()
    decoder.train()
    rnn.train()
    mlp.train()
    nb_tokens = 0
    # optim.param_groups[0]['lr'] = args.lr * ((1 - float(ep) / float(args.epoch)) ** 0.9)
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_train, desc="Training...") as pbar:
            for ids, profiles, profiles_len, last_jobs, last_jobs_len in pbar:
                # try:
                b_size = len(profiles)

                optim.zero_grad()

                faulty_tuples = []
                for i in range(b_size):
                    if len(profiles[i]) < 0:
                        faulty_tuples.append(i)
                if faulty_tuples:
                    ids = list(ids)
                    profiles = list(profiles)
                    profile_len = list(profiles_len)
                    last_jobs = list(last_jobs)
                    last_jobs_len = list(last_jobs_len)
                    for index in faulty_tuples:
                        del ids[index]
                        del profiles[index]
                        del profile_len[index]
                        del last_jobs[index]
                        del last_jobs_len[index]
                    del faulty_tuples
                b_size = len(profiles)

                num_lines = sum([len(e) for e in profiles])
                prof_tensor = torch.zeros(num_lines, MAX_SEQ_LENGTH, dtype=torch.int64).cuda()
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
                    if seq_len > 0:
                        enc_output, attention, enc_hidden_out = rnn(target, [seq_len], enforce_sorted=False)
                        job_reps[i, :, :] = enc_output

                # lj_app = job_reps.expand(b_size, max(last_jobs_len), 32)
                tmp = mlp(job_reps)
                lj_app = tmp.expand(b_size, max(last_jobs_len), 32)
                decoder_hidden = (torch.zeros(1, b_size, 32).cuda(), torch.zeros(1, b_size, 32).cuda())

                lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
                for i in range(len(last_jobs)):
                    lj_tensor[i, :len(last_jobs[i])] = torch.LongTensor(last_jobs[i]).cuda()

                decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, lj_tensor)
                loss = criterion(decoder_output.transpose(2, 1), lj_tensor)

                nb_tokens += sum([i for e in profiles_len for i in e])

                # loss = criterion(lj_approximation, enc_lj.unsqueeze(1))
                loss_list.append(loss.item())
                loss.backward()
                optim.step()
                # except RuntimeError:
                #     continue

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_perplexity": 2 ** (sum(loss_list) / nb_tokens)}


def valid(encoder, decoder, rnn, mlp, dataloader_valid, criterion, MAX_SEQ_LENGTH, hidden_size):
    loss_list = []
    encoder.eval()
    decoder.eval()
    rnn.eval()
    mlp.eval()
    nb_tokens = 0
    with tqdm(dataloader_valid, desc="Validating...") as pbar:
        for ids, profiles, profiles_len, last_jobs, last_jobs_len in pbar:
            # try:
            b_size = len(profiles)

            faulty_tuples = []
            for i in range(b_size):
                if len(profiles[i]) < 0:
                    faulty_tuples.append(i)
            if faulty_tuples:
                ids = list(ids)
                profiles = list(profiles)
                profile_len = list(profiles_len)
                last_jobs = list(last_jobs)
                last_jobs_len = list(last_jobs_len)
                for index in faulty_tuples:
                    del ids[index]
                    del profiles[index]
                    del profile_len[index]
                    del last_jobs[index]
                    del last_jobs_len[index]
                del faulty_tuples
            b_size = len(profiles)

            num_lines = sum([len(e) for e in profiles])
            num_col = max([e for i in profiles_len for e in i])
            prof_tensor = torch.zeros(num_lines, num_col, dtype=torch.int64).cuda()
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
                if seq_len > 0:
                    enc_output, attention, enc_hidden_out = rnn(target, [seq_len], enforce_sorted=False)
                    job_reps[i, :, :] = enc_output

            tmp = mlp(job_reps)
            lj_app = tmp.expand(b_size, max(last_jobs_len), 32)

            decoder_hidden = (torch.zeros(1, b_size, 32).cuda(), torch.zeros(1, b_size, 32).cuda())

            lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
            for i in range(len(last_jobs)):
                lj_tensor[i, :len(last_jobs[i])] = torch.LongTensor(last_jobs[i]).cuda()

            decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, lj_tensor)
            loss = criterion(decoder_output.transpose(2, 1), lj_tensor)

            nb_tokens += sum([i for e in profiles_len for i in e])

            # loss = criterion(lj_approximation, enc_lj.unsqueeze(1))
            loss_list.append(loss.item())

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_perplexity": 2 ** (sum(loss_list) / nb_tokens)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tf", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=.5)
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="s2s_agg")
    parser.add_argument("--enc_model", type=str,
                        default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_enc_best_ep_92")
    parser.add_argument("--dec_model", type=str,
                        default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_dec_best_ep_92")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

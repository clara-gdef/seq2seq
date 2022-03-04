import argparse
import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import DecoderLSTM, EncoderBiLSTM
import os
import pickle as pkl
import torch
from utils.Utils import collate_for_jobs, save_best_model, model_checkpoint
from datetime import datetime
import glob
from torch.utils.data import DataLoader, Subset
from utils.JobDataset import JobDataset
import random
import math
import ipdb
import itertools


def main(args):
    xp_title = "tf" + str(args.tf) + " decay lr" + str(args.lr) + " hs" + str(args.hidden_size) + " bs" + str(
        args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    dataset_train = JobDataset(datadict["train"])
    dataset_valid = JobDataset(datadict["valid"])
    del datadict

    dimension = 100

    hidden_size = args.hidden_size
    num_layers = 1

    decoder = DecoderLSTM(embeddings, hidden_size, num_layers, dimension, embeddings.size(0), args.MAX_CAREER_LENGTH)
    encoder = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)

    actual_params_e = filter(lambda p: p.requires_grad, encoder.parameters())
    actual_params_d = filter(lambda p: p.requires_grad, decoder.parameters())
    optim = torch.optim.Adam(itertools.chain(actual_params_e, actual_params_d), lr=args.lr, weight_decay=1e-5)

    # # for efficient memory save
    del embeddings

    if args.from_trained_model:
        enc_weights = os.path.join(args.model_dir, args.enc_model)
        dec_weights = os.path.join(args.model_dir, args.dec_model)
        encoder.load_state_dict(torch.load(enc_weights))
        decoder.load_state_dict(torch.load(dec_weights))
        optim_weights = os.path.join(args.model_dir, args.optim)
        optim.load_state_dict(torch.load(optim_weights))

    enc = encoder.cuda()
    dec = decoder.cuda()

    dictionary = main_for_one_split(args, enc, dec, dataset_train, dataset_valid, index, optim)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/splitless/" + args.model_type + "_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder, decoder, dataset_train, dataset_valid, vocab_index, optim):
    res_epoch = {}

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    best_val_loss = 1e+300

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate_for_jobs,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate_for_jobs,
                                  shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, encoder, decoder, criterion,
                                  args, best_val_loss, optim, vocab_index, dataloader_train, dataloader_valid)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']
    return res_epoch


@quiviz.log
def main_for_one_epoch(epoch, encoder, decoder, criterion,
                       args, best_val_loss, optim, vocab_index, dataloader_train, dataloader_valid):
    print("Training and validating for epoch " + str(epoch))

    if epoch == 1:
        print("INITIAL LOSS: " + str(valid(encoder, decoder, dataloader_valid, criterion, vocab_index)))

    train_loss = train(args, encoder, decoder, dataloader_train, criterion, optim, vocab_index, epoch)

    valid_loss = valid(encoder, decoder, dataloader_valid, criterion, vocab_index)

    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_model(args, epoch, args.model_dir, encoder, decoder, optim)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint(args, epoch, args.model_dir, encoder, decoder, optim)
        dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(args, encoder, decoder, dataloader_train, criterion, optim, vocab_index, ep):
    enforce_sorted = True
    nb_tokens = 0
    loss_list = []
    # optim.param_groups[0]['lr'] = args.lr * ((1 - float(ep) / float(args.epoch)) ** 0.9)
    # with ipdb.launch_ipdb_on_exception():
    with tqdm(dataloader_train, desc="Training...") as pbar:
        for job, seq_length in pbar:
            b_size = len(job)

            optim.zero_grad()
            loss = 0
            job_tensor = torch.zeros(b_size, max(seq_length), dtype=torch.int64)
            for i in range(b_size):
                job_tensor[i][:seq_length[i]] = torch.LongTensor(job[i])

            job_tensor = job_tensor.cuda()

            # train the encoder
            enc_output, attention, encoder_hidden = encoder(job_tensor, seq_length, enforce_sorted)

            decoder_hidden = (encoder_hidden[0].view(1, b_size, -1), encoder_hidden[1].view(1, b_size, -1))

            weighted_rep = enc_output.unsqueeze(1).expand(b_size, max(seq_length), enc_output.shape[-1])
            decoder_output, decoder_hidden = decoder(weighted_rep, decoder_hidden, job_tensor)
            loss += criterion(decoder_output.transpose(2, 1), job_tensor)

            nb_tokens += sum(seq_length)

            loss_list.append(loss.item())
            # ipdb.set_trace()

            loss.backward()

            actual_params_e = filter(lambda p: p.requires_grad, encoder.parameters())
            actual_params_d = filter(lambda p: p.requires_grad, decoder.parameters())

            torch.nn.utils.clip_grad_norm_(itertools.chain(actual_params_e, actual_params_d), 1)
            optim.step()
    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_perplexity": math.exp((sum(loss_list) / nb_tokens))}


def valid(encoder, decoder, dataloader_valid, criterion, vocab_index):
    enforce_sorted = True
    loss_list = []

    nb_tokens = 0
    with tqdm(dataloader_valid, desc="Validating...") as pbar:
        for job, seq_length in pbar:
            b_size = len(job)
            loss = 0
            job_tensor = torch.zeros(b_size, max(seq_length), dtype=torch.int64)
            for i in range(b_size):
                job_tensor[i][:seq_length[i]] = torch.LongTensor(job[i])

            job_tensor = job_tensor.cuda()
            enc_output, attention, encoder_hidden = encoder(job_tensor, seq_length, enforce_sorted)

            decoder_hidden = (encoder_hidden[0].view(1, b_size, -1), encoder_hidden[1].view(1, b_size, -1))

            weighted_rep = enc_output.unsqueeze(1).expand(b_size, max(seq_length), enc_output.shape[-1])
            decoder_output, decoder_hidden = decoder(weighted_rep, decoder_hidden, job_tensor)
            loss += criterion(decoder_output.transpose(2, 1), job_tensor)

            nb_tokens += sum(seq_length)

            loss_list.append(loss.item())

            pbar.update(1)
    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_perplexity": math.exp((sum(loss_list) / nb_tokens))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/indices_jobs.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="s2s_hard_decay")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--enc_model", type=str, default=None)
    parser.add_argument("--dec_model", type=str, default=None)
    parser.add_argument("--tf", type=float, default=1)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

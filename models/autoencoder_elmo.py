import argparse
import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import DecoderWithElmo, EncoderWithElmo
import os
import pickle as pkl
import torch
from utils.Utils import collate_for_jobs_elmo, save_best_model_elmo, model_checkpoint_elmo
from datetime import datetime
from torch.utils.data import DataLoader
from utils.JobDatasetElmo import JobDatasetElmo

import random
from allennlp.modules.elmo import Elmo
import ipdb
import itertools
from line_profiler import LineProfiler


def main(args):
    xp_title = "ELMO s2s lr" + str(args.lr) + " hs" + str(args.dec_hidden_size) + " bs" + str(args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    dataset_train = JobDatasetElmo(datadict["train_data"])
    dataset_valid = JobDatasetElmo(datadict["valid_data"])
    del datadict

    hidden_size = args.dec_hidden_size
    num_layers = 1
    elmo_dimension = 1024

    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("Initializing ELMo...")
    elmo = Elmo(options_file, weight_file, 2, requires_grad=False, dropout=0)
    print("ELMo ready.")
    encoder = EncoderWithElmo(elmo, elmo_dimension, args.batch_size)
    decoder = DecoderWithElmo(elmo, elmo_dimension, hidden_size, num_layers, len(index))

    actual_params_e = filter(lambda p: p.requires_grad, encoder.parameters())
    actual_params_d = filter(lambda p: p.requires_grad, decoder.parameters())

    prev_epochs = 0

    if args.from_trained_model:
        prev_epochs = int(str.split(args.dec_model, sep='_')[-1])
        dec_weights = os.path.join(args.model_dir, args.dec_model)
        decoder.load_state_dict(torch.load(dec_weights))
        enc = encoder.cuda()
        dec = decoder.cuda()
        optim = torch.optim.Adam(itertools.chain(actual_params_e, actual_params_d), lr=args.lr)
        optim_weights = os.path.join(args.model_dir, args.optim)
        optim.load_state_dict(torch.load(optim_weights))
    else:
        optim = torch.optim.Adam(itertools.chain(actual_params_e, actual_params_d), lr=args.lr)
        enc = encoder.cuda()
        dec = decoder.cuda()

    dictionary = main_for_one_split(args, enc, dec, dataset_train, dataset_valid, index, optim, prev_epochs)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/splitless/" + args.model_type + "_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder, decoder, dataset_train, dataset_valid, vocab_index, optim, prev_epochs):
    res_epoch = {}

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = 1e+300

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate_for_jobs_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate_for_jobs_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, encoder, decoder, criterion,
                                  args, best_val_loss, optim, vocab_index, dataloader_train, dataloader_valid, prev_epochs)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']
    return res_epoch


@quiviz.log
def main_for_one_epoch(epoch, encoder, decoder, criterion,
                       args, best_val_loss, optim, vocab_index, dataloader_train, dataloader_valid, prev_epochs):
    epoch += prev_epochs
    print("Training and validating for epoch " + str(epoch))

    # if epoch == 1:
    #    print("INITIAL LOSS: " + str(valid(encoder, decoder, dataloader_valid, criterion, vocab_index)))

    train_loss = train(args, encoder, decoder, dataloader_train, criterion, optim, vocab_index, epoch)

    valid_loss = valid(encoder, decoder, dataloader_valid, criterion, vocab_index)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_model_elmo(args, epoch, target_dir, decoder, optim)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_elmo(args, epoch, target_dir, decoder, optim)
        dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(args, encoder, decoder, dataloader_train, criterion, optim, vocab_index, ep):
    nb_tokens = 0
    loss_list = []
    reversed_index = {v: k for k, v in vocab_index.items()}
    # optim.param_groups[0]['lr'] = args.lr * ((1 - float(ep) / float(args.epoch)) ** 0.9)
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_train, desc="Training...") as pbar:
            for job, seq_length, indices in pbar:

                optim.zero_grad()
                b_size = len(job)

                # encode the profile
                enc_output = encoder(job)
                job_rep = torch.mean(enc_output, dim=1)

                # turn words to indices for decoding
                profile_max_len = max([len(e[0]) for e in indices])
                indices_tensor = torch.zeros(b_size, profile_max_len).cuda()
                for e in range(b_size):
                    indices_tensor[e, :len(indices[e][0])] = torch.LongTensor(indices[e][0]).cuda()
                indices_tensor = indices_tensor.type(torch.LongTensor).cuda()

                # initialize the hidden state of the decoder
                h_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                c_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                decoder_hidden = (h_0, c_0)

                job_rep_transformed = job_rep.unsqueeze(1).expand(b_size, profile_max_len, 1024)
                tokens = turn_indices_to_words(indices_tensor, reversed_index)
                decoder_output, decoder_hidden = decoder(job_rep_transformed, decoder_hidden, tokens)
                decoder_output = torch.transpose(decoder_output, 2, 1)
                loss = criterion(decoder_output, indices_tensor)

                nb_tokens += sum(seq_length)

                if torch.isnan(loss).any():
                    ipdb.set_trace()
                else:
                    loss_list.append(loss.item())
                    loss.backward()
                    optim.step()

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_perplexity": 2 ** (sum(loss_list) * b_size / nb_tokens)}


def valid(encoder, decoder, dataloader_valid, criterion, vocab_index):
    loss_list = []
    reversed_index = {v: k for k, v in vocab_index.items()}
    nb_tokens = 0
    # with ipdb.launch_ipdb_on_exception():
    with tqdm(dataloader_valid, desc="Validating...") as pbar:
        for job, seq_length, indices in pbar:
            b_size = len(job)

            # encode the profile
            enc_output = encoder(job)
            job_rep = torch.mean(enc_output, dim=1)

            # turn words to indices for decoding
            profile_max_len = max([len(e[0]) for e in indices])
            indices_tensor = torch.zeros(b_size, profile_max_len).cuda()
            for e in range(b_size):
                indices_tensor[e, :len(indices[e][0])] = torch.LongTensor(indices[e][0]).cuda()
            indices_tensor = indices_tensor.type(torch.LongTensor).cuda()

            # initialize the hidden state of the decoder
            h_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
            c_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
            decoder_hidden = (h_0, c_0)

            job_rep_transformed = job_rep.unsqueeze(1).expand(b_size, profile_max_len, 1024)
            tokens = turn_indices_to_words(indices_tensor, reversed_index)
            decoder_output, decoder_hidden = decoder(job_rep_transformed, decoder_hidden, tokens)
            decoder_output = torch.transpose(decoder_output, 2, 1)
            loss = criterion(decoder_output, indices_tensor)

            nb_tokens += sum(seq_length)

            if torch.isnan(loss).any():
                ipdb.set_trace()
            else:
                loss_list.append(loss.item())


    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_perplexity": 2 ** (sum(loss_list) * b_size / nb_tokens)}


def turn_indices_to_words(indices, reversed_index):
    word_list = []
    for sample in indices:
        tmp = []
        for i in sample:
            tmp.append(reversed_index[i.item()])
        word_list.append(tmp)
    return word_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/jobs_elmo.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    # parser.add_argument("--emb_file", type=str, default="pkl/tensor_vocab_3j40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=180)
    parser.add_argument("--dec_hidden_size", type=int, default=512)
    parser.add_argument("--model_type", type=str, default="s2s_elmo")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--dec_model", type=str, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--tf", type=float, default=1)
    parser.add_argument("--dpo", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

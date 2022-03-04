import argparse
import itertools

import quiviz
from allennlp.modules import Elmo
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import CareerEncoderLSTM, MLP, DecoderWithElmo
import os
import pickle as pkl
import torch

from utils import ProfileDatasetElmo
from utils.Utils import collate_profiles_lj_elmo, transform_for_elmo_lj, \
    labels_to_indices, save_best_model_mlp_rnn, model_checkpoint_mlp_rnn
from datetime import datetime
from torch.utils.data import DataLoader
import ipdb


def init(args):
    # loading data
    print("Loading data...")
    train_file = os.path.join(args.DATA_DIR, args.train_file)
    datadict_train = {"data": []}
    flag_err = False

    with open(train_file, "rb") as f:
        while not flag_err:
            try:
                datadict_train["data"].append(pkl.load(f))
            except EOFError:
                flag_err = True
                continue
    print("Train file loaded.")
    valid_file = os.path.join(args.DATA_DIR, args.valid_file)
    datadict_valid = {"data": []}
    flag_err = False
    with open(valid_file, "rb") as f:
        while not flag_err:
            try:
                datadict_valid["data"].append(pkl.load(f))
            except EOFError:
                flag_err = True
                continue
    print("Valid file loaded.")

    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Data loaded.")

    dataset_train = ProfileDatasetElmo(datadict_train, transform_for_elmo_lj)
    dataset_valid = ProfileDatasetElmo(datadict_valid, transform_for_elmo_lj)
    del datadict_train, datadict_valid

    dec_hs = str.split(args.dec_model, sep="_")[6]
    dec_lr = str.split(args.dec_model, sep="_")[3]
    # dec_type = str.split(args.dec_model, sep="_")[0]
    dec_ep = str.split(args.dec_model, sep="_")[-1]

    num_layers = 1
    elmo_dimension = 1024

    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("Initializing ELMo...")
    elmo = Elmo(options_file, weight_file, 2, requires_grad=False, dropout=0)
    print("ELMo ready.")

    decoder = DecoderWithElmo(elmo, elmo_dimension, int(dec_hs), num_layers, len(index)).cuda()
    dec_weights = os.path.join("/net/big/gainondefor/work/trained_models/seq2seq/elmo/", args.dec_model)
    decoder.load_state_dict(torch.load(dec_weights))

    ## profile RNN
    rnn = CareerEncoderLSTM(elmo_dimension, int(dec_hs), 1, args.dpo, args.bidirectional)
    rnn = rnn.cuda()

    ## career dynamic mlp
    mlp = MLP(int(dec_hs) * 2, elmo_dimension)
    mlp = mlp.cuda()

    optim = torch.optim.Adam(itertools.chain(rnn.parameters(), mlp.parameters()), lr=args.lr)

    prev_epochs = 0

    if args.from_trained_model:
        prev_epochs = int(str.split(args.dec_model, sep='_')[-1])
        rnn_weights = os.path.join(args.model_dir, args.rnn_model)
        rnn.load_state_dict(torch.load(rnn_weights))
        mlp_weights = os.path.join(args.model_dir, args.mlp_model)
        mlp.load_state_dict(torch.load(mlp_weights))
        optim_weights = os.path.join(args.model_dir, args.optim)
        optim.load_state_dict(torch.load(optim_weights))

    return dataset_train, dataset_valid, rnn, mlp, decoder, index, dec_hs, dec_lr, dec_ep, prev_epochs, optim


def main(args):
    dataset_train, dataset_valid, rnn, mlp, decoder_job, index, dec_hs, dec_lr, dec_ep, prev_epochs, optim = init(args)

    xp_title = "mlp_rnn bs" + str(args.batch_size) + " hs" + str(dec_hs) + " lr" + str(args.lr)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    dictionary = main_for_one_split(args, decoder_job, rnn, mlp, dataset_train, dataset_valid, index, dec_hs, dec_lr,
                                    dec_ep, prev_epochs, optim)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/elmo/" + args.model_type + "_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, decoder_job, rnn, mlp, dataset_train, dataset_valid, index, dec_hs, dec_lr, dec_ep,
                       prev_epochs, optim):
    res_epoch = {}

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    best_val_loss = 1e+300

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, decoder_job, rnn, mlp, criterion,
                                  args, best_val_loss, optim, dataloader_train,
                                  dataloader_valid, index, dec_hs, dec_lr, dec_ep, prev_epochs)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']
    return res_epoch


@quiviz.log
def main_for_one_epoch(epoch, decoder, rnn, mlp, criterion, args, best_val_loss, optim, dataloader_train,
                       dataloader_valid, index, dec_hs, dec_lr, dec_ep, prev_epochs):
    print("Training and validating for epoch " + str(epoch + prev_epochs))

    train_loss = train(decoder, rnn, mlp, dataloader_train, criterion, optim, int(dec_hs), epoch, index)

    valid_loss = valid(decoder, rnn, mlp, dataloader_valid, criterion, int(dec_hs), index)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_model_mlp_rnn(args, epoch, target_dir, decoder, rnn, mlp, optim, dec_hs, dec_lr, dec_ep)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_mlp_rnn(args, epoch, target_dir, decoder, rnn, mlp, optim, dec_hs, dec_lr, dec_ep)
        best_val_loss = valid_loss['valid_loss']

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(decoder, rnn, mlp, dataloader_train, criterion, optim, hidden_size, ep, index):
    loss_list = []
    elmo_dimension = 1024
    rnn.train()
    decoder.train()
    mlp.train()
    nb_tokens = 0
    # optim.param_groups[0]['lr'] = args.lr * ((1 - float(ep) / float(args.epoch)) ** 0.9)
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_train, desc="Training...") as pbar:
            for ids, jobs, career_len, last_jobs, last_jobs_len in pbar:
                b_size = len(ids)

                optim.zero_grad()

                max_seq_length = max(career_len)
                profile_tensor = torch.zeros(b_size, max_seq_length, elmo_dimension)

                for person in range(b_size):
                    profile_tensor[person, :len(jobs[person]), :] = torch.cat(jobs[person])
                prof_tensor = profile_tensor.cuda()
                z_people, hidden_state = rnn(prof_tensor, list(career_len), enforce_sorted=False)

                tmp = mlp(z_people)
                lj_app = tmp.expand(b_size, max(last_jobs_len), elmo_dimension)

                lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
                for i in range(len(last_jobs)):
                    lj_tensor[i, :len(last_jobs[i])] = labels_to_indices(last_jobs[i], index)

                decoder_hidden = (
                torch.zeros(1, b_size, hidden_size).cuda(), torch.zeros(1, b_size, hidden_size).cuda())
                decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, last_jobs)
                loss = criterion(decoder_output.transpose(2, 1), lj_tensor)

                nb_tokens += sum([i for i in last_jobs_len])

                loss_list.append(loss.item())
                loss.backward()
                optim.step()

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_perplexity": 2 ** (sum(loss_list) / nb_tokens)}


def valid(decoder, rnn, mlp, dataloader_valid, criterion, hidden_size, index):
    loss_list = []
    elmo_dimension = 1024
    decoder.eval()
    rnn.eval()
    mlp.eval()
    nb_tokens = 0
    with tqdm(dataloader_valid, desc="Validating...") as pbar:
        for ids, jobs, career_len, last_jobs, last_jobs_len in pbar:
            # try:
            b_size = len(ids)

            max_seq_length = max(career_len)
            profile_tensor = torch.zeros(b_size, max_seq_length, elmo_dimension)

            for person in range(b_size):
                profile_tensor[person, :len(jobs[person]), :] = torch.cat(jobs[person])
            prof_tensor = profile_tensor.cuda()
            z_people, hidden_state = rnn(prof_tensor, list(career_len), enforce_sorted=False)

            tmp = mlp(z_people)
            lj_app = tmp.expand(b_size, max(last_jobs_len), elmo_dimension)

            lj_tensor = torch.zeros(len(last_jobs), max(last_jobs_len), dtype=torch.int64).cuda()
            for i in range(len(last_jobs)):
                lj_tensor[i, :len(last_jobs[i])] = labels_to_indices(last_jobs[i], index)

            decoder_hidden = (torch.zeros(1, b_size, hidden_size).cuda(), torch.zeros(1, b_size, hidden_size).cuda())
            decoder_output, decoder_hidden = decoder(lj_app, decoder_hidden, last_jobs)
            loss = criterion(decoder_output.transpose(2, 1), lj_tensor)

            nb_tokens += sum([i for i in last_jobs_len])

            loss_list.append(loss.item())

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_perplexity": 2 ** (sum(loss_list) / nb_tokens)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="pkl/prof_rep_elmo_train_lj_cpu.pkl")
    parser.add_argument("--valid_file", type=str, default="pkl/prof_rep_elmo_valid_lj_cpu.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tf", type=float, default=0.)
    parser.add_argument("--dpo", type=float, default=.5)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--mlp_model", type=str, default=None)
    parser.add_argument("--rnn_model", type=str, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="elmo_agg")
    parser.add_argument("--dec_model", type=str,
                        default="s2s_elmo_bs128_lr0.001_tf1_hs_256_max_ep_300_40k_dec_best_ep_19")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/mlp_rnn')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

import argparse
import itertools
import os
import pickle as pkl
import torch

from allennlp.modules.elmo import Elmo
from quiviz import quiviz
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader
from torch.autograd import gradcheck
from tqdm import tqdm

from classes import CareerEncoderRNN, CareerDecoderRNN
from utils import ProfileDatasetElmo
from utils.Utils import transform_for_elmo_lj, collate_profiles_lj_elmo, save_best_model_elmo_w2v, \
    model_checkpoint_elmo_w2v
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
    print("Data loaded.")

    dataset_train = ProfileDatasetElmo(datadict_train, transform_for_elmo_lj)
    dataset_valid = ProfileDatasetElmo(datadict_valid, transform_for_elmo_lj)
    del datadict_train, datadict_valid

    hidden_size = args.hidden_size
    elmo_size = 1024

    encoder_career = CareerEncoderRNN(elmo_size, hidden_size, 1, args.dpo, args.bidirectional).cuda()
    decoder_career = CareerDecoderRNN(elmo_size, hidden_size * 2, 1, args.dpo).cuda()
    optim = torch.optim.SGD(itertools.chain(encoder_career.parameters(), decoder_career.parameters()), lr=args.lr)

    prev_epochs = 0

    if args.from_trained_model:
        prev_epochs = int(str.split(args.dec_model, sep='_')[-1])
        enc_weights = os.path.join(args.model_dir, args.enc_model)
        encoder_career.load_state_dict(torch.load(enc_weights))
        dec_weights = os.path.join(args.model_dir, args.dec_model)
        decoder_career.load_state_dict(torch.load(dec_weights))
        optim_weights = os.path.join(args.model_dir, args.optim)
        optim.load_state_dict(torch.load(optim_weights))

    return dataset_train, dataset_valid, encoder_career, decoder_career, optim, prev_epochs


def main(args):
    xp_title = "w2v rnn lr" + str(args.lr) + " hs" + str(args.hidden_size) + " bs" + str(
        args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    dataset_train, dataset_valid, encoder_career, decoder_career, optim, prev_epochs = init(args)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_lj_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)
    res_epoch = {}
    criterion = torch.nn.MSELoss(reduction="mean")

    best_val_loss = float('Inf')

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, encoder_career, decoder_career, criterion,
                                  args, best_val_loss, optim, dataloader_train, dataloader_valid, prev_epochs)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']

@quiviz.log
def main_for_one_epoch(epoch, encoder_career, decoder_career, criterion,
                       args, best_val_loss, optim, dataloader_train, dataloader_valid, prev_epochs):
    epoch += prev_epochs
    print("Training and validating for epoch " + str(epoch))

    train_loss = train(args, encoder_career, decoder_career, dataloader_train, criterion, optim, epoch)
    valid_loss = valid(encoder_career, decoder_career, dataloader_valid, criterion)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_model_elmo_w2v(args, epoch, target_dir, encoder_career, decoder_career, optim)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_elmo_w2v(args, epoch, target_dir, encoder_career, decoder_career, optim)
        dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(args, encoder_career, decoder_career, dataloader_train, criterion, optim, epoch):
    b_size = args.batch_size
    elmo_dimension = 1024
    loss_list = []
    with ipdb.launch_ipdb_on_exception():
        for ids, jobs, career_len, lj, lj_len in tqdm(dataloader_train):
            optim.zero_grad()

            loss = 0
            max_seq_length = max(career_len)
            profile_tensor = torch.zeros(b_size, max_seq_length, elmo_dimension)

            for person in range(b_size):
                profile_tensor[person, :len(jobs[person]), :] = torch.cat(jobs[person])
            prof_tensor = profile_tensor.cuda()
            z_people, hidden_state = encoder_career(prof_tensor, list(career_len), enforce_sorted=False)

            h_0 = torch.zeros(decoder_career.num_layers, b_size, decoder_career.hidden_size).cuda()

            prev_job = torch.zeros(b_size, 1, elmo_dimension).cuda()

            for i in range(max_seq_length):
                next_job, h_0 = decoder_career(z_people, h_0, prev_job)
                prev_job = next_job
                loss += criterion(next_job, prof_tensor[:, i, :].unsqueeze(1))

            loss_list.append(loss.item())
            loss.backward()

            actual_params_e = filter(lambda p: p.requires_grad, encoder_career.parameters())
            actual_params_d = filter(lambda p: p.requires_grad, decoder_career.parameters())

            torch.nn.utils.clip_grad_norm_(itertools.chain(actual_params_e, actual_params_d), 1)

            optim.step()

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item()}


def valid(encoder_career, decoder_career, dataloader_valid, criterion):
    b_size = args.batch_size
    elmo_dimension = 1024
    loss_list = []
    with ipdb.launch_ipdb_on_exception():
        for ids, jobs, career_len, lj, lj_len in tqdm(dataloader_valid):
            loss = 0
            max_seq_length = max(career_len)
            profile_tensor = torch.zeros(b_size, max_seq_length, elmo_dimension).cuda()

            for person in range(b_size):
                profile_tensor[person, :len(jobs[person]), :] = torch.cat(jobs[person]).cuda()
            z_people, hidden_state = encoder_career(profile_tensor, list(career_len), enforce_sorted=False)

            h_0 = torch.zeros(decoder_career.num_layers, b_size, decoder_career.hidden_size).cuda()

            prev_job = torch.zeros(b_size, 1, elmo_dimension).cuda()

            # without teacher forcing
            for i in range(max_seq_length):
                next_job, h_0 = decoder_career(z_people, h_0, prev_job)
                prev_job = next_job
                loss += criterion(next_job, profile_tensor[:, i, :].unsqueeze(1))

            loss_list.append(loss.item())

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="pkl/prof_rep_elmo_train_cpu.pkl")
    parser.add_argument("--valid_file", type=str, default="pkl/prof_rep_elmo_valid_cpu.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--model_type", type=str, default="elmo_w2v_rnn")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--dec_model", type=str, default=None)
    parser.add_argument("--enc_model", type=str, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tf", type=float, default=0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_dir", type=str,
                        default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/elmo_w2v')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--dpo", type=float, default=.0)
    parser.add_argument("--bidirectional", type=bool, default=True)
    args = parser.parse_args()
    main(args)


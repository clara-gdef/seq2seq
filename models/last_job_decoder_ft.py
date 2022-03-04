import argparse
import math

import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import DecoderWithFT
import os
import pickle as pkl
import torch
from utils.Utils import save_best_model_elmo, model_checkpoint_elmo
from datetime import datetime
from torch.utils.data import DataLoader
import ipdb
import itertools


def main(args):
    suffix = str(args.model_type)[:2]
    xp_title = "FT " + args.model_type + " dec bs" + str(args.batch_size) + " tf " + str(args.tf)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    print("Loading data...")
    data_train = []
    data_valid = []
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep2_lj_ft_" + suffix + "_train.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_train.append(data)
            except EOFError:
                flag_err = True
                continue
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep2_lj_ft_" + suffix + "_valid.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_valid.append(data)
            except EOFError:
                flag_err = True
                continue
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)
    print("Data loaded.")

    hidden_size = args.dec_hidden_size
    num_layers = 1

    decoder = DecoderWithFT(int(args.emb_size), hidden_size, num_layers, len(index))

    prev_epochs = 0

    if args.from_trained_model:
        prev_epochs = int(str.split(args.dec_model, sep='_')[-1])
        dec_weights = os.path.join(args.model_dir, args.dec_model)
        decoder.load_state_dict(torch.load(dec_weights))
        dec = decoder.cuda()
        optim = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        optim_weights = os.path.join(args.model_dir, args.optim)
        optim.load_state_dict(torch.load(optim_weights))
    else:
        optim = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        dec = decoder.cuda()

    dictionary = main_for_one_split(args, dec, data_train, data_valid, index, optim, prev_epochs)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/esann20/" + args.model_type + "_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, decoder, dataset_train, dataset_valid, vocab_index, optim, prev_epochs):
    res_epoch = {}

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    best_val_loss = 1e+300

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, decoder, criterion,
                                  args, best_val_loss, optim, vocab_index, dataloader_train, dataloader_valid, prev_epochs)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']
    return res_epoch


@quiviz.log
def main_for_one_epoch(epoch, decoder, criterion,
                       args, best_val_loss, optim, vocab_index, dataloader_train, dataloader_valid, prev_epochs):
    epoch += prev_epochs
    if epoch == 1:
        print("Initial Validation")
        valid_loss = valid(decoder, dataloader_valid, criterion, vocab_index)
        print(valid_loss)

    print("Training and validating for epoch " + str(epoch))

    train_loss = train(args, decoder, dataloader_train, criterion, optim, vocab_index, epoch)

    valid_loss = valid(decoder, dataloader_valid, criterion, vocab_index)

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


def train(args, decoder, dataloader_train, criterion, optim, vocab_index, ep):
    nb_tokens = 0
    loss_list = []
    rev_index = {v: k for k, v in vocab_index.items()}
    pred_file = os.path.join(args.DATA_DIR, "debug_ft_" + args.model_type + ".txt")
    with open(pred_file, 'a') as f:
        f.write("EPOCH ================================================== " + str(ep) + "\n")
        with ipdb.launch_ipdb_on_exception():
            with tqdm(dataloader_train, desc="Training...") as pbar:
                for ids, profile, lj_indices in pbar:
                        optim.zero_grad()
                        b_size = len(ids)

                        # turn words to indices for decoding
                        lj_max_len = min(max([len(e) for e in lj_indices]), args.MAX_SEQ_LENGTH)
                        lj_indices_tensor = torch.zeros(b_size, lj_max_len).cuda()
                        for e in range(b_size):
                            lj_indices_tensor[e, :len(lj_indices[e])] = torch.LongTensor(lj_indices[e]).cuda()
                        lj_indices_tensor = lj_indices_tensor.type(torch.LongTensor).cuda()

                        # initialize the hidden state of the decoder
                        h_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                        c_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                        decoder_hidden = (h_0, c_0)

                        profile_tensor = torch.stack(profile)

                        job_rep_transformed = profile_tensor.unsqueeze(1).expand(b_size, lj_max_len-1, args.emb_size).cuda()

                        decoded_tokens = []

                        if args.tf == 1:
                            decoder_output, decoder_hidden = decoder(job_rep_transformed, decoder_hidden, lj_indices_tensor[:, :-1])
                            decoded_tokens.append(decoder_output.argmax(-1))
                            decoder_output = torch.transpose(decoder_output, 2, 1)
                            loss = criterion(decoder_output, lj_indices_tensor[:, 1:]) / b_size

                        else:
                            max_seq_len = max([len(e) for e in lj_indices])
                            tokens = torch.LongTensor(b_size, max_seq_len, 1)
                            for batch in b_size:
                                tokens[batch, :, :] = vocab_index["SOT"]
                            for i in range(max_seq_len):
                                targets = lj_indices_tensor[:, i]
                                decoder_output, decoder_hidden = decoder(job_rep_transformed, decoder_hidden, tokens)
                                dec_tokens = decoder_output.argmax(-1)
                                loss += criterion(torch.transpose(dec_tokens, 2, 1), targets)
                                if args.tf <= math.random():
                                    tokens = targets
                                else:
                                    tokens = torch.LongTensor(dec_tokens)

                        nb_tokens += sum([len(e) for e in lj_indices])

                        loss_list.append(loss.item())
                        loss.backward()
                        optim.step()

        f.write("PREDICTION =================================== \n")
        for word in decoded_tokens[0][0]:
            f.write(rev_index[word.item()] + ' ')
        f.write("\n")
        f.write("LABEL ======================================== \n")
        for word in lj_indices[0]:
            f.write(rev_index[word] + ' ')
        f.write("\n")
    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_perplexity": 2 ** ((sum(loss_list) * b_size / nb_tokens) / math.log(2))}


def valid(decoder, dataloader_valid, criterion, vocab_index):
    loss_list = []
    nb_tokens = 0
    rev_index = {v: k for k, v in vocab_index.items()}
    # with ipdb.launch_ipdb_on_exception():
    pred_file = os.path.join(args.DATA_DIR, "debug_ft_" + args.model_type + ".txt")
    with open(pred_file, 'a') as f:
        f.write("VALID ============================================= \n")
        with tqdm(dataloader_valid, desc="Validating...") as pbar:
            for ids, profile, lj_indices in pbar:

                    b_size = len(ids)

                    # turn words to indices for decoding
                    lj_max_len = max([len(e) for e in lj_indices])
                    lj_indices_tensor = torch.zeros(b_size, lj_max_len).cuda()
                    for e in range(b_size):
                        lj_indices_tensor[e, :len(lj_indices[e])] = torch.LongTensor(lj_indices[e]).cuda()
                    lj_indices_tensor = lj_indices_tensor.type(torch.LongTensor).cuda()

                    # initialize the hidden state of the decoder
                    h_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                    c_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                    decoder_hidden = (h_0, c_0)

                    profile_tensor = torch.stack(profile)
                    job_rep_transformed = profile_tensor.unsqueeze(1).expand(b_size, lj_max_len-1, args.emb_size).cuda()

                    decoded_tokens = []
                    decoder_output, decoder_hidden = decoder(job_rep_transformed, decoder_hidden,
                                                             lj_indices_tensor[:, :-1])
                    decoded_tokens.append(decoder_output.argmax(-1))
                    decoder_output = torch.transpose(decoder_output, 2, 1)
                    loss = criterion(decoder_output, lj_indices_tensor[:, 1:]) / b_size

                    nb_tokens += sum([len(e) for e in lj_indices])

                    loss_list.append(loss.item())
        f.write("PREDICTION =================================== \n")
        for word in decoded_tokens[0][0]:
            f.write(rev_index[word.item()] + ' ')
        f.write("\n")
        f.write("LABEL ======================================== \n")
        for word in lj_indices[0]:
            f.write(rev_index[word] + ' ')
        f.write("\n")

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_perplexity": 2 ** ((sum(loss_list) * b_size / nb_tokens) / math.log(2))}


def collate(batch):
    tmp = list(zip(*batch))
    ids, profiles, lj_ind = tmp[0], tmp[1], tmp[2]
    identifiers, prof, lj_indices = zip(*sorted(zip(ids, profiles, lj_ind), key=lambda item: item[2], reverse=True))
    return list(identifiers), list(prof), list(lj_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--dec_hidden_size", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="fs2")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--dec_model", type=str, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--tf", type=float, default=1)
    parser.add_argument("--dpo", type=float, default=.5)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20/job_dec')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

import argparse
import math

import quiviz
from allennlp.modules import Elmo
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import DecoderWithElmo, DecoderWithFT
import os
import pickle as pkl
import torch

from utils.Utils import save_best_model_elmo, model_checkpoint_elmo
from datetime import datetime
from torch.utils.data import DataLoader
import ipdb


def main(args):
    enc_type = str.split(args.enc_model, sep="_")[1]
    enc_hs = str.split(args.enc_model, sep="_")[6]
    enc_lr = str.split(args.enc_model, sep="_")[3]
    enc_ep = str.split(args.enc_model, sep="_")[-1]

    xp_title = "ELMO LSTM job dec " + enc_type + " bs" + str(args.batch_size) + " " + enc_lr + " hs" + enc_hs
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    with open(os.path.join(args.DATA_DIR, "train_lj.pkl"), "rb") as f:
        data_train = pkl.load(f)
    with open(os.path.join(args.DATA_DIR, "valid_lj.pkl"), "rb") as f:
        data_valid = pkl.load(f)
    print("Data loaded.")

    hidden_size = args.dec_hidden_size
    num_layers = 1
    elmo_dimension = 1024

    # options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # print("Initializing ELMo...")
    # elmo = Elmo(options_file, weight_file, 2, requires_grad=False, dropout=0)
    # print("ELMo ready.")
    # decoder = DecoderWithElmo(elmo, elmo_dimension, hidden_size, num_layers, len(index))
    decoder = DecoderWithFT(elmo_dimension, hidden_size, num_layers, len(index))
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
    print("Training and validating for epoch " + str(epoch))
    #
    # if epoch == 1:
    #     print("INITIAL LOSS: " + str(valid(decoder, dataloader_valid, criterion, vocab_index)))

    train_loss = train(decoder, dataloader_train, criterion, optim, vocab_index)

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


def train(dec, dataloader_train, criterion, optim, vocab_index):
    rev_index = {v: k for k, v in vocab_index.items()}
    nb_tokens = 0
    loss_list = []
    debug_file = os.path.join(args.DATA_DIR, "debug_elmo_LSTM.txt")
    with open(debug_file, 'a') as f:
        f.write("TRAINING ===================================================== \n")
        with ipdb.launch_ipdb_on_exception():
            for ids, tensors, indices, words in tqdm(dataloader_train, desc="Training..."):
                # if len(loss_list) < 100:
                optim.zero_grad()

                b_size = len(ids)

                # initialize the hidden state of the decoder
                h_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
                c_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
                dec_hidden = (h_0, c_0)

                predictions = torch.cat(tensors).cuda()
                pred_transformed = predictions.expand(b_size, max([len(e) for e in words]) - 1, 1024)

                # words_wo_eod = [e[:-1] for e in words]
                # max_len = max([len(e) for e in words_wo_eod])
                # for i in range(max_len):

                truth = torch.zeros(b_size, max([len(e) for e in indices])).type(torch.int64)
                for i, lab in enumerate(indices):
                    truth[i, :len(lab)] = torch.LongTensor(lab)

                decoded_job, dec_hidden = dec(pred_transformed, dec_hidden, truth[:, :-1])
                decoder_output = torch.transpose(decoded_job, 2, 1)

                loss = criterion(decoder_output, truth.cuda()[:, 1:]) / b_size
                loss_list.append(loss.item())

                nb_tokens += sum([len(e) for e in indices])

                loss.backward()
                optim.step()
                if len(loss_list) % 1000:
                    f.write("PREDICTION ===================================================== \n")
                    for item in torch.transpose(decoder_output, 2, 1)[0].argmax(-1):
                        f.write(rev_index[item.item()] + ' ')
                    f.write("\n")
                    f.write("LABEL ===================================================== \n")
                    for item in truth[0]:
                        f.write(rev_index[item.item()] + ' ')
                    f.write("\n")

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_perplexity": 2 ** ((sum(loss_list) * b_size / nb_tokens) / math.log(2))}


def valid(dec, dataloader_valid, criterion, vocab_index):
    rev_index = {v: k for k, v in vocab_index.items()}
    loss_list = []
    nb_tokens = 0
    debug_file = os.path.join(args.DATA_DIR, "debug_elmo_LSTM.txt")
    with open(debug_file, 'a') as f:
        f.write("VALIDATION ===================================================== \n")
        with ipdb.launch_ipdb_on_exception():

            for ids, tensors, indices, words in tqdm(dataloader_valid, desc="Validating..."):
                # if len(loss_list) < 100:

                b_size = len(ids)

                # initialize the hidden state of the decoder
                h_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
                c_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
                dec_hidden = (h_0, c_0)

                predictions = torch.cat(tensors).cuda()
                pred_transformed = predictions.expand(b_size, max([len(e) for e in words]) - 1, 1024)

                truth = torch.zeros(b_size, max([len(e) for e in indices])).type(torch.int64)
                for i, lab in enumerate(indices):
                    truth[i, :len(lab)] = torch.LongTensor(lab)

                decoded_job, dec_hidden = dec(pred_transformed, dec_hidden, truth[:, :-1])
                decoder_output = torch.transpose(decoded_job, 2, 1)

                loss = criterion(decoder_output, truth.cuda()[:, 1:]) /b_size
                loss_list.append(loss.item())

                nb_tokens += sum([len(e) for e in indices])
                if len(loss_list) % 1000:
                    f.write("PREDICTION ===================================================== \n")
                    for item in torch.transpose(decoder_output, 2, 1)[0].argmax(-1):
                        f.write(rev_index[item.item()] + ' ')
                    f.write("\n")

                    f.write("LABEL ===================================================== \n")
                    for item in truth[0]:
                        f.write(rev_index[item.item()] + ' ')
                    f.write("\n")

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_perplexity": 2 ** ((sum(loss_list) * b_size / nb_tokens) / math.log(2))}


def label_data(train_data, valid_data, test_data, labels):
    train_labelled = []
    for tup in train_data:
        if tup[0] in labels["train_data"].keys():
            ind = torch.LongTensor(labels["train_data"][tup[0]][0])
            words = labels["train_data"][tup[0]][1]
            train_labelled.append((tup[0], tup[1], ind, words))
    valid_labelled = []
    for tup in valid_data:
        if tup[0] in labels["valid_data"].keys():
            ind = torch.LongTensor(labels["valid_data"][tup[0]][0])
            words = labels["valid_data"][tup[0]][1]
            valid_labelled.append((tup[0], tup[1], ind, words))
    test_labelled = []
    for tup in test_data:
        if tup[0] in labels["test_data"].keys():
            ind = torch.LongTensor(labels["test_data"][tup[0]][0])
            words = labels["test_data"][tup[0]][1]
            test_labelled.append((tup[0], tup[1], ind, words))

    return train_labelled, valid_labelled, test_labelled


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    indices = [e[2] for e in batch]
    words = [e[3] for e in batch]
    return ids, tensors, indices, words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="pkl/prof_ind_elmo_train_cpu.pkl")
    parser.add_argument("--valid_file", type=str, default="pkl/prof_ind_elmo_valid_cpu.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--dec_hidden_size", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="elmo_lstm")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--dec_model", type=str, default="elmo_bis_bs160_lr0.001_tf1_hs_256_max_ep_300_40k_dec_best_ep_14")
    parser.add_argument("--optim", type=str, default="elmo_bs160_lr0.001_tf1_hs_256_max_ep_300_40k_optim_best_ep_4")
    parser.add_argument("--tf", type=float, default=1)
    parser.add_argument("--record_data", type=str, default=True)
    parser.add_argument("--label_file", type=str, default="pkl/lj_labels.pkl")
    parser.add_argument("--dpo", type=float, default=.5)
    parser.add_argument("--emb_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--enc_model", type=str, default="elmo_w2v_bs64_lr0.0001_tf0_hs_512_max_ep_300_encCareer_best_ep_185")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20/job_dec')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

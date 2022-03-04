import argparse
import os

import ipdb
import torch
import pickle as pkl

from allennlp.modules import Elmo
from quiviz import quiviz
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import DecoderWithElmo, LastJobPredictor
from utils import save_best_classifier, model_checkpoint_classifier


def main(args):
    enc_hs = str.split(args.enc_model, sep="_")[8]
    enc_lr = str.split(args.enc_model, sep="_")[5]
    enc_type = str.split(args.enc_model, sep="_")[1]
    enc_ep = str.split(args.enc_model, sep="_")[-1]

    suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)

    xp_title = "LJ pred bs" + str(args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())
    print("Loading data...")
    data_train = []
    data_valid = []
    data_test = []
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/career_train_cpu" + suffix + ".pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_train.append(data)
            except EOFError:
                flag_err = True
                continue
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/career_valid_cpu" + suffix + ".pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_valid.append(data)
            except EOFError:
                flag_err = True
                continue
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/career_test_cpu" + suffix + ".pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_test.append(data)
            except EOFError:
                flag_err = True
                continue
    label_file = os.path.join(args.DATA_DIR, args.label_file)
    with open(label_file, 'rb') as f:
        labeled_data = pkl.load(f)

    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)
    print("Data loaded.")

    train_set, valid_set, test_set = label_data(data_train, data_valid, data_test, labeled_data)

    if args.record_data:
        with open(os.path.join(args.DATA_DIR, "train_lj.pkl"), "wb") as f:
            pkl.dump(train_set, f)
        with open(os.path.join(args.DATA_DIR, "valid_lj.pkl"), "wb") as f:
            pkl.dump(valid_set, f)
        with open(os.path.join(args.DATA_DIR, "test_lj.pkl"), "wb") as f:
            pkl.dump(test_set, f)

    hidden_size = args.dec_hidden_size * 2
    num_layers = 1
    elmo_dimension = 1024

    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("Initializing ELMo...")
    elmo = Elmo(options_file, weight_file, 2, requires_grad=False, dropout=0)
    print("ELMo ready.")

    decoder = DecoderWithElmo(elmo, elmo_dimension, hidden_size, num_layers, len(index))

    predictor = LastJobPredictor(int(enc_hs)*2, elmo_dimension)

    optim = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float('Inf')

    res_epoch = {}
    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, predictor.cuda(), decoder.cuda(), criterion, args, best_val_loss, optim, train_set, valid_set, suffix)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']


def main_for_one_epoch(ep, predictor, dec, criterion, args, best_val_loss, optim, train_dataset, valid_dataset, suffix):

    train_loss = train(predictor, dec, train_dataset, criterion, optim)
    valid_loss = valid(predictor, dec, valid_dataset, criterion)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_classifier(args, ep, target_dir, predictor, optim, suffix)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_classifier(args, ep, target_dir, predictor, optim, suffix)

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(predictor, dec, train_dataset, criterion, optim):
    loss_list = []
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    with ipdb.launch_ipdb_on_exception():
        for ids, tensors, indices, words in tqdm(dataloader_train, desc="Training..."):
            optim.zero_grad()

            b_size = len(ids)

            # initialize the hidden state of the decoder
            h_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
            c_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
            dec_hidden = (h_0, c_0)

            predictions = predictor(torch.cat(tensors).cuda())
            pred_transformed = predictions.expand(b_size, max([len(e) for e in words]), 1024)

            decoded_job, dec_hidden = dec(pred_transformed, dec_hidden, words)

            decoder_output = torch.transpose(decoded_job, 2, 1)

            truth = torch.zeros(b_size, max([len(e) for e in indices])).type(torch.int64)
            for i, lab in enumerate(indices):
                truth[i, :len(lab)] = torch.LongTensor(lab)

            loss = criterion(decoder_output, truth.cuda())
            loss_list.append(loss)

            loss.backward()
            optim.step()

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item()}


def valid(predictor, dec, valid_dataset, criterion):
    loss_list = []
    dataloader_train = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    with ipdb.launch_ipdb_on_exception():
        for ids, tensors, indices, words in tqdm(dataloader_train, desc="Validating..."):

            b_size = len(ids)

            # initialize the hidden state of the decoder
            h_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
            c_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
            dec_hidden = (h_0, c_0)

            predictions = predictor(torch.cat(tensors).cuda())
            pred_transformed = predictions.expand(b_size, max([len(e) for e in words]), 1024)

            decoded_job, dec_hidden = dec(pred_transformed, dec_hidden, words)
            decoder_output = torch.transpose(decoded_job, 2, 1)

            truth = torch.zeros(b_size, max([len(e) for e in indices])).type(torch.int64)
            for i, lab in enumerate(indices):
                truth[i, :len(lab)] = torch.LongTensor(lab)

            loss = criterion(decoder_output, truth.cuda())
            loss_list.append(loss)

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item()}


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
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--label_file", type=str, default="pkl/lj_labels.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--record_data", type=str, default=True)
    parser.add_argument("--dec_hidden_size", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="lj")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/elmo_w2v')
    parser.add_argument("--enc_model", type=str, default="elmo_w2v_gradclip_sgd_bs64_lr0.001_tf0_hs_512_max_ep_300_encCareer_best_ep_185")
    parser.add_argument("--dec_model", type=str, default="")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

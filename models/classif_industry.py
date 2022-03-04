import argparse

import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import IndustryClassifier
import os
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import ipdb

from utils import save_best_classifier, model_checkpoint_classifier


def main(args):

    enc_type = str.split(args.enc_model, sep="_")[1]

    if enc_type == "w2v":
        enc_hs = str.split(args.enc_model, sep="_")[7]
        enc_lr = str.split(args.enc_model, sep="_")[3]
        enc_ep = str.split(args.enc_model, sep="_")[-2]
    else:
        enc_hs = str.split(args.enc_model, sep="_")[9]
        enc_lr = str.split(args.enc_model, sep="_")[3]
        enc_ep = str.split(args.enc_model, sep="_")[-1]

    xp_title = "ind sftmx " + enc_type + " bs" + str(args.batch_size) + " " + enc_lr + " hs" + enc_hs
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)

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
    print("Data loaded.")

    train, valid, test, class_dict = label_data(data_train, data_valid, data_test, labeled_data)

    if args.record_data:
        with open(os.path.join(args.DATA_DIR, "train_industry" + suffix + ".pkl"), "wb") as f:
            pkl.dump(train, f)
        with open(os.path.join(args.DATA_DIR, "valid_industry" + suffix + ".pkl"), "wb") as f:
            pkl.dump(valid, f)
        with open(os.path.join(args.DATA_DIR, "test_industry" + suffix + ".pkl"), "wb") as f:
            pkl.dump(test, f)
        with open(os.path.join(args.DATA_DIR, "classes_industry" + suffix + ".pkl"), "wb") as f:
            pkl.dump(class_dict, f)

    input_size = int(enc_hs) * 2
    classifier = IndustryClassifier(input_size, input_size, len(class_dict))

    optim = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float('Inf')

    res_epoch = {}
    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, classifier.cuda(), criterion, args, best_val_loss, optim, train, valid, suffix)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']



@quiviz.log
def main_for_one_epoch(epoch, classifier, criterion, args, best_val_loss, optim, train_dataset, valid_dataset, suffix):
    print("Training and validating for epoch " + str(epoch))

    train_loss = train(classifier, train_dataset, criterion, optim)
    valid_loss = valid(classifier, valid_dataset, criterion)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_classifier(args, epoch, target_dir, classifier, optim, suffix)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_classifier(args, epoch, target_dir, classifier, optim, suffix)

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(classifier, train, criterion, optim):
    loss_list = []

    dataloader_train = DataLoader(train, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)

    with ipdb.launch_ipdb_on_exception():
        true_positive = 0
        for ids, tensors, labels in tqdm(dataloader_train, desc="Training..."):
            optim.zero_grad()
            pred = classifier(torch.cat(tensors).cuda())
            truth = []
            for i in range(len(ids)):
                truth.append(labels[i])
            loss = criterion(pred.transpose(2, 1), torch.LongTensor(truth).unsqueeze(1).cuda())
            loss_list.append(loss)

            loss.backward()
            optim.step()

            for i, t in enumerate(pred):
                if t.argmax().item() == truth[i]:
                    true_positive += 1

        acc = true_positive / (len(dataloader_train) * args.batch_size)

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_acc": acc}


def valid(classifier, valid, criterion):
    loss_list = []
    dataloader_valid = DataLoader(valid, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    with ipdb.launch_ipdb_on_exception():
        true_positive = 0
        for ids, tensors, labels in tqdm(dataloader_valid, desc="Validating..."):
            pred = classifier(torch.cat(tensors).cuda())
            truth = []
            for i in range(len(ids)):
                truth.append(labels[i])
            loss = criterion(pred.transpose(2, 1), torch.LongTensor(truth).unsqueeze(1).cuda())
            loss_list.append(loss)

            for i, t in enumerate(pred):
                if t.argmax().item() == truth[i]:
                    true_positive += 1

        acc = true_positive / (len(dataloader_valid) * args.batch_size)

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_acc": acc}


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return ids, tensors, labels


def label_data(train, valid, test, labels):
    train_tmp = []
    all_labels = set()
    for tup in train:
        if tup[0] in labels.keys():
            train_tmp.append((tup[0], tup[1], labels[tup[0]]))
            all_labels.add(labels[tup[0]])
    valid_tmp = []
    for tup in valid:
        if tup[0] in labels.keys():
            valid_tmp.append((tup[0], tup[1], labels[tup[0]]))
            all_labels.add(labels[tup[0]])
    test_tmp = []
    for tup in test:
        if tup[0] in labels.keys():
            test_tmp.append((tup[0], tup[1], labels[tup[0]]))
            all_labels.add(labels[tup[0]])

    classes = sorted(list(all_labels))
    class_dict = {name: indice for indice, name in enumerate(classes)}

    train_labelled = []
    for tup in train_tmp:
        train_labelled.append((tup[0], tup[1], class_dict[tup[2]]))
    valid_labelled = []
    for tup in valid_tmp:
        valid_labelled.append((tup[0], tup[1], class_dict[tup[2]]))
    test_labelled = []
    for tup in test_tmp:
        test_labelled.append((tup[0], tup[1], class_dict[tup[2]]))

    return train_labelled, valid_labelled, test_labelled, class_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--label_file", type=str, default="pkl/labels.p")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--record_data", type=str, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_type", type=str, default="industry")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/mlp_rnn')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--enc_model", type=str, default="elmo_w2v_rnn_bs64_lr0.001_tf0_hs_512_max_ep_300_encCareer_best_ep_98_savec")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

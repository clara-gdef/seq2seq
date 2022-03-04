import argparse
from collections import Counter

import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
import os
import pickle as pkl
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader
import ipdb


def main(args):

    enc_hs = str.split(args.enc_model, sep="_")[8]
    enc_lr = str.split(args.enc_model, sep="_")[5]
    enc_type = str.split(args.enc_model, sep="_")[1]
    enc_ep = str.split(args.enc_model, sep="_")[-1]

    xp_title = "industry " + enc_type + " bs" + str(args.batch_size) + " lr" + enc_lr + " hs" + enc_hs
    # quiviz.name_xp(xp_title)
    # quiviz.register(LinePlotObs())

    #suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)
    suffix = "_w2v_lr0.001_256_184"
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
    # with open(os.path.join(args.DATA_DIR, "pkl/career_valid_cpu" + suffix + ".pkl"), 'rb') as f:
    #     while not flag_err:
    #         try:
    #             data = pkl.load(f)
    #             data_valid.append(data)
    #         except EOFError:
    #             flag_err = True
    #             continue
    # flag_err = False
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

    train, valid, test, class_dict = label_data(data_train, None, data_test, labeled_data)

    dataloader_train = DataLoader(train, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    # dataloader_valid = DataLoader(valid, batch_size=args.batch_size, collate_fn=collate,
    #                               shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(test, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)

    class_count_train = Counter()
    class_count_valid = Counter()
    class_count_test = Counter()
    #
    for ids, tensors, labels in tqdm(dataloader_train, desc="Scanning for train..."):
        class_count_train[labels[0]] += 1
    #
    # for ids, tensors, labels in tqdm(dataloader_valid, desc="Scanning for valid..."):
    #     class_count_valid[labels[0]] += 1

    for ids, tensors, labels in tqdm(dataloader_test, desc="Scanning for test..."):
        class_count_test[labels[0]] += 1

    most_common_train = class_count_train.most_common(1)[0][0]
    #most_common_valid = class_count_valid.most_common(1)[0][0]
    most_common_test = class_count_test.most_common(1)[0][0]
    print(most_common_train)
    #print(most_common_valid)
    print(most_common_test)

    true_positive_train = 0
    true_positive_valid = 0
    true_positive_test = 0

    for ids, tensors, labels in tqdm(dataloader_train, desc="Scanning for train..."):
        if labels[0] == most_common_train:
            true_positive_train += 1

    # for ids, tensors, labels in tqdm(dataloader_valid, desc="Scanning for valid..."):
    #     if labels[0] == most_common_valid:
    #         true_positive_valid += 1

    labs = []
    for ids, tensors, labels in tqdm(dataloader_test, desc="Scanning for test..."):
        labs.append(labels[0])
        if labels[0] == most_common_test:
            true_positive_test += 1


    dico = {'acc_train': true_positive_train/len(dataloader_train),
            # 'acc_valid': true_positive_valid/len(dataloader_valid),
            'acc_test': true_positive_test/len(dataloader_test)}


    num_c = range(523)
    handle = "mc_ind"
    label = torch.LongTensor(labs)
    tmp = torch.LongTensor(1)
    tmp[0] = most_common_train
    pred = tmp.expand(len(labs), 1)
    ipdb.set_trace()

    dico2 = {
        "acc_" + handle: accuracy_score(label, pred) * 100,
        "precision_" + handle: precision_score(label, pred, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(label, pred, average='weighted', labels=num_c,
                                         zero_division=0) * 100,
        "f1_" + handle: f1_score(label, pred, average='weighted', labels=num_c, zero_division=0) * 100}

    ipdb.set_trace()
    # most common at 10
    mc_10 = class_count_test.most_common(10)
    handle = "mc_ind_@10"
    

    print(dico)


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
    # for tup in valid:
    #     if tup[0] in labels.keys():
    #         valid_tmp.append((tup[0], tup[1], labels[tup[0]]))
    #         all_labels.add(labels[tup[0]])
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
    # for tup in valid_tmp:
    #     valid_labelled.append((tup[0], tup[1], class_dict[tup[2]]))
    test_labelled = []
    for tup in test_tmp:
        test_labelled.append((tup[0], tup[1], class_dict[tup[2]]))

    return train_labelled, valid_labelled, test_labelled, class_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--label_file", type=str, default="pkl/labels.p")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="industry")
    parser.add_argument("--enc_model", type=str, default="elmo_w2v_gradclip_sgd_bs64_lr0.0001_tf0_hs_512_max_ep_300_encCareer_best_ep_185")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/elmo_w2v')
    args = parser.parse_args()
    main(args)

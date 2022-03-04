import argparse
import itertools
from collections import Counter

import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
import os
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import ipdb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss


def main(args):

    # enc_hs = str.split(args.enc_model, sep="_")[8]
    # enc_lr = str.split(args.enc_model, sep="_")[5]
    # enc_type = str.split(args.enc_model, sep="_")[1]
    # enc_ep = str.split(args.enc_model, sep="_")[-1]

    xp_title = "sk ft mc"
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    # suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)

    suffix = "_pt"

    print("Loading data...")
    data_train = []
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_ft" + suffix + "_train.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_train.append(data)
            except EOFError:
                flag_err = True
                continue
    data_valid = []
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_ft" + suffix + "_valid.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_valid.append(data)
            except EOFError:
                flag_err = True
                continue
    data_test = []
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep_ft" + suffix + "_test.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_test.append(data)
            except EOFError:
                flag_err = True
                continue
    label_file = os.path.join(args.DATA_DIR, args.label_file)
    with open(label_file, 'rb') as f:
        labels = pkl.load(f)
    class_file = os.path.join(args.DATA_DIR, args.class_file)
    with open(class_file, 'rb') as f:
        classes = pkl.load(f)
    print("Data loaded.")

    train_set, valid_set, test_set, class_dict, avg_skill_number = label_data(data_train, data_valid, data_test, labels, classes)


    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)

    with ipdb.launch_ipdb_on_exception():
        indices = []
        for ids, tensors, labels in tqdm(itertools.chain(dataloader_train, dataloader_valid), desc="Building labels..."):
            for i, val in enumerate(labels[0]):
                if val == 1.:
                    indices.append(i)

        class_count_train_valid = Counter()
        for i in indices:
            class_count_train_valid[i] += 1

        most_common_train_valid = [i[0] for i in class_count_train_valid.most_common(avg_skill_number)]
        print(most_common_train_valid)

        most_common_tensor = torch.zeros(1, len(classes))
        for element in most_common_train_valid:
            most_common_tensor[0, element] = 1.

        avg_precision = []
        avg_recall = []

        truth = most_common_tensor.type(dtype=torch.uint8)

        labs = []
        for ids, tensors, labels in tqdm(dataloader_test, desc="Scanning for test..."):
            pred = torch.from_numpy(labels[0]).type(dtype=torch.uint8)
            labs.append(pred)
            true_positive = torch.sum(pred & truth, dim=1)

            false_positive = torch.sum((truth == 0) & (pred == 1), dim=1)
            false_negative = torch.sum((truth == 1) & (pred == 0), dim=1)

            tmp = true_positive.type(torch.float32) / ((true_positive + false_positive).type(torch.float32) + 1e-15)

            avg_precision.extend(tmp)
            avg_recall.extend(
                true_positive.type(torch.float32) / (true_positive + false_negative).type(torch.float32))

        precision = torch.mean(torch.FloatTensor(avg_precision)).item()
        recall = torch.mean(torch.FloatTensor(avg_recall)).item()

        dico = {"precision": precision,
                "recall": recall,
                "F1": 2*(precision * recall) / (recall + precision)}

        num_c = range(523)
        handle = "mc_sk"

        ipdb.set_trace()
        label = torch.stack(labs)
        pred = most_common_tensor.expand(len(label), 523)
        dico2 = {
            "hamming_" + handle: hamming_loss(label, pred) * 100,
            "precision_" + handle: precision_score(label, pred, average='micro',
                                                   labels=num_c, zero_division=0) * 100,
            "recall_" + handle: recall_score(label, pred, average='micro', labels=num_c,
                                             zero_division=0) * 100,
            "f1_" + handle: f1_score(label, pred, average='micro', labels=num_c, zero_division=0) * 100}


    print(dico)
    print(dico2)


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return ids, tensors, labels


def label_data(train, valid, test, labels, classes):
    avg_number_skills = []
    # all_labels = set()
    train_tmp = []
    for tup in train:
        if tup[0] in labels.keys():
            avg_number_skills.append(sum(labels[tup[0]]))
            train_tmp.append((tup[0], tup[1], labels[tup[0]]))
            # all_labels.add(labels[tup[0]])

    valid_tmp = []
    for tup in valid:
        if tup[0] in labels.keys():
            avg_number_skills.append(sum(labels[tup[0]]))
            valid_tmp.append((tup[0], tup[1], labels[tup[0]]))
            # all_labels.add(labels[tup[0]])

    test_tmp = []
    for tup in test:
        if tup[0] in labels.keys():
            test_tmp.append((tup[0], tup[1], labels[tup[0]]))

    # classes = sorted(list(all_labels))
    class_dict = {name: indice for indice, name in enumerate(classes)}
    #
    # train_labelled = []
    # for tup in train_tmp:
    #     train_labelled.append((tup[0], tup[1], class_dict[tup[2]]))
    #
    # valid_labelled = []
    # for tup in valid_tmp:
    #     valid_labelled.append((tup[0], tup[1], class_dict[tup[2]]))
    #
    # test_labelled = []
    # for tup in test_tmp:
    #     test_labelled.append((tup[0], tup[1], class_dict[tup[2]]))

    # return train_labelled, valid_labelled, test_labelled, class_dict, torch.mean(torch.LongTensor(avg_number_skills))
    return train_tmp, valid_tmp, test_tmp, class_dict, int(torch.mean(torch.FloatTensor(avg_number_skills)).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--label_file", type=str, default="pkl/labels_skills.p")
    parser.add_argument("--class_file", type=str, default="pkl/good_skills.p")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="sk")
    # parser.add_argument("--ft_type", type=s)
    # parser.add_argument("--enc_model", type=str, default="elmo_w2v_gradclip_sgd_bs64_lr0.0001_tf0_hs_512_max_ep_300_encCareer_best_ep_185")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/elmo_w2v')
    args = parser.parse_args()
    main(args)

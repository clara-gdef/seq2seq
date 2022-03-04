import argparse
import itertools
import os
import pickle as pkl
import torch

from quiviz import quiviz
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader
from torch.autograd import gradcheck
from tqdm import tqdm

from classes import IndustryClassifier, SkillsPredictor
from utils import ProfileDatasetElmo
from utils.Utils import transform_for_elmo_lj, collate_profiles_lj_elmo, save_best_model_elmo_w2v, \
    model_checkpoint_elmo_w2v, save_best_classifier, model_checkpoint_classifier
import ipdb


def init(args):
    # # loading data
    # print("Loading data...")
    # train_file = os.path.join(args.DATA_DIR, args.train_file)
    # datadict_train = {"data": []}
    # flag_err = False
    #
    # with open(train_file, "rb") as f:
    #     while not flag_err:
    #         try:
    #             datadict_train["data"].append(pkl.load(f))
    #         except EOFError:
    #             flag_err = True
    #             continue
    # print("Train file loaded.")
    # valid_file = os.path.join(args.DATA_DIR, args.valid_file)
    # datadict_valid = {"data": []}
    # flag_err = False
    # with open(valid_file, "rb") as f:
    #     while not flag_err:
    #         try:
    #             datadict_valid["data"].append(pkl.load(f))
    #         except EOFError:
    #             flag_err = True
    #             continue
    # print("Valid file loaded.")
    # test_file = os.path.join(args.DATA_DIR, args.test_file)
    # datadict_test = {"data": []}
    # flag_err = False
    # with open(test_file, "rb") as f:
    #     while not flag_err:
    #         try:
    #             datadict_test["data"].append(pkl.load(f))
    #         except EOFError:
    #             flag_err = True
    #             continue
    # print("Test file loaded.")
    # print("Data loaded.")

    with ipdb.launch_ipdb_on_exception():
        # data_train = ProfileDatasetElmo(datadict_train, transform_for_elmo_lj)
        # data_valid = ProfileDatasetElmo(datadict_valid, transform_for_elmo_lj)
        # data_test = ProfileDatasetElmo(datadict_test, transform_for_elmo_lj)
        # del datadict_train, datadict_valid, datadict_test

        elmo_size = 1024

        if args.bl_type == "ind":
            print("Loading data...")
            with open(os.path.join(args.DATA_DIR, "train_ind_bl.pkl"), "rb") as f:
                data_train = pkl.load(f)
            with open(os.path.join(args.DATA_DIR, "valid_ind_bl.pkl"), "rb") as f:
                data_valid = pkl.load(f)
            with open(os.path.join(args.DATA_DIR, "classes_ind_bl.pkl"), "rb") as f:
                class_dict_ind = pkl.load(f)
            print("Data loaded.")

            load_ind(elmo_size, data_train, data_valid, class_dict_ind=class_dict_ind)
        else:

            print("Loading data...")
            with open(os.path.join(args.DATA_DIR, "train_sk_bl.pkl"), "rb") as f:
                data_train = pkl.load(f)
            with open(os.path.join(args.DATA_DIR, "valid_sk_bl.pkl"), "rb") as f:
                data_valid = pkl.load(f)
            print("Data loaded.")

            load_sk(elmo_size, data_train, data_valid)


def load_ind(elmo_size, data_train, data_valid, class_dict_ind=None):
    xp_title = "BL ind bs" + str(args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    # with open(os.path.join(args.DATA_DIR, args.label_file_ind), "rb") as f:
    #     labeled_data_ind = pkl.load(f)
    # train_ind, valid_ind, test_ind, class_dict_ind = label_data_ind(data_train, data_valid, data_test, labeled_data_ind)
    #
    # if args.record_data:
    #     with open(os.path.join(args.DATA_DIR, "train_ind_bl.pkl"), "wb") as f:
    #         pkl.dump(train_ind, f)
    #     with open(os.path.join(args.DATA_DIR, "valid_ind_bl.pkl"), "wb") as f:
    #         pkl.dump(valid_ind, f)
    #     with open(os.path.join(args.DATA_DIR, "test_ind_bl.pkl"), "wb") as f:
    #         pkl.dump(test_ind, f)
    #     with open(os.path.join(args.DATA_DIR, "classes_ind_bl.pkl"), "wb") as f:
    #         pkl.dump(class_dict_ind, f)

    ind_classifier = IndustryClassifier(elmo_size, elmo_size, len(class_dict_ind)).cuda()
    optim_ind = torch.optim.Adam(ind_classifier.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float('Inf')

    dataloader_train_ind = DataLoader(data_train, batch_size=args.batch_size, collate_fn=collate,
                                      shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid_ind = DataLoader(data_valid, batch_size=args.batch_size, collate_fn=collate,
                                      shuffle=True, num_workers=0, drop_last=True)
    res_epoch = {}
    for e in range(1, args.epoch + 1):
        dico = main(dataloader_train_ind, dataloader_valid_ind, ind_classifier, criterion, optim_ind, best_val_loss, e,
                    "_IND")
        res_epoch[e] = {'train_loss': dico['train_loss'],
                        'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']


def load_sk(elmo_size, data_train, data_valid):
    xp_title = "BL skills bs" + str(args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    with open(os.path.join(args.DATA_DIR, args.classes_file_sk), "rb") as f:
        classes_sk = pkl.load(f)

    sk_classifier = SkillsPredictor(elmo_size, elmo_size, len(classes_sk)).cuda()
    optim_sk = torch.optim.Adam(sk_classifier.parameters(), lr=args.lr)

    criterion_sk = torch.nn.BCELoss()
    best_val_loss = float('Inf')

    dataloader_train_sk = DataLoader(data_train, batch_size=args.batch_size, collate_fn=collate,
                                     shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid_sk = DataLoader(data_valid, batch_size=args.batch_size, collate_fn=collate,
                                     shuffle=True, num_workers=0, drop_last=True)

    res_epoch = {}
    for e in range(1, args.epoch + 1):
        dico = main(dataloader_train_sk, dataloader_valid_sk, sk_classifier, criterion_sk, optim_sk, best_val_loss, e,
                    "_SK")
        res_epoch[e] = {'train_loss': dico['train_loss'],
                        'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']


@quiviz.log
def main(dl_train, dl_valid, classifier, crit, optim, best_val_loss, epoch, suffix):
    exp_type = "ind " if args.bl_type == "ind" else "sk "
    print("Training and validating " + exp_type + "for epoch " + str(epoch))

    train_loss = train(classifier, dl_train, crit, optim)
    valid_loss = valid(classifier, dl_valid, crit)

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


def train(classifier, dataloader_train, criterion, optim):
    loss_list = []
    with ipdb.launch_ipdb_on_exception():
        for ids, tensors_list, labels in tqdm(dataloader_train, desc="Training..."):
            optim.zero_grad()

            if args.bl_type == "ind":
                tensors = torch.zeros(len(ids), 1, tensors_list[0][0].shape[-1]).cuda()
                for b in range(len(ids)):
                    tensors[b, :, :] = torch.mean(torch.stack(tensors_list[b]), dim=0).cuda()

                pred = classifier(tensors)
                truth = torch.LongTensor(labels)

                loss = criterion(pred.transpose(2, 1), torch.LongTensor(truth).unsqueeze(1).cuda())
            else:
                tensors = torch.zeros(len(ids), 1, tensors_list[0][0].shape[-1]).cuda()
                for b in range(len(ids)):
                    tensors[b, :, :] = torch.mean(torch.stack(tensors_list[b]), dim=0).cuda()

                pred = classifier(tensors)
                truth = torch.stack(labels)

                loss = criterion(pred, truth.unsqueeze(1).cuda())
            loss_list.append(loss)

            loss.backward()
            optim.step()

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item()}


def valid(classifier, dataloader_valid, criterion):
    loss_list = []
    with ipdb.launch_ipdb_on_exception():
        for ids, tensors_list, labels in tqdm(dataloader_valid, desc="Validating..."):

            if args.bl_type == "ind":
                tensors = torch.zeros(len(ids), 1, tensors_list[0][0].shape[-1]).cuda()
                for b in range(len(ids)):
                    tensors[b, :, :] = torch.mean(torch.stack(tensors_list[b]), dim=0).cuda()

                pred = classifier(tensors)
                truth = torch.LongTensor(labels)

                loss = criterion(pred.transpose(2, 1), torch.LongTensor(truth).unsqueeze(1).cuda())
            else:
                tensors = torch.zeros(len(ids), 1, tensors_list[0][0].shape[-1]).cuda()
                for b in range(len(ids)):
                    tensors[b, :, :] = torch.mean(torch.stack(tensors_list[b]), dim=0).cuda()

                pred = classifier(tensors)
                truth = torch.stack(labels)

                loss = criterion(pred, truth.unsqueeze(1).cuda())
            loss_list.append(loss)

            loss.backward()

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item()}


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return ids, tensors, labels


def label_data_ind(train, valid, test, labels):
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


def label_data_sk(train, valid, test, labels):
    train_labelled = []
    for tup in train:
        if tup[0] in labels.keys():
            lab = torch.from_numpy(labels[tup[0]])
            train_labelled.append((tup[0], tup[1], lab.type(torch.float32)))
    valid_labelled = []
    for tup in valid:
        if tup[0] in labels.keys():
            lab = torch.from_numpy(labels[tup[0]])
            valid_labelled.append((tup[0], tup[1], lab.type(torch.float32)))
    test_labelled = []
    for tup in test:
        if tup[0] in labels.keys():
            lab = torch.from_numpy(labels[tup[0]])
            test_labelled.append((tup[0], tup[1], lab.type(torch.float32)))

    return train_labelled, valid_labelled, test_labelled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bl_type", type=str, default="ind")
    parser.add_argument("--train_file", type=str, default="pkl/prof_rep_elmo_train_cpu.pkl")
    parser.add_argument("--valid_file", type=str, default="pkl/prof_rep_elmo_valid_cpu.pkl")
    parser.add_argument("--test_file", type=str, default="pkl/prof_rep_elmo_test_cpu.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--model_type", type=str, default="baseline")
    parser.add_argument("--record_data", type=bool, default=True)
    parser.add_argument("--label_file_sk", type=str, default="pkl/labels_skills.p")
    parser.add_argument("--label_file_ind", type=str, default="pkl/labels.p")
    parser.add_argument("--classes_file_sk", type=str, default="pkl/good_skills.p")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/bl')
    parser.add_argument("--epoch", type=int, default=300)
    args = parser.parse_args()
    init(args)

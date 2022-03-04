import argparse

import itertools
import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import IndustryClassifier
import os
import pickle as pkl
import math
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from utils.LinkedInDataset import LinkedInDataset
import random
import ipdb


def main(args):
    xp_title = "industry classif bs" + str(args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    print("Loading data...")
    mean_jobs = []
    last_jobs = []
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/job_mean_rep_train.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                mean_jobs.append(data)
            except EOFError:
                flag_err = True
                continue
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/last_job_rep_train.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                last_jobs.append(data)
            except EOFError:
                flag_err = True
                continue
    with open(os.path.join(args.DATA_DIR, "pkl/labels.pkl"), 'rb') as file:
        labels_full = pkl.load(file)
    print("Data loaded.")

    tmp_dat = mean_jobs[800:]
    mean_jobs_train = tmp_dat
    tmp_dat = mean_jobs[:-200]
    mean_jobs_valid = tmp_dat

    tmp_dat = last_jobs[800:]
    last_jobs_train = tmp_dat
    tmp_dat = last_jobs[:-200]
    last_jobs_valid = tmp_dat

    classes = sorted(list(set(v for k, v in labels_full.items())))
    class_dico = {name: i for i, name in enumerate(classes)}
    train_indices = [e[0] for e in mean_jobs_train]
    valid_indices = [e[0] for e in mean_jobs_valid]
    train_indices.extend(valid_indices)

    labels = {}
    for i in train_indices:
        labels[i] = class_dico[labels_full[i]]

    input_size = args.hidden_size * 2
    num_classes = len(classes)
    classifier_mean = IndustryClassifier(input_size, 32, num_classes)
    classifier_last = IndustryClassifier(input_size, 32, num_classes)

    optim_mean = torch.optim.Adam(classifier_mean.parameters())
    optim_last = torch.optim.Adam(classifier_last.parameters())

    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss_last = 1e+300
    best_val_loss_mean = 1e+300

    res_epoch = {}
    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, classifier_mean.cuda(), classifier_last.cuda(), criterion, args, best_val_loss_mean,
                                  best_val_loss_last, optim_mean, optim_last, mean_jobs_train, mean_jobs_valid,
                                  last_jobs_train, last_jobs_valid, labels)
        res_epoch[epoch] = {'trainMean_loss': dico['trainMean_loss'],
                            'validMean_loss': dico['validMean_loss'],
                            'trainLast_loss': dico['trainLast_loss'],
                            'validLast_loss': dico['validLast_loss']}
        best_val_loss_mean = dico['best_val_loss_mean']
        best_val_loss_last = dico['best_val_loss_last']


@quiviz.log
def main_for_one_epoch(epoch, classifier_mean, classifier_last, criterion, args, best_val_loss_mean,
                                  best_val_loss_last, optim_mean, optim_last,
                       mean_jobs_train, mean_jobs_valid, last_jobs_train, last_jobs_valid, labels):
    print("Training and validating for epoch " + str(epoch))

    train_loss = train(classifier_mean, classifier_last, mean_jobs_train, last_jobs_train, criterion, optim_mean,
                       optim_last, labels)
    valid_loss = valid(classifier_mean, classifier_last, mean_jobs_valid, last_jobs_valid, criterion, labels)

    target_dir = args.model_dir
    if valid_loss['validMean_loss'] < best_val_loss_mean:
        if not args.DEBUG:
            torch.save(classifier_mean.state_dict(), os.path.join(target_dir, 'best_classif_mean_' + str(epoch)))
        best_val_loss_mean = valid_loss['validMean_loss']
    if args.save_last:
        if not args.DEBUG:
            torch.save(classifier_mean.state_dict(), os.path.join(target_dir, 'last_classif_mean_' + str(epoch)))

    target_dir = args.model_dir
    if valid_loss['validLast_loss'] < best_val_loss_last:
        if not args.DEBUG:
            torch.save(classifier_mean.state_dict(), os.path.join(target_dir, 'best_classif_lj_' + str(epoch)))
        best_val_loss_last = valid_loss['validLast_loss']
    if args.save_last:
        if not args.DEBUG:
            torch.save(classifier_mean.state_dict(), os.path.join(target_dir, 'last_classif_lj_' + str(epoch)))

    dictionary = {**train_loss, **valid_loss, 'best_val_loss_mean': best_val_loss_mean,
                  'best_val_loss_last': best_val_loss_last}
    return dictionary


def train(classifier_mean, classifier_last, mean_jobs_train, last_jobs_train, criterion, optim_mean, optim_last, labels):
    loss_mean_list = []
    loss_last_list = []
    dataloader_mean = DataLoader(mean_jobs_train, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_last = DataLoader(last_jobs_train, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    with ipdb.launch_ipdb_on_exception():
        for ids, tensors in tqdm(dataloader_mean, desc="Training for mean"):
            optim_mean.zero_grad()
            pred_mean = classifier_mean(torch.stack(tensors).cuda())
            truth_mean = []
            for i in ids:
                truth_mean.append(labels[i])
            loss_mean = criterion(pred_mean, torch.LongTensor(truth_mean).cuda())
            loss_mean.backward()

            loss_mean_list.append(loss_mean)

            optim_mean.step()

        for ids, tensors in tqdm(dataloader_last, desc="Training for last"):
            optim_mean.zero_grad()
            pred_last = classifier_last(torch.stack(tensors).cuda())
            truth_last = []
            for i in ids:
                truth_last.append(labels[i])
            loss_last = criterion(pred_last.squeeze(1), torch.LongTensor(truth_last).cuda())
            loss_last.backward()

            loss_last_list.append(loss_last)

            optim_last.step()

    return {"trainMean_loss": torch.mean(torch.FloatTensor(loss_mean_list)),
            "trainLast_loss": torch.mean(torch.FloatTensor(loss_last_list))}


def valid(classifier_mean, classifier_last, mean_jobs_valid, last_jobs_valid, criterion, labels):
    loss_mean_list = []
    loss_last_list = []
    dataloader_mean = DataLoader(mean_jobs_valid, batch_size=args.batch_size, collate_fn=collate,
                                 shuffle=True, num_workers=0, drop_last=True)
    dataloader_last = DataLoader(last_jobs_valid, batch_size=args.batch_size, collate_fn=collate,
                                 shuffle=True, num_workers=0, drop_last=True)
    with ipdb.launch_ipdb_on_exception():
        for ids, tensors in tqdm(dataloader_mean, desc="Validating for mean"):
            pred_mean = classifier_mean(torch.stack(tensors).cuda())
            truth_mean = []
            for i in ids:
                truth_mean.append(labels[i])
            loss_mean = criterion(pred_mean, torch.LongTensor(truth_mean).cuda())

            loss_mean_list.append(loss_mean)

        for ids, tensors in tqdm(dataloader_last, desc="Validating for last"):
            pred_last = classifier_last(torch.stack(tensors).cuda())
            truth_last = []
            for i in ids:
                truth_last.append(labels[i])
            loss_last = criterion(pred_last.squeeze(1), torch.LongTensor(truth_last).cuda())
            loss_last.backward()

            loss_last_list.append(loss_last)
    return {"validMean_loss": torch.mean(torch.FloatTensor(loss_mean_list)),
            "validLast_loss": torch.mean(torch.FloatTensor(loss_last_list))}


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    return ids, tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="industry")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

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
import ipdb


def init(args):

        elmo_size = 1024

        if args.bl_type == "ind":
            print("Loading data...")
            with open(os.path.join(args.DATA_DIR, "test_ind_bl.pkl"), "rb") as f:
                data_test = pkl.load(f)
            with open(os.path.join(args.DATA_DIR, "classes_ind_bl.pkl"), "rb") as f:
                class_dict_ind = pkl.load(f)
            print("Data loaded.")

            load_ind(elmo_size, data_test, class_dict_ind=class_dict_ind)
        else:

            print("Loading data...")
            with open(os.path.join(args.DATA_DIR, "test_sk_bl.pkl"), "rb") as f:
                data_test = pkl.load(f)
            print("Data loaded.")

            load_sk(elmo_size, data_test)


def load_ind(elmo_size, data_test, class_dict_ind=None):

    ind_classifier = IndustryClassifier(elmo_size, elmo_size, len(class_dict_ind)).cuda()
    weights = os.path.join(args.model_dir, args.model)
    ind_classifier.load_state_dict(torch.load(weights))

    dataloader_test_ind = DataLoader(data_test, batch_size=args.batch_size, collate_fn=collate,
                                      shuffle=True, num_workers=0, drop_last=True)

    dico = main(dataloader_test_ind, ind_classifier)
    print(dico)
    with open(os.path.join(args.DATA_DIR, "res_bl_ind.pkl"), 'wb') as f:
        pkl.dump(dico, f)


def load_sk(elmo_size, data_test):
    xp_title = "BL skills bs" + str(args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    with open(os.path.join(args.DATA_DIR, args.classes_file_sk), "rb") as f:
        classes_sk = pkl.load(f)

    sk_classifier = SkillsPredictor(elmo_size, elmo_size, len(classes_sk))
    weights = os.path.join(args.model_dir, args.model)
    sk_classifier.load_state_dict(torch.load(weights))

    dataloader_test_sk = DataLoader(data_test, batch_size=args.batch_size, collate_fn=collate,
                                     shuffle=True, num_workers=0, drop_last=True)

    dico = main(dataloader_test_sk, sk_classifier.cuda())
    print(dico)
    with open(os.path.join(args.DATA_DIR, "res_bl_sk.pkl"), 'wb') as f:
        pkl.dump(dico, f)


@quiviz.log
def main(dl_test, classifier):

    with ipdb.launch_ipdb_on_exception():
        if args.bl_type == "ind":
            true_positive = 0
            for ids, tensors_list, labels in tqdm(dl_test, desc="Evaluating..."):
                tensors = torch.zeros(len(ids), 1, tensors_list[0][0].shape[-1]).cuda()
                for b in range(len(ids)):
                    tensors[b, :, :] = torch.mean(torch.stack(tensors_list[b]), dim=0).cuda()

                pred = classifier(tensors)
                truth = torch.LongTensor(labels)

                if pred[0].argmax().item() == truth[0]:
                    true_positive += 1
            acc = true_positive / len(dl_test)
            return {"accuracy": acc}

        else:
            true_positive_global = []
            false_positive_global = []
            false_negative_global = []

            avg_precision = []
            avg_recall = []

            for ids, tensors_list, labels in tqdm(dl_test, desc="Evaluating..."):
                tensors = torch.zeros(len(ids), 1, tensors_list[0][0].shape[-1]).cuda()
                for b in range(len(ids)):
                    tensors[b, :, :] = torch.mean(torch.stack(tensors_list[b]), dim=0).cuda()

                outputs = classifier(tensors)
                predictions = []
                for o in outputs:
                    predictions.append(((o > .25).float() * 1).type(torch.uint8))
                pred = torch.cat(predictions).type(torch.uint8)
                truth = torch.stack(labels).type(torch.uint8).cuda()

                true_positive = torch.sum(pred & truth, dim=1)
                true_positive_global.extend(true_positive)

                false_positive = torch.sum((truth == 0) & (pred == 1), dim=1)
                false_negative = torch.sum((truth == 1) & (pred == 0), dim=1)

                false_positive_global.extend(false_positive)
                false_negative_global.extend(false_negative)

                tmp = true_positive.type(torch.float32) / ((true_positive + false_positive).type(torch.float32) + 1e-15)
                avg_precision.extend(tmp)
                avg_recall.extend(
                    true_positive.type(torch.float32) / (true_positive + false_negative).type(torch.float32))

            fpg = torch.FloatTensor(true_positive_global)
            micro_avg_precision = torch.mean(fpg) / torch.mean(fpg + torch.FloatTensor(false_positive_global))
            micro_avg_recall = torch.mean(fpg) / torch.mean(fpg + torch.FloatTensor(false_negative_global))

            return {"precision": torch.mean(torch.FloatTensor(avg_precision)).item(),
                    "map": micro_avg_precision.item(),
                    "recall": torch.mean(torch.FloatTensor(avg_recall)).item(),
                    "mar": micro_avg_recall.item()}





def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return ids, tensors, labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bl_type", type=str, default="ind")
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
    parser.add_argument("--model", type=str, default='baseline_bs128_lr0.001_max_ep_300__best_ep_34_IND')
    parser.add_argument("--epoch", type=int, default=300)
    args = parser.parse_args()
    init(args)

import argparse
import quiviz
from quiviz.contrib import LinePlotObs
from tqdm import tqdm
from classes import SkillsPredictor
import os
import pickle as pkl
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss

from torch.utils.data import DataLoader
from utils import save_best_classifier, model_checkpoint_classifier
import ipdb


def main(args):

    with ipdb.launch_ipdb_on_exception():
        enc_type = str.split(args.enc_model, sep="_")[1]

        if enc_type == "w2v":
            enc_hs = str.split(args.enc_model, sep="_")[7]
            enc_lr = str.split(args.enc_model, sep="_")[3]
            enc_ep = str.split(args.enc_model, sep="_")[-2]
            suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)

        elif enc_type == "mlp_rnn":
            enc_hs = str.split(args.enc_model, sep="_")[9]
            enc_lr = str.split(args.enc_model, sep="_")[3]
            enc_ep = str.split(args.enc_model, sep="_")[-1]
            suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)

        elif enc_type == "pre":
            suffix = "pt"
            enc_hs = args.emb_size
        elif enc_type == "from":
            suffix = "fs"
            enc_hs = args.emb_size

        # 9/11/20
        input_size = int(enc_hs)
        # input_size = 300
        # suffix = "pt"

        print("Loading data...")
        with open(os.path.join(args.DATA_DIR, "test_skills" + suffix + ".pkl"), 'rb') as f:
            data_test = pkl.load(f)
# =======
#         xp_title = "Sk eval " + suffix
#         # quiviz.name_xp(xp_title)
#         # quiviz.register(LinePlotObs())
#         suffix = ""
#         print("Loading data...")
#         with open(os.path.join(args.DATA_DIR, "test_skills" + suffix + ".pkl"), 'rb') as f:
#             data_test = pkl.load(f)
# >>>>>>> ef1bef058c325094c06e206187902562cf5f0efb

        print("Data loaded.")

        with open(os.path.join(args.DATA_DIR, args.classes_file), "rb") as f:
            classes = pkl.load(f)


        classifier = SkillsPredictor(input_size, input_size, len(classes))
        weights = os.path.join(args.model_dir, args.sk_model)
        classifier.load_state_dict(torch.load(weights))
# =======
#         # input_size = int(enc_hs)
#         input_size = 512
#         classifier = SkillsPredictor(input_size, input_size, len(classes))
#         weights = os.path.join(args.model_dir, args.sk_model)
#         classifier.load_state_dict(torch.load(weights))
# >>>>>>> ef1bef058c325094c06e206187902562cf5f0efb

        dico = dict()
        threshold = list(np.linspace(0, 1, 20))

        for t in threshold:
            if t > .25 and t < .27:
                dico[t] = evaluate(classifier.cuda(), args, data_test, t, suffix, classes)

        with open(os.path.join(args.DATA_DIR, "res_skills" + suffix), 'wb') as f:
            pkl.dump(dico, f)


# @quiviz.log
def evaluate(classifier, args, test_dataset, threshold, suffix, classes):
    classifier.eval()
    nb_skills_predits = []

    avg_precision = []
    avg_recall = []

    rev_classes = {index: name for index, name in enumerate(classes)}

    good_predictions = []
    wrong_predictions = []
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    right_file = os.path.join(args.DATA_DIR, "skills_right_" + suffix + "2.txt")
    wrong_file = os.path.join(args.DATA_DIR, "skills_wrong_" + suffix + "2.txt")

    with ipdb.launch_ipdb_on_exception():
        preds = []
        labs = []
        counter = 0
        for ids, tensors, labels in tqdm(dataloader_test, desc="Evaluating..."):
            #if ids[0] == 100088:
            outputs = classifier(torch.cat(tensors).cuda())
            predictions = []
            for o in outputs:
                predictions.append(((o > threshold).float()*1).type(torch.uint8))
            preds.append(torch.stack(predictions).cpu().numpy())
            labs.append(torch.stack(labels).cpu().numpy())
            pred = torch.stack(predictions).type(torch.uint8)
            truth = torch.stack(labels).type(torch.uint8).cuda()

            true_positive = torch.sum(pred & truth, dim=1)

            false_positive = torch.sum((truth == 0) & (pred == 1), dim=1)
            false_negative = torch.sum((truth == 1) & (pred == 0), dim=1)

            tmp_precision = true_positive.type(torch.float32)/((true_positive + false_positive).type(torch.float32) + 1e-15)

            avg_precision.extend(tmp_precision)
            tmp_recall = true_positive.type(torch.float32)/(true_positive + false_negative).type(torch.float32)
            avg_recall.extend(tmp_recall)
            counter += 1
            if counter >= 5000:
                break


        pred = np.stack(preds)
        label = np.stack(labs).reshape(-1, 523)

        

        num_c = range(523)
        handle = "sk_" + suffix
        res_dict = {
            "hamming_" + handle: hamming_loss(label, pred) * 100,
            "precision_" + handle: precision_score(label, pred, average='weighted',
                                                   labels=num_c, zero_division=0) * 100,
            "recall_" + handle: recall_score(label, pred, average='weighted', labels=num_c,
                                             zero_division=0) * 100,
            "f1_" + handle: f1_score(label, pred, average='weighted', labels=num_c, zero_division=0) * 100}

        print(res_dict)
        ipdb.set_trace()



        #     nb_skills_predits.append(sum(pred))
        #
        #     if torch.mean(tmp_precision).item() + torch.mean(tmp_recall).item() != 0:
        #         tmp_f1 = 2 * (torch.mean(tmp_precision).item() * torch.mean(tmp_recall).item()) / \
        #              (torch.mean(tmp_precision).item() + torch.mean(tmp_recall).item())
        #
        #         if tmp_f1 * 100 > (35):
        #             good_predictions.append((ids, pred, truth, tmp_f1, tmp_recall, tmp_precision))
        #
        #         if tmp_f1 * 100 < (35 / 3):
        #             wrong_predictions.append((ids, pred, truth, tmp_f1, tmp_recall, tmp_precision))
        #
        # print("Right predictions: " + str(len(good_predictions)))
        # print("Wrong predictions: " + str(len(wrong_predictions)))
        #
        # with open(right_file, 'a') as r_f:
        #     for ids, pred, truth, tmp_f1, tmp_recall, tmp_precision in good_predictions:
        #         r_f.write('==================================================================')
        #         r_f.write("ID: " + str(ids[0]))
        #         r_f.write('PRED: ')
        #         for i, sk in enumerate(pred[0]):
        #             if sk == 1:
        #                 r_f.write(rev_classes[i] + "===")
        #         r_f.write('TRUTH: ')
        #         for j, true in enumerate(truth[0]):
        #             if true == 1:
        #                 r_f.write(rev_classes[i] + "===")
        #         r_f.write("f1: " + str(tmp_f1) + ", recall: " + str(tmp_recall) + ", precision: " + str(tmp_precision))
        #         r_f.write("\n")
        #
        # with open(wrong_file, 'a') as w_f:
        #     for ids, pred, truth, tmp_f1, tmp_recall, tmp_precision in wrong_predictions:
        #         w_f.write('==================================================================')
        #         w_f.write("ID: " + str(ids[0]))
        #         w_f.write('PRED: ')
        #         for i, sk in enumerate(pred[0]):
        #             if sk == 1:
        #                 w_f.write(rev_classes[i] + "===")
        #         w_f.write('TRUTH: ')
        #         for j, true in enumerate(truth[0]):
        #             if true == 1:
        #                 w_f.write(rev_classes[i] + "===")
        #         w_f.write("f1: " + str(tmp_f1) + ", recall: " + str(tmp_recall) + ", precision: " + str(tmp_precision))
        #         w_f.write("\n")

        precision = torch.mean(torch.FloatTensor(avg_precision)).item()
        recall = torch.mean(torch.FloatTensor(avg_recall)).item()
        if (recall + precision) != 0:
            F1 = 2*(precision * recall) / (recall + precision)
        else:
            F1 = 0.0

        res = {"precision": precision,
                "recall": recall,
                "F1": F1}
        print(res)

        return res



def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return ids, tensors, labels


def label_data(train, valid, test, labels):
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
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--classes_file", type=str, default="pkl/good_skills.p")
    parser.add_argument("--label_file", type=str, default="pkl/labels_skills.p")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--record_data", type=str, default=True)
    parser.add_argument("--model_type", type=str, default="skills")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20/skills')
    # parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/skills')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--enc_model", type=str, default="ft_pre_trained.bin")
    parser.add_argument("--sk_model", type=str, default="skills_bs128_lr0.001_max_ep_100__best_ep_97_ftpt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

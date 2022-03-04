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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss



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

        xp_title = "Sk eval " + suffix
        # quiviz.name_xp(xp_title)
        # quiviz.register(LinePlotObs())

        print("Loading data...")

        suffix = "fs"

        with open(os.path.join(args.DATA_DIR, "test_industry" + suffix + ".pkl"), "rb") as f:
            test_set = pkl.load(f)
        # with open(os.path.join(args.DATA_DIR, "classes_industry" + suffix + ".pkl"), "rb") as f:
        #     classes = pkl.load(f)

        print("Data loaded.")

        input_size = int(args.emb_size)
        classifier = IndustryClassifier(input_size, input_size, 150)
        weights = os.path.join(args.model_dir, args.ind_model)
        classifier.load_state_dict(torch.load(weights))

        dico = evaluate(classifier.cuda(), args, test_set)
        print(dico)
        with open(os.path.join(args.DATA_DIR, "res_ind" + suffix), 'wb') as f:
            pkl.dump(dico, f)

# @quiviz.log
def evaluate(classifier, args, test_set):
    dataloader_test = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate,
                                  shuffle=True, num_workers=0, drop_last=True)
    lab = []
    preds = []
    with ipdb.launch_ipdb_on_exception():
        true_positive = 0
        for ids, tensors, labels in tqdm(dataloader_test, desc="Evaluating..."):
            # if ids[0] == 100088:
            pred = classifier(torch.stack(tensors).cuda())
            preds.append(torch.argsort(pred, dim=-1, descending=True))
            lab.append(labels[0])
            #ipdb.set_trace()


        handle = "industry"
        num_c = range(150)
        best_pred = torch.cat(preds)[:, 0].cpu().numpy()
        res_dict = {
            "acc_" + handle: accuracy_score(lab, best_pred) * 100,
            "precision_" + handle: precision_score(lab, best_pred, average='weighted',
                                                   labels=num_c, zero_division=0) * 100,
            "recall_" + handle: recall_score(lab, best_pred, average='weighted', labels=num_c,
                                             zero_division=0) * 100,
            "f1_" + handle: f1_score(lab, best_pred, average='weighted', labels=num_c, zero_division=0) * 100}
        print(res_dict)
        ipdb.set_trace()
        ##### at 10
        out_predictions = []
        for index, pred in enumerate(torch.cat(preds)[:, :10]):
            if lab[index] in pred:
                out_predictions.append(lab[index])
            else:
                if type(pred[0]) == torch.Tensor:
                    out_predictions.append(pred[0].item())
                else:
                    out_predictions.append(pred[0].item())

        handle = "industry_@10"
        res_dict = {
            "acc_" + handle: accuracy_score(lab, out_predictions) * 100,
            "precision_" + handle: precision_score(lab, out_predictions, average='weighted',
                                                   labels=num_c, zero_division=0) * 100,
            "recall_" + handle: recall_score(lab, out_predictions, average='weighted', labels=num_c,
                                             zero_division=0) * 100,
            "f1_" + handle: f1_score(lab, out_predictions, average='weighted', labels=num_c, zero_division=0) * 100}
        print(res_dict)
        #     truth = []
        #     for i in range(len(ids)):
        #         truth.append(labels[i])
        #     for i, t in enumerate(pred):
        #         if pred[i].argmax().item() == truth[i]:
        #             true_positive += 1
        # acc = true_positive / len(dataloader_test)
    return {"accuracy": acc}


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return ids, tensors, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--label_file", type=str, default="pkl/labels.p")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--record_data", type=str, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_type", type=str, default="industry")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20/industry')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--bl", type=bool, default=True)
    parser.add_argument("--enc_model", type=str, default="ft_from_scratch.bin")
    parser.add_argument("--ind_model", type=str, default="industry_bs128_lr0.001_max_ep_100__best_ep_70fs")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

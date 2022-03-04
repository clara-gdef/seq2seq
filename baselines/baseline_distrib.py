import argparse
from torch.utils.data import DataLoader, Subset
import pickle as pkl
import os
from utils.Utils import collate_profiles_lj, compute_nll_eval, transform_indices
from utils.LinkedInDataset import LinkedInDataset
from tqdm import tqdm
import torch
import ipdb
from math import log

def eval_bl(args):
    print("Loading data...")
    with open(os.path.join(args.DATA_DIR, args.input_file), 'rb') as file:
        data = pkl.load(file)
    print("Data loaded.")

    dataset_test = LinkedInDataset(data["test"], transform_indices)

    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate_profiles_lj,
                                  shuffle=True, num_workers=2, drop_last=True)

    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    most_common_words(dataloader_test, index, args)


def most_common_words(dataloader, voc_index, args):
    ce_desc = []
    ce_title = []
    ce_overall = []
    file_name_suffix = "indices"
    # with open(os.path.join(args.DATA_DIR, "avg_len_indices.pkl"), "rb") as f:
    #     avg_lengths = pkl.load(f)
    with open(os.path.join(args.DATA_DIR, "word_dist_wo_st_" + file_name_suffix + ".pkl"), "rb") as f:
        distrib = pkl.load(f)

    # pred_title = torch.zeros(1, len(voc_index))
    # pred_desc = torch.zeros(1, len(voc_index))
    pred_overall = torch.zeros(1, len(voc_index))

    # for index in distrib["title"].keys():
    #     pred_title[0, index] = distrib["title"][index]
    # for index in distrib["desc"].keys():
    #     pred_desc[0, index] = distrib["desc"][index]

    ipdb.set_trace()

    for index in distrib["overall"].keys():
        pred_overall[0, index] = distrib["overall"][index]

    # pred_title = torch.log(pred_title).cuda()
    # pred_desc = torch.log(pred_desc).cuda()
    pred_overall = torch.log(pred_overall).cuda()

    rev_ind = {v: k for k, v in voc_index.items()}

    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader, desc="Computing most common baseline...") as pbar:
            for ids, profiles, profile_len, last_jobs, last_jobs_len in pbar:
                labels_title = []
                labels_desc = []

                flag_end_of_title = False
                for tok in last_jobs[0]:
                    if not flag_end_of_title:
                        labels_title.append(tok)
                        if tok == voc_index["EOT"]:
                            flag_end_of_title = True
                    else:
                        labels_desc.append(tok)

                labels_overall = torch.LongTensor(last_jobs[0])
                # labels_title = torch.LongTensor(labels_title)
                # labels_desc = torch.LongTensor(labels_desc)

                # pred_title_exp = pred_title.expand(len(labels_title), len(voc_index))
                # pred_desc_exp = pred_desc.expand(len(labels_desc), len(voc_index))
                pred_overall_exp = pred_overall.expand(len(labels_overall), len(voc_index))

                # loss_t = compute_nll_eval(pred_title_exp, labels_title)
                # loss_d = compute_nll_eval(pred_desc_exp, labels_desc)
                loss_overall = compute_nll_eval(pred_overall_exp, labels_overall)

                # if (float("inf") not in loss_d) and (float("inf") not in loss_t):
                #
                #     ce_title.extend(loss_t.tolist())
                #     ce_desc.extend(loss_d.tolist())
                #     ce_overall.extend(loss_overall.tolist())
                ce_overall.extend(loss_overall.tolist())

                if len(ce_desc) % 1000 == 0:
                    # ce_tensor_desc = torch.mean(torch.FloatTensor(ce_desc))
                    # ce_tensor_title = torch.mean(torch.FloatTensor(ce_title))
                    ce_tensor_overall = torch.mean(torch.FloatTensor(ce_overall))

                    # print("Perplexity TITLE for this split is: " + str(2 ** (ce_tensor_title / log(2))))
                    # print("Perplexity DESC for this split is: " + str(2 ** (ce_tensor_desc / log(2))))
                    print("Perplexity OVERALL for this split is: " + str(2 ** (ce_tensor_overall / log(2))))

                    # ipdb.set_trace()

                # if len(ce_desc) > 10:
                #    break

        ce_tensor_desc = torch.mean(torch.FloatTensor(ce_desc))
        ce_tensor_title = torch.mean(torch.FloatTensor(ce_title))

        print("Perplexity TITLE for this split is: " + str(2 ** (ce_tensor_title / log(2))))
        print("Perplexity DESC for this split is: " + str(2 ** (ce_tensor_desc / log(2))))
        print("Perplexity OVERALL for this split is: " + str(2 ** (ce_tensor_overall / log(2))))

        with open(os.path.join(args.DATA_DIR, "res_mc_s" + str(args.split) + ".pkl"), "wb") as f:
            pkl.dump({"Distrib desc baseline (perplexity) for split "+str(args.split): 2 ** (ce_tensor_desc / log(2)),
                      "Distrib title baseline (perplexity) for split " + str(args.split): 2 ** (ce_tensor_title / log(2)),
                      "Distrib overall baseline (perplexity) for split " + str(args.split): 2 ** (ce_tensor_overall / log(2))}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices_num.pkl")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_vocab_3j40k.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_vocab_3j40k.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="s2s")
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    eval_bl(args)

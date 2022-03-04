import os
import heapq
import argparse

from torch.utils.data import DataLoader
import pickle as pkl
from utils.Utils import collate_for_jobs
from utils.JobDataset import JobDataset
from operator import itemgetter
from tqdm import tqdm
import torch
import ipdb


def main(args):
    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")
    print("Loading splits")
    with open(os.path.join(args.DATA_DIR, "pkl/jobs_s" + str(args.split) + "_indices" + str(args.voc_size) + ".pkl"),
              'rb') as file:
        indices = pkl.load(file)

    all_indices = indices["train"][:]
    all_indices.extend(indices["valid"])
    all_indices.extend(indices["test"])
    dataset = JobDataset(datadict, all_indices)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_for_jobs,
                            shuffle=True, num_workers=0, drop_last=False)

    if args.fn == "avg_len":
        compute_avg_length(dataloader)
    if args.fn == "most_common":
        most_common_words(dataloader, args)


def most_common_words(dataloader, args):
    file_name_suffix = "indices"
    with ipdb.launch_ipdb_on_exception():
        avg_len_title = 4
        avg_len_desc = 36
        word_count_title = {}
        word_count_desc = {}
        EOT = 2
        for job in tqdm(dataloader, desc="Parsing all the tuples for most common words..."):
            flag_end_of_title = False
            for tok in job[0][0]:
                if not flag_end_of_title:
                    if tok in word_count_title.keys():
                        word_count_title[tok] += 1
                    else:
                        word_count_title[tok] = 1
                    if tok == EOT:
                        flag_end_of_title = True
                else:
                    if tok in word_count_desc.keys():
                        word_count_desc[tok] += 1
                    else:
                        word_count_desc[tok] = 1
        # word_count_title.pop(1)
        # word_count_title.pop(2)
        # word_count_desc.pop(3)
        # word_count_desc.pop(4)
        most_common_title = heapq.nlargest(avg_len_title, word_count_title.items(), key=itemgetter(1))
        most_common_desc = heapq.nlargest(avg_len_desc, word_count_desc.items(), key=itemgetter(1))
        dico = {"title": most_common_title,
                "desc": most_common_desc}
    
        if args.build_distribution:
            dist_title = {}
            dist_desc = {}
            for word in word_count_title.keys():
                dist_title[word] = word_count_title[word] / float(sum(word_count_title.values()))
            for word in word_count_desc.keys():
                dist_desc[word] = word_count_desc[word] / float(sum(word_count_desc.values()))
            distrib = {"title": dist_title,
                       "desc": dist_desc}
            with open(os.path.join(args.DATA_DIR, "word_dist_wo_st_" + file_name_suffix + ".pkl"),
                      "wb") as f:
                pkl.dump(distrib, f)
        
        with open(os.path.join(args.DATA_DIR, "most_common_" + file_name_suffix + ".pkl"), "wb") as f:
            pkl.dump(dico, f)


def compute_avg_length(dataloader):
    desc_len_list = []
    title_len_list = []
    EOT = 2
    for job in tqdm(dataloader, desc="Parsing all the tuples for average length..."):
        counter_title = 0
        counter_desc = 0
        flag_end_of_title = False
        for tok in job[0][0]:
            if not flag_end_of_title:
                counter_title += 1
                if tok == EOT:
                    flag_end_of_title = True
            else:
                counter_desc += 1
        title_len_list.append(counter_title - 2)
        desc_len_list.append(counter_desc - 2)
    avg_len_title = torch.mean(torch.FloatTensor(title_len_list))
    avg_len_desc = torch.mean(torch.FloatTensor(desc_len_list))
    file_name_suffix = "indices"
    print("average title len: " + str(avg_len_title))
    print("average desc len: " + str(avg_len_desc))
    dico = {"title": avg_len_title,
            "desc": avg_len_desc}
    with open(os.path.join(args.DATA_DIR, "avg_len_" + file_name_suffix + ".pkl"), "wb") as f:
        pkl.dump(dico, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/indices_all_wo_padding.pkl")
    parser.add_argument("--fn", type=str, default="most_common")
    parser.add_argument("--avg_len", type=int, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--build_distribution", type=bool, default=True)
    args = parser.parse_args()
    main(args)

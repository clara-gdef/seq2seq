import argparse
import os
import pickle as pkl

import ipdb
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pre_proc.from_indices_to_words import decode_indices
from utils.LinkedInDataset import LinkedInDataset
from utils.Utils import collate_profiles_lj, transform_indices, separate_title_from_desc


def eval_bl(args):
    print("Loading splits")
    with open(os.path.join(args.DATA_DIR, "pkl/ppl_s" + str(args.split) + "_indices40k.pkl"), 'rb') as file:
        indices = pkl.load(file)
    with open(os.path.join(args.DATA_DIR, args.input_file), 'rb') as file:
        data = pkl.load(file)

    all_indices = indices["train"][:]
    all_indices.extend(indices["valid"])
    all_indices.extend(indices["test"])
    dataset = LinkedInDataset(all_indices, data["data"], data["lengths"], transform_indices)

    dataloader_test = DataLoader(Subset(dataset, indices["test"]), batch_size=args.batch_size,
                                 collate_fn=collate_profiles_lj,
                                 shuffle=True, num_workers=0, drop_last=True)
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)
        prev_job(dataloader_test, index, args)


def prev_job(dataloader, voc_index, args):
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader, desc="Computing prev_job baseline...") as pbar:
            for ids, profiles, profile_len, last_jobs, last_jobs_len in pbar:
                last_job = last_jobs[0]
                profile = profiles[0]
                if len(profile) > 0:

                    prev_job = profile[0]

                    labels_title, labels_desc = separate_title_from_desc(last_job, voc_index["EOT"])

                    label_title = torch.LongTensor(labels_title)
                    label_desc = torch.LongTensor(labels_desc)

                    profile_title, profile_desc = separate_title_from_desc(prev_job, voc_index["EOT"])
                    profile_title = torch.LongTensor(profile_title)
                    profile_desc = torch.LongTensor(profile_desc)
                    prev_job = torch.LongTensor(prev_job)
                    last_job = torch.LongTensor(last_job)


                    pred_file_title = os.path.join(args.DATA_DIR, "results/prev_job_title_pred.txt")
                    pred_text_title = decode_indices(profile_title, voc_index)
                    pred_text_title += "\n"
                    with open(pred_file_title, "a") as pf:
                        pf.write(pred_text_title)
                    label_file_title = os.path.join(args.DATA_DIR, "results/prev_job_title_label.txt")
                    label_text_title = decode_indices(label_title, voc_index)
                    label_text_title += "\n"
                    with open(label_file_title, "a") as pf:
                        pf.write(label_text_title)

                    pred_file_desc = os.path.join(args.DATA_DIR, "results/prev_job_desc_title.txt")
                    pred_text_desc = decode_indices(profile_desc, voc_index)
                    pred_text_desc += "\n"
                    with open(pred_file_desc, "a") as pf:
                        pf.write(pred_text_desc)
                    label_file_desc = os.path.join(args.DATA_DIR, "results/prev_job_desc_label.txt")
                    label_text_desc = decode_indices(label_desc, voc_index)
                    label_text_desc += "\n"
                    with open(label_file_desc, "a") as pf:
                        pf.write(label_text_desc)

                    pred_file_overall = os.path.join(args.DATA_DIR, "results/prev_job_pred.txt")
                    pred_text_overall = decode_indices(prev_job, voc_index)
                    pred_text_overall += "\n"
                    with open(pred_file_overall, "a") as pf:
                        pf.write(pred_text_overall)
                    label_file_overall = os.path.join(args.DATA_DIR, "results/prev_job_label.txt")
                    label_text_overall = decode_indices(last_job, voc_index)
                    label_text_overall += "\n"
                    with open(label_file_overall, "a") as lf:
                        lf.write(label_text_overall)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/ppl_indices.pkl")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_vocab_3j.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="s2s")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    eval_bl(args)

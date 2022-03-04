import argparse
import json
import os
import ipdb
import pickle as pkl
from tqdm import tqdm
import torch


def main(args):
    main_dict = {}
    src_file = os.path.join(args.DATA_DIR, args.file_to_decode)
    tgt_file = os.path.join(args.DATA_DIR, args.tgt_file)
    index_file = os.path.join(args.DATA_DIR, args.index_file)
    with open(index_file, "rb") as i_f:
        index = pkl.load(i_f)
    rev_index = {v: k for k, v in index.items()}
    with open(src_file, "rb") as s_f:
        data = pkl.load(s_f)
    with ipdb.launch_ipdb_on_exception():
        for k in tqdm(data.keys()):
            pred_decoded = decode_indices(data[k]["pred"], rev_index)
            label_decoded = decode_indices(data[k]["label"], rev_index)
            main_dict[k] = {"pred": pred_decoded, "label": label_decoded}
    with open(tgt_file, "w") as t_f:
        json.dump(main_dict, t_f)


def decode_indices(indices, index):
    out = ""
    for indice in indices:
        if indice.item() in index.keys():
            out += index[indice.item()]
            out += " "
        else:
            out += "ERROR "
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_to_decode", type=str, default="results/eval_output_v1_splitless.txt")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--tgt_file", type=str, default="v1_splitless.json")
    args = parser.parse_args()
    main(args)

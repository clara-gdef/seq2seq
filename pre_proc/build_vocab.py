import os
import pickle as pkl
import argparse
import torch
from tqdm import tqdm
import fastText
import ipdb
from collections import Counter
import re


def main(args):
    print("Loading word vectors...")
    ft = fastText.load_model(os.path.join(args.DATA_DIR, "ft_pre_trained.bin"))
    print("Word vectors loaded.")

    print("Laoding all jobs...")
    with open(os.path.join(args.DATA_DIR, args.input_file), 'rb') as f:
        data = pkl.load(f)
    print("Jobs loaded.")

    word_list = build_word_set(data, args)
    build_index_and_tensor(word_list, ft, args)


def build_word_set(data, args):
    jobs = data["train_data"]
    jobs.extend(data['valid_data'])
    word_count = Counter()
    number_regex = re.compile(r'\d+(,\d+)?') # match 3
    with tqdm(total=len(jobs), desc="Counting words...") as pbar:
        for person in jobs:
            for word in person["position"]:
                if re.match(number_regex, word):
                    word_count["NUM"] += 1
                else:
                    word_count[word] += 1
            for word in person["description"]:
                if re.match(number_regex, word):
                    word_count["NUM"] += 1
                else:
                    word_count[word] += 1
            pbar.update(1)

    word_count.pop("")
    ordered_words = sorted([(v, k) for k, v in word_count.items()], reverse=True)

    if len(ordered_words) > args.max_voc_len:
        keys = ordered_words[:args.max_voc_len]
    else:
        keys = ordered_words

    word_list = [x[1] for x in keys]

    with open(os.path.join(args.DATA_DIR, "pkl/vocab" + str(args.max_voc_len) + ".pkl"), "wb") as f:
        pkl.dump(word_list, f)
    return word_list


def build_index_and_tensor(word_list, ft, args):
    word_to_index = dict()
    print("Length of the vocabulary: " + str(len(word_list)))
    with tqdm(total=len(word_list), desc="Building tensors and index...") as pbar:
        tensor_updated, w2i_updated, num_tokens = build_special_tokens(word_to_index)
        for i, word in enumerate(word_list):
            if word is not '':
                tensor_updated = torch.cat([tensor_updated, torch.FloatTensor(ft.get_word_vector(word)).view(1, -1)], dim=0)
                w2i_updated[word] = i + num_tokens
            pbar.update(1)
    print(len(word_to_index))
    with open(os.path.join(args.DATA_DIR, "pkl/tensor_40k.pkl"), "wb") as f:
        pkl.dump(tensor_updated, f)
    with open(os.path.join(args.DATA_DIR, "pkl/index_40k.pkl"), "wb") as f:
        pkl.dump(w2i_updated, f)


def build_special_tokens(word_to_index):
    """
    SOT stands for 'start of title'
    EOT stands for 'end of title'
    SOD stands for 'start of description'
    EOD stands for 'end of description'
    PAD stands for 'padding index'
    UNK stands for 'unknown word'
    """
    SOT = torch.randn(1, 300)
    EOT = torch.randn(1, 300)
    SOD = torch.randn(1, 300)
    EOD = torch.randn(1, 300)
    PAD = torch.randn(1, 300)
    UNK = torch.randn(1, 300)
    word_to_index["PAD"] = 0
    tensor = PAD
    word_to_index["SOT"] = 1
    tensor = torch.cat([tensor, SOT], dim=0)
    word_to_index["EOT"] = 2
    tensor = torch.cat([tensor, EOT], dim=0)
    word_to_index["SOD"] = 3
    tensor = torch.cat([tensor, SOD], dim=0)
    word_to_index["EOD"] = 4
    tensor = torch.cat([tensor, EOD], dim=0)
    word_to_index["UNK"] = 5
    tensor = torch.cat([tensor, UNK], dim=0)
    return tensor, word_to_index, 6


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/jobs.pkl")
    parser.add_argument("--model_version", type=str, default='s2s')
    parser.add_argument("--DATA_DIR", type=str, default='/local/gainondefor/work/data')
    parser.add_argument("--pre_trained_model", type=str, default='ft_pre_trained.bin')
    parser.add_argument("--max_voc_len", type=int, default=40000)
    parser.add_argument("--min_occurence", type=int, default=5)
    args = parser.parse_args()
    main(args)


import argparse
import os
from tqdm import tqdm
import glob
import pickle as pkl


def build_dict(args):
    if args.vocab_file == "scowl":
        build_dict_from_scowl(args)
    else:
        word_list = []
        data_file = "/local/gainondefor/work/lip6/data/seq2seq/words.txt"
        with open(data_file, "r") as f:
            tmp = f.read().split("\n")
            word_l = [i.strip() for i in tmp]
            word_list.extend(word_l)
        print(len(word_list))
        sorted_list = sorted(list(set(word_list)))
        with open(args.target_file, "wb") as ft:
            pkl.dump(sorted_list, ft)


def build_dict_from_scowl(args):
    word_list = []
    df_am = glob.glob(args.DATA_DIR + "/american-words.10")
    df_aus = glob.glob(args.DATA_DIR + "/australian-words.10")
    df_br = glob.glob(args.DATA_DIR + "/british-words.10")
    df_en = glob.glob(args.DATA_DIR + "/english-words.10")
    df = df_am
    df.extend(df_aus)
    df.extend(df_br)
    df.extend(df_en)
    for file in tqdm(df):
        with open(file, "r", encoding="ISO-8859-1") as f:
            string = f.read().split("\n")
            if len(string) > 1 and type(string) is list:
                word_list.extend([e.lower() for e in string])
            else:
                word_list.append(string.lower())
    print(len(word_list))
    sorted_list = sorted(list(set(word_list)))
    with open(args.target_file, "wb") as ft:
        pkl.dump(sorted_list, ft)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default='../data/scowl-2018.04.16/final')
    parser.add_argument("--target_file", type=str, default="../data/en_dict.pkl")
    parser.add_argument("--vocab_file", type=str, default="scowl")
    args = parser.parse_args()
    build_dict(args)

import argparse
import os
import pickle as pkl


def main(args):
    word_dict = dict()
    char_dict = dict()
    with open(args.input_word_file, "r") as f1:
        word_id = f1.read()
    with open(args.input_char_file, "r") as f2:
        char_id = f2.read()

    word_list = word_id.split("\n")
    char_list = char_id.split("\n")

    for i in word_list:
        if len(i) > 0:
            tmp = i.split("\t")
            word_dict[tmp[0]] = tmp[1]

    for i in char_list:
        if len(i) > 0:
            tmp = i.split("\t")
            char_dict[tmp[0]] = tmp[0]

    with open(os.path.join(args.DATA_DIR, args.target_word_file), "wb") as f:
        pkl.dump(word_dict, f)

    with open(os.path.join(args.DATA_DIR, args.target_char_file), "wb") as f:
        pkl.dump(char_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_word_file", type=str, default="ELMo/word.dic")
    parser.add_argument("--input_char_file", type=str, default="ELMo/char.dic")
    parser.add_argument("--DATA_DIR", type=str, default="data/")
    parser.add_argument("--target_word_file", type=str, default="pkl/word_index.pkl")
    parser.add_argument("--target_char_file", type=str, default="pkl/char_index.pkl")
    args = parser.parse_args()

    main(args)

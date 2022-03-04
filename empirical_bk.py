import pickle as pkl
import argparse
import os
from collections import Counter
from tqdm import tqdm
import ipdb


def main(args):
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    dataset = datadict["train"]["data"]
    dataset.extend(datadict["valid"]["data"])
    word_count = Counter()

    for word_list in tqdm(dataset):
        for word in word_list:
            word_count[word] += 1

    ipdb.set_trace()
    print("hello")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/indices_jobs.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    args = parser.parse_args()
    main(args)

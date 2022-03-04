import argparse
import os
import pickle as pkl
import random
from tqdm import tqdm
import ipdb


def main(args):
    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")
    data_test = []
    data_valid = []
    data_train = []
    # ipdb.set_trace()
    for d in tqdm(datadict["data"]):
        if random.random() > args.ratio:
            if random.random() > .5:
                data_test.append(d[:2])
            else:
                data_valid.append(d[:2])
        else:
            data_train.append(d[:2])

    with open(os.path.join(args.DATA_DIR, args.output_train), "wb") as f:
        pkl.dump({"train_data": data_train, "valid_data": data_valid}, f)
    with open(os.path.join(args.DATA_DIR, args.output_test), "wb") as f:
        pkl.dump(data_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/bp_fr_desc_3j.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--ratio", type=float, default=.8)
    parser.add_argument("--output_train", type=str, default="people_train.pkl")
    parser.add_argument("--output_test", type=str, default="people_test.pkl")
    args = parser.parse_args()
    main(args)

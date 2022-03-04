import argparse
from tqdm import tqdm
import os
import pickle as pkl


def main(args):
    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")
    for i in range(5):
        main_for_one_split(datadict, i, args)


def main_for_one_split(datadict, split, args):
    train, test = [], []
    for idx, s in tqdm(enumerate(datadict['splits']), total=len(datadict['splits']), desc="Building train/test of split #{}".format(split)):
        if s == split:
            test.append(idx)
        else:
            train.append(idx)

    validation = .5
    print("Building validation...")
    val_len = int(validation * len(test))
    validation = test[-val_len:]
    test = test[:-val_len]
    print("Validation built...")

    output = {"train": train, "valid": validation, "test": test}

    with open(os.path.join(args.DATA_DIR, "pkl/" + str(args.output_prefix) + "_s" + str(split) + "_indices" + str(args.voc_size) + ".pkl"), "wb") as f:
        pkl.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--output_prefix", type=str, default="ppl")
    args = parser.parse_args()
    main(args)

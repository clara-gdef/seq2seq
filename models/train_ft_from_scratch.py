import argparse
import fastText
import os
import ipdb
import pickle as pkl
from tqdm import tqdm


def main(args):
    with ipdb.launch_ipdb_on_exception():
        print("Loading data...")
        data_file = os.path.join(args.DATA_DIR, args.input_file)
        with open(data_file, 'rb') as file:
            datadict = pkl.load(file)
        print("Data loaded")
        train_set = datadict["train_data"]
        del datadict
        train_file = os.path.join(args.DATA_DIR, args.train_file)
        if args.build_tgt_file:
            with open(train_file, 'a+') as f:
                for person in tqdm(train_set["data"]):
                    for job in person[1]:
                        for word in job:
                            f.write(word + " ")
                    f.write("\n")
        model = fastText.train_unsupervised(train_file, dim=300)
        ipdb.set_trace()
        model_file = os.path.join(args.model_dir, args.ft_model)
        model.save_model(model_file)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--input_file", type=str, default="pkl/profiles_elmo.pkl")
    parser.add_argument("--build_tgt_file", type=bool, default=True)
    parser.add_argument("--train_file", type=str, default="pkl/train_ft.txt")
    parser.add_argument("--ft_model", type=str, default='ft_from_scratch.bin')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20')

    args = parser.parse_args()
    main(args)

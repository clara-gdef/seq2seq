import argparse
import itertools
import os
import pickle as pkl

from tqdm import tqdm


def main(args):
    # loading data
    print("Loading data...")
    # train_file = os.path.join(args.DATA_DIR, args.train_file)
    # tgt_train_file = os.path.join(args.DATA_DIR, args.tgt_train_file)
    # flag_err = False
    #
    # num_line_train = 0
    # with open(train_file, "rb") as f:
    #     while not flag_err:
    #         try:
    #             person = list(pkl.load(f))
    #             num_line_train += 1
    #         except EOFError:
    #             flag_err = True
    #             continue
    #
    # flag_err = False
    # with open(train_file, "rb") as f:
    #     pbar = tqdm(f, total=num_line_train)
    #     with open(tgt_train_file, "ab") as f2:
    #         while not flag_err:
    #             try:
    #                 person = list(pkl.load(f))
    #                 pkl.dump((person[0], [job.cpu() for job in person[1]]), f2)
    #                 pbar.update(1)
    #             except EOFError:
    #                 flag_err = True
    #                 continue
    # print("Train file loaded.")
    #
    # valid_file = os.path.join(args.DATA_DIR, args.valid_file)
    # tgt_valid_file = os.path.join(args.DATA_DIR, args.tgt_valid_file)
    # flag_err = False
    #
    # num_line_val = 0
    # with open(valid_file, "rb") as f:
    #     while not flag_err:
    #         try:
    #             person = list(pkl.load(f))
    #             num_line_val += 1
    #         except EOFError:
    #             flag_err = True
    #             continue
    #
    # flag_err = False
    # with open(valid_file, "rb") as f:
    #     pbar = tqdm(f, total=num_line_val)
    #     with open(tgt_valid_file, 'ab') as f2:
    #         while not flag_err:
    #             try:
    #                 person = list(pkl.load(f))
    #                 pkl.dump((person[0], [job.cpu() for job in person[1]]), f2)
    #                 pbar.update(1)
    #             except EOFError:
    #                 flag_err = True
    #                 continue
    # print("Valid file loaded.")
    test_file = os.path.join(args.DATA_DIR, args.test_file)
    tgt_test_file = os.path.join(args.DATA_DIR, args.tgt_test_file)
    flag_err = False

    num_line_val = 0
    with open(test_file, "rb") as f:
        while not flag_err:
            try:
                person = list(pkl.load(f))
                num_line_val += 1
            except EOFError:
                flag_err = True
                continue

    flag_err = False
    with open(test_file, "rb") as f:
        pbar = tqdm(f, total=num_line_val)
        with open(tgt_test_file, 'ab') as f2:
            while not flag_err:
                try:
                    person = list(pkl.load(f))
                    pkl.dump((person[0], [job.cpu() for job in person[1]]), f2)
                    pbar.update(1)
                except EOFError:
                    flag_err = True
                    continue
    print("test file loaded.")
    print("Data loaded.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="pkl/prof_rep_elmo_train.pkl")
    parser.add_argument("--valid_file", type=str, default="pkl/prof_rep_elmo_valid.pkl")
    parser.add_argument("--test_file", type=str, default="pkl/prof_rep_elmo_test.pkl")
    parser.add_argument("--tgt_train_file", type=str, default="pkl/prof_rep_elmo_train_cpu.pkl")
    parser.add_argument("--tgt_valid_file", type=str, default="pkl/prof_rep_elmo_valid_cpu.pkl")
    parser.add_argument("--tgt_test_file", type=str, default="pkl/prof_rep_elmo_test_cpu.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--model_type", type=str, default="elmo_w2v_test")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tf", type=float, default=0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--dpo", type=float, default=.0)
    parser.add_argument("--bidirectional", type=bool, default=True)
    args = parser.parse_args()
    main(args)
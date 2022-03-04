import argparse
import pickle as pkl
import os
from tqdm import tqdm
from random import shuffle
import ipdb


def load_bypath_data(args):
    data_file_train = os.path.join(args.DATA_DIR, args.input_train)

    jobs_train = []
    jobs_valid = []
    jobs_test = []

    with open(data_file_train, 'rb') as file:
        data = pkl.load(file)

    with tqdm(data["train_data"]) as pbar:
        for person in pbar:
            for experience in person[1]:
                jobs_train.append({'position': experience["position"],
                                   'description': experience["description"]})
    with tqdm(data["valid_data"]) as pbar:
        for person in pbar:
            for experience in person[1]:
                jobs_valid.append({'position': experience["position"],
                                   'description': experience["description"]})
    with tqdm(data["test_data"]) as pbar:
        for person in pbar:
            for experience in person[1]:
                jobs_test.append({'position': experience["position"],
                                  'description': experience["description"]})
    return jobs_train, jobs_valid, jobs_test


def build_dataset(args):
    train, valid, test = load_bypath_data(args)

    print('Number of jobs created in train: ' + str(len(train)))
    print('Number of jobs created in test: ' + str(len(test)))
    print('Number of jobs created in valid: ' + str(len(valid)))

    shuffle(train)
    shuffle(valid)
    shuffle(test)

    return {"train_data": train, "valid_data": valid, "test_data": test}


def main(args):
    DATA_DIR = args.DATA_DIR
    ds = build_dataset(args)
    with open(os.path.join(DATA_DIR, args.output), "wb") as f:
        pkl.dump(ds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train", type=str, default="pkl/people_edu_sk_ind.pkl")
    parser.add_argument("--output", type=str, default="pkl/jobs_edu_sk_ind.pkl")
    parser.add_argument("--nb_splits", type=int, default=5)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--MIN_SEQ_LEN", type=int, default=5)
    parser.add_argument("--language", type=str, default="fr")

    args = parser.parse_args()

    main(args)

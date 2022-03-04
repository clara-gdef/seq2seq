import argparse
import pickle as pkl
import os
import ipdb
import spacy
import json
from tqdm import tqdm
from random import random, shuffle
from collections import Counter


def to_array_comp(doc):
        return [w.orth_ for w in doc]


def custom_pipeline(nlp):
    return (nlp.tagger, to_array_comp)


def load_bypath_data(args, nlp):
    id_counter = 0
    data_file = os.path.join(args.DATA_DIR, args.input)
    tp_list = []
    with open(data_file, 'r') as file:
        print(data_file)
        num_lines = sum(1 for line in file)
    with open(data_file, 'r') as file:
        pbar = tqdm(file, total=num_lines)
        for line in pbar:
            person = json.loads(line)
            jobs = []
            for job in person[1]:
                if len(job['description']) > 0:
                    pos = tokenize_list([job['position']], nlp)
                    desc = tokenize_list([job['description']], nlp)
                    jobs.append({'position': pos,
                                 'description': desc})
            education = []
            for step in person[3]:
                deg = tokenize_list([step['degree']], nlp)
                inst = tokenize_list([step['institution']], nlp)
                education.append({'degree': deg,
                                  'institution': inst})

            skills = tokenize_list(person[2], nlp)

            industry = person[4]
            tp_list.append((person[0], jobs, education, skills, industry))
            pbar.update(1)
            id_counter += 1
    for tup in tqdm(tp_list, desc="Removing empty jobs..."):
        for job in tup[1]:
            if len(job) < 1 or (len(job["position"]) + len(job["description"])) < args.MIN_SEQ_LEN:
                tup[1].remove(job)
    return tp_list


def build_dataset(args):
    nlp = spacy.load(args.language, create_pipeline=custom_pipeline)

    tp_list = load_bypath_data(args, nlp)

    data = [person for person in tp_list if (len(person[1]) > 0) and (len(person[2]) > 0) and (len(person[3]) > 0) and (len(person[4]) > 0)]
    data_test = []
    data_valid = []
    data_train = []
    tmp = []

    print('Number of tuples created: ' + str(len(data)))

    print("Length of data : " + str(len(data)))
    shuffle(data)

    for d in data:
        if random() > args.split_ratio:
            data_test.append(d)
        else:
            tmp.append(d)
    for d in tmp:
        if random() > args.split_ratio:
            data_valid.append(d)
        else:
            data_train.append(d)

    return {"train_data": data_train, "valid_data": data_valid, "test_data": data_test}


def tokenize_list(sentence_list, nlp):
    word_list = []
    # Check that it's not empty
    if len(sentence_list) > 0:
        for job in nlp.tokenizer.pipe((j for j in sentence_list), batch_size=1000000, n_threads=8):
            for word in job:
                w = str(word)
                word_list.append(w.lower().strip())
    return word_list


def main(args):
    with ipdb.launch_ipdb_on_exception():
        DATA_DIR = args.DATA_DIR
        ds = build_dataset(args)
        with open(os.path.join(DATA_DIR, args.output), "wb") as f:
            pkl.dump(ds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="bp_3jobs_desc_edu_skills_industry_FR_indexed.json")
    parser.add_argument("--output", type=str, default="pkl/people_edu_sk_ind.pkl")
    parser.add_argument("--split_ratio", type=float, default=.8)
    parser.add_argument("--DATA_DIR", type=str, default=os.path.realpath('../data'))
    parser.add_argument("--MIN_SEQ_LEN", type=int, default=1)
    parser.add_argument("--language", type=str, default="fr")

    args = parser.parse_args()

    main(args)

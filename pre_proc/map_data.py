import ipdb
import pickle as pkl
import argparse
import json
import spacy
from tqdm import tqdm
from os.path import join
import os
import gzip


def to_array_comp(doc):
    return [w.orth_ for w in doc]


def custom_pipeline(nlp):
    return (nlp.tagger, to_array_comp)


def main(args):
    # nlp = spacy.load("fr", create_pipeline=custom_pipeline)
    print("getting indices...")
    train, valid, test = get_indices(args)
    print("indices gotten!")

    tgt_file_train = join(args.DATA_DIR, args.target_train_file)
    tgt_file_valid = join(args.DATA_DIR, args.target_valid_file)
    tgt_file_test = join(args.DATA_DIR, args.target_test_file)

    # src_file = os.path.join(args.DATA_DIR, args.pkl_to_split)
    # with open(src_file, 'rb') as f:
    #     src_data = pkl.load(f)

    # data = src_data['train_data']
    # data.extend(src_data['test_data'])
    #
    # datadict = dict()
    # for item in data:
    #     datadict[item[0]] = item

    # mapping = map_json_files(args)
    # tgt_file = join(args.DATA_DIR, "mapping.pkl")
    # with open(tgt_file, 'wb') as f:
    #     mapping = pkl.load(f)
    # ipdb.set_trace()
    jfe = join(args.DATA_DIR, args.json_file_extended)

    with open(jfe, 'r') as j_file:
        num_lines = sum(1 for line in j_file)

    # ipdb.set_trace()

    # tgt_file_train = os.path.join(args.DATA_DIR, "pkl/people_index.pkl")
    # tgt_file_test = os.path.join(args.DATA_DIR, "pkl/people_index_test.pkl")

    # train_data = []
    # for index in tqdm(train):
    #     train_data.append(datadict[index])
    # valid_data = []
    # for index in tqdm(valid):
    #     valid_data.append(datadict[index])
    # test_data = []
    # for index in tqdm(test):
    #     test_data.append(datadict[index])

    # with open(tgt_file_train, 'wb') as f:
    #     pkl.dump({"train": train_data, "valid": valid_data}, f)
    #
    # with open(tgt_file_test, 'wb') as f:
    #     pkl.dump({"test": test_data}, f)

    with open(tgt_file_test, 'a+') as test_tgt:
        with open(tgt_file_train, 'a+') as train_tgt:
            with open(tgt_file_valid, "a+") as valid_tgt:
                with open(jfe, 'r') as j_file:
                    pbar = tqdm(j_file, total=num_lines, desc="splitting json file into train, valid and test...")
                    for line in pbar:
                        liste = json.loads(line)
                        if liste[0] in train:
                            string = json.dumps(liste) + "\n"
                            train_tgt.write(string)
                        if liste[0] in valid:
                            string = json.dumps(liste) + "\n"
                            valid_tgt.write(string)
                        if liste[0] in test:
                            string = json.dumps(liste) + "\n"
                            test_tgt.write(string)

#
# def map_json_files(args):
#     mapping = dict()
#     jf = join(args.DATA_DIR, args.json_file)
#     jfe = join(args.DATA_DIR, args.json_file_extended)
#     ipdb.set_trace()
#     with open(jfe, 'r') as j_file_extended:
#         num_lines = sum(1 for line in j_file_extended)
#     with open(jf, 'r') as j_file:
#         with open(jfe, 'r') as j_file_extended:
#             pbar = tqdm(j_file_extended, total=num_lines, desc="Mapping json files...")
#             for line_e in pbar:
#                 liste_extended = json.loads(line_e)
#                 for line in j_file:
#                     liste = json.loads(line)
#                     if liste_extended[1:-1] == liste[1:]:
#                         mapping[liste[0]] = liste_extended[0]
#     tgt_file = join(args.DATA_DIR, "mapping.pkl")
#     with open(tgt_file, 'wb') as f:
#         pkl.dump(mapping, f)
#     return mapping
#
#
# def get_indices(args):
#     with open(join(args.DATA_DIR, "pkl/people_s0_indices.pkl"), 'rb') as file:
#         indices = pkl.load(file)
#     with open(join(args.DATA_DIR, args.input_indices), 'rb') as file:
#         data = pkl.load(file)
#
#     ind_train = indices["train"][:]
#     ind_valid = indices["valid"][:]
#     ind_test = indices["test"][:]
#
#     identifiers_train = []
#     identifiers_valid = []
#     identifiers_test = []
#
#     for i in ind_train:
#         identifiers_train.append(data["data"][i][0])
#     for i in ind_valid:
#         identifiers_valid.append(data["data"][i][0])
#     for i in ind_test:
#         identifiers_test.append(data["data"][i][0])
#
#     return identifiers_train, identifiers_valid, identifiers_test


def get_indices(args):
    train_file = os.path.join(args.DATA_DIR, args.train_file)
    valid_file = os.path.join(args.DATA_DIR, args.valid_file)
    test_file = os.path.join(args.DATA_DIR, args.test_file)
    with open(train_file, 'rb') as train_f:
        train = pkl.load(train_f)
    with open(valid_file, 'rb') as valid_f:
        valid = pkl.load(valid_f)
    with open(test_file, 'rb') as test_f:
        test = pkl.load(test_f)

    return train, valid, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file_extended", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR.json")
    parser.add_argument("--train_file", type=str, default="pkl/train.p")
    parser.add_argument("--valid_file", type=str, default="pkl/valid.p")
    parser.add_argument("--test_file", type=str, default="pkl/test.p")
    parser.add_argument("--pkl_to_split", type=str, default="pkl/people_industry_indexed.pkl")
    parser.add_argument("--json_file", type=str, default="bp_3jobs_desc_edu_skills_FR.json")
    parser.add_argument("--folder", type=str, default="/local/gainondefor/work/lip6/data/bypath")
    parser.add_argument("--target_train_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR_TRAIN.json")
    parser.add_argument("--target_valid_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR_VALID.json")
    parser.add_argument("--target_test_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR_TEST.json")
    parser.add_argument("--input_indices", type=str, default="pkl/people_industry_indexed.pkl.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    args = parser.parse_args()
    main(args)

#
# counter_index = 0
# src_file = "/local/gainondefor/work/lip6/data/seq2seq/bp_3jobs_desc_edu_skills_industry_FR2.json"
# tgt_file = "/local/gainondefor/work/lip6/data/seq2seq/bp_3jobs_desc_edu_skills_industry_FR_indexed.json"
# with open(src_file, 'r') as f1:
#     num_lines = sum(1 for line in f1)
# with open(src_file, 'r') as f1:
#     with open(tgt_file, 'a') as f2:
#         pbar = tqdm(f1, total=num_lines)
#         for line in pbar:
#             current_person = json.loads(line)
#             current_person[0] = counter_index
#             counter_index += 1
#             string = json.dumps(current_person) + '\n'
#             f2.write(string)
#
#
#
#


import os
import argparse
import gzip
import json
import pickle as pkl
from tqdm import tqdm
import re
# from nltk import wordpunct_tokenize
# from nltk.corpus import stopwords


def subsample_json_files(args):
    with open(args.dict, "rb") as f:
        dictionary = pkl.load(f)
    folder = args.DATA_DIR
    with open(args.target_file_en, 'a+') as fr_file:
        for file in os.listdir(folder):
            if file.endswith('.jl.gz'):
                print("Processing file " + str(file))
                file_path = os.path.join(folder, file)
                with gzip.open(file_path, 'r') as file:
                    num_lines = sum(1 for line in file)
                with gzip.open(file_path, 'r') as f:
                    pbar = tqdm(f, total=num_lines)
                    counter_fr = 0
                    counter_en = 0
                    for line in pbar:
                        current_person = json.loads(line)
                        if all(k in current_person.keys() for k in ('experience', 'education')):
                            if len(current_person['experience']) > 4:
                                tup = from_dict_to_tuple(current_person, counter_fr)
                                if len(tup[1]) > 4:
                                    language = detect_language(tup, dictionary)
                                    string = json.dumps(tup) + '\n'
                                    if str(language) == 'english':
                                        fr_file.write(string)
                                        counter_en += 1
                        pbar.update(1)
                    print("Proportion of english tuples is " + str(100 * counter_fr / num_lines))


def from_dict_to_tuple(dictionary, index):
    job_list = [{'position': job['position'],
                 'from_ts': job['from_ts'],
                 'from': job["from"],
                 'to': job['to'],
                 'description': job["description"]}
                if all(k in job.keys() for k in ('position', 'from_ts', 'from', 'to')) and "description" in job.keys()
                else {'position': job['position'],
                      'from_ts': job['from_ts'],
                      'from': job["from"],
                      'to': job['to'],
                      'description': []}
                for job in dictionary['experience']
                if all(k in job.keys() for k in ('position', 'from_ts', 'from', 'to'))]
    if "skills" in dictionary.keys():
        skills = [skill for skill in dictionary["skills"]]
    else:
        skills = []
    education = [{'degree': degree["degree"],
                  'institution': degree['name'],
                  'from': degree['from'],
                  'to': degree['to']}
                 for degree in dictionary["education"] if
                 all(k in degree.keys() for k in ('degree', 'name', 'from', 'to'))]
    ordered_job_list = sorted(job_list, key=lambda tup: - tup['from_ts'])
    ordered_education = sorted(education, key=lambda tup: - int(tup['to']))
    return (index, ordered_job_list, skills, ordered_education)


def detect_language(text, dictionary):
    tmp = []
    profile = []
    for exp in text[1]:
        profile.extend([i for i in exp["position"].split(" ")])
        if len(exp["description"]) > 0:
            tmp.extend([i for i in exp["description"].split(" ")])
    for j in tmp:
        profile.extend(re.split('(\W+)', j))

    # this is an arbitrary limit, meaning we consider as english the profiles containing at least 21 words of our english dictionary
    if len(set(dictionary) & set(profile)) > 20:
        return "english"
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/lip6/data/bypath")
    parser.add_argument("--dict", type=str, default="../data/en_dict.pkl")
    parser.add_argument("--target_file_fr", type=str, default="../data/fr.json")
    parser.add_argument("--target_file_en", type=str, default="../data/en.json")
    args = parser.parse_args()
    subsample_json_files(args)


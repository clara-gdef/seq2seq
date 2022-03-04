import os
import argparse
import gzip
import json
import pickle as pkl
import ipdb
from tqdm import tqdm
import numpy as np


def main(args):
    with open(args.target_file, "rb") as f:
        cie_dict = pkl.load(f)


def build_dict(args):
    folder = args.DATA_DIR
    word_counter = []
    people_counter = 0
    job_counter = 0
    with ipdb.launch_ipdb_on_exception():
        for file in os.listdir(folder):
            if file.endswith('.jl.gz'):
                print("Processing file " + str(file))
                file_path = os.path.join(folder, file)
                with gzip.open(file_path, 'r') as file:
                    num_lines = sum(1 for line in file)
                with gzip.open(file_path, 'r') as f:
                    pbar = tqdm(f, total=num_lines)
                    for line in pbar:
                        current_person = json.loads(line)
                        is_relevant, job_list = select_relevant_people(current_person, args.min_job_count)
                        if is_relevant:
                            people_counter += 1
                            for xp in job_list:
                                job_counter += 1
                                word_counter.append(len(xp["description"]) + len(xp['position']))
                        pbar.update(1)
        with open(args.target_file, "wb") as tgt_file:
            pkl.dump(word_counter, tgt_file)
        print("Average sequence length " + str(np.mean(word_counter)))
        print("Num. of retained jobs: " + str(job_counter))
        print("Num of retained PEOPLE: " + str(people_counter))


def select_relevant_people(current_person, min_job_count):
    relevant_jobs = []
    if "experience" in current_person.keys():
        if len(current_person["experience"]) >= min_job_count:
            for xp in current_person["experience"]:
                if all(k in xp.keys() for k in ('position', 'description')):
                    relevant_jobs.append(xp)
    if len(relevant_jobs) >= min_job_count:
        return True, relevant_jobs
    else:
        return False, []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/lip6/data/bypath")
    parser.add_argument("--target_file", type=str,
                        default="/local/gainondefor/work/lip6/data/seq_lengths.pkl")
    parser.add_argument("--min_job_count", type=int, default=3)
    args = parser.parse_args()
    build_dict(args)

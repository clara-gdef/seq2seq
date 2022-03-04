import os
import argparse
import gzip
import json
import pickle as pkl
from tqdm import tqdm
import unidecode


def main(args):
    with open(args.target_file, "rb") as f:
        cie_dict = pkl.load(f)


def build_dict(args):
    cie_dict = dict()
    folder = args.DATA_DIR
    comp_counter = 0
    job_counter = 0
    profile_counter = 0
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
                        profile_counter += 1
                        for xp in job_list:
                            job_counter += 1
                            if "company" in xp.keys():
                                if "name" in xp["company"].keys():
                                    comp_counter += 1
                                    normalized_name = unidecode.unidecode(xp["company"]["name"].split(" ")[0].lower())
                                    if normalized_name not in cie_dict.keys():
                                        cie_dict[normalized_name] = 1
                                    else:
                                        cie_dict[normalized_name] += 1
                    pbar.update(1)
    with open(args.target_file, "wb") as tgt_file:
        pkl.dump(cie_dict, tgt_file)
    print("Num of unique companies: " + str(len(cie_dict)))
    print("Num. of retained jobs: " + str(job_counter))
    print("Num of jobs containing a company name: " + str(comp_counter))
    print("Num of retained PEOPLE: " + str(comp_counter))


def select_relevant_people(current_person, min_job_count):
    relevant_jobs = []
    if "experience" in current_person.keys():
        if len(current_person["experience"]) >= min_job_count:
            for xp in current_person["experience"]:
                if "position" and "description" in xp.keys():
                    relevant_jobs.append(xp)
    if len(relevant_jobs) >= min_job_count:
        return True, relevant_jobs
    else:
        return False, []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/lip6/data/bypath")
    parser.add_argument("--target_file", type=str,
                        default="/local/gainondefor/work/lip6/data/bp_companies_dict_1stw.pkl")
    parser.add_argument("--min_job_count", type=int, default=1)
    args = parser.parse_args()
    build_dict(args)

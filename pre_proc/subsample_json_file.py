import os
import argparse
import gzip
import json
import ipdb

from tqdm import tqdm
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords


def subsample_json_files(args):
    folder = args.DATA_DIR
    counter_fr = 0
    with open(args.target_file_fr, 'a+') as fr_file:
        for file in os.listdir(folder):
            if file.endswith('.jl.gz'):
                print("Processing file " + str(file))
                file_path = os.path.join(folder, file)
                # we read the lines in order to have a nice print with tqdm
                with gzip.open(file_path, 'r') as file:
                    num_lines = sum(1 for line in file)
                with gzip.open(file_path, 'r') as f:
                    pbar = tqdm(f, total=num_lines)
                    for line in pbar:
                        current_person = json.loads(line)
                        # We check that the keys of interest do appear in the CV we loaded
                        if all(k in current_person.keys() for k in ('experience', 'education')):
                            if len(current_person['experience']) > args.min_job_count:
                                is_relevant, tup = from_dict_to_tuple(current_person, counter_fr, args.min_job_count)
                                if is_relevant:
                                    language = detect_language(tup)
                                    string = json.dumps(tup) + '\n'
                                    if str(language) == 'french':
                                        fr_file.write(string)
                                        counter_fr += 1
                        pbar.update(1)
                    print("Proportion of french tuples is " + str(100 * counter_fr / num_lines))


def from_dict_to_tuple(dictionary, index, min_job_count):
    is_relevant = True
    job_list = select_relevant_jobs(dictionary, min_job_count)
    if len(job_list) < 1:
        is_relevant = False
    if is_relevant:
        if "skills" in dictionary.keys():
            skills = [skill for skill in dictionary["skills"]]
        else:
            skills = []
        industry = dictionary["industry"] if "industry" in dictionary.keys() else ""
        education = [{'degree': degree["degree"],
                      'institution': degree['name'],
                      'from': degree['from'],
                      'to': degree['to']}
                     for degree in dictionary["education"] if
                     all(k in degree.keys() for k in ('degree', 'name', 'from', 'to'))]
        ordered_job_list = sorted(job_list, key=lambda tup: - tup['from_ts'])
        ordered_education = sorted(education, key=lambda tup: - int(tup['to']))
        return is_relevant, (index, ordered_job_list, skills, ordered_education, industry)
    else:
        return is_relevant, ()


def select_relevant_jobs(dictionary, min_job_count):
    relevant_jobs = []
    job_list = []
    for exp in dictionary['experience']:
        if "description" in exp.keys():
            if len(exp["description"]) > 0:
                relevant_jobs.append(exp)
    if len(relevant_jobs) > min_job_count:
        for job in relevant_jobs:
            if all(k in job.keys() for k in ('position', 'from_ts', 'from', 'to', 'description')):
                tmp = {'position': job['position'],
                       'from_ts': job["from_ts"],
                       'from': job["from"],
                       'to': job['to'],
                       'description': job["description"]}
                if "company" in job.keys():
                    if "name" in job["company"].keys():
                        tmp["company"] = job["company"]["name"]
                job_list.append(tmp)
    return job_list


# Code by Alejandro Nolla ----------------------------------------------------------------------
def _calculate_languages_ratios(tup):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}

    @param text: Text whose language want to be detected
    @type text: str

    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """

    languages_ratios = {"french": 0,
                        "english": 0}

    '''
    nltk.wordpunct_tokenize() splits all punctuations into separate tokens

    >>> wordpunct_tokenize("That's thirty minutes away. I'll be there in ten.")
    ['That', "'", 's', 'thirty', 'minutes', 'away', '.', 'I', "'", 'll', 'be', 'there', 'in', 'ten', '.']
    '''
    for job in tup[1]:
        tokens = wordpunct_tokenize(job["position"])
        tokens.extend(wordpunct_tokenize(job["description"]))
        words = [word.lower() for word in tokens]

        # Compute per language included in nltk number of unique stopwords appearing in analyzed text
        for language in ["french", "english"]:
            stopwords_set = set(stopwords.words(language))
            words_set = set(words)
            common_elements = words_set.intersection(stopwords_set)

            languages_ratios[language] += len(common_elements)  # language "score"

    return languages_ratios


# Code by Alejandro Nolla ----------------------------------------------------------------------
def detect_language(text):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.

    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.

    @param text: Text whose language want to be detected
    @type text: str

    @return: Most scored language guessed
    @rtype: str
    """

    ratios = _calculate_languages_ratios(text)

    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default=os.path.realpath('/local/gainondefor/work/lip6/data/bypath'))
    parser.add_argument("--target_file_fr", type=str,
                        default="/local/gainondefor/work/lip6/data/bp_3jobs_desc_edu_skills_industry_date_company_FR.json")
    parser.add_argument("--min_job_count", type=int, default=3)
    args = parser.parse_args()
    subsample_json_files(args)

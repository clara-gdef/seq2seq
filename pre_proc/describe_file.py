import argparse
import pickle as pkl
from scipy import stats
import ipdb


def main(args):
    with open(args.src_file, "rb") as f:
        data = pkl.load(f)
    sorted_tuples = sorted([(k, v) for k, v in data.items()], key=lambda obj: obj[1], reverse=True)
    sorted_apparitions = sorted([v for _, v in data.items()], reverse=True)
    with open(args.tgt_file, 'w') as tgt_f:
        tgt_f.write("====================== OVERVIEW ======================\n")
        tgt_f.write(str(stats.describe(sorted_apparitions)))
        tgt_f.write("\n")
        tgt_f.write("median: " + str(sorted_apparitions[round(len(sorted_apparitions)/2)]))
        tgt_f.write("\n")
        tgt_f.write("====================== MORE THAN 5 ======================\n")
        more_than_5 = [e for e in sorted_apparitions if e > 4]
        tgt_f.write(str(stats.describe(more_than_5)))
        tgt_f.write("\n")
        tgt_f.write("median: " + str(more_than_5[round(len(more_than_5)/2)]))
        tgt_f.write("\n")
        tgt_f.write("====================== MORE THAN 10 ======================\n")
        more_than_10 = [e for e in sorted_apparitions if e > 9]
        tgt_f.write(str(stats.describe(more_than_10)))
        tgt_f.write("\n")
        tgt_f.write("median: " + str(more_than_5[round(len(more_than_10)/2)]))
        tgt_f.write("\n")
        tgt_f.write("Showing " + str(args.tup_to_show) + " first tuples: ")
        tgt_f.write("\n")
        tgt_f.write(str(sorted_tuples[:args.tup_to_show]))
        tgt_f.write("\n")
        with open(args.blacklist_file, "rb") as f:
            blacklist = pkl.load(f)
    
        tgt_f.write("====================== FILTERED TUPLES ======================\n")
        filtered_tuples = [e for e in sorted_tuples if e[0] not in blacklist]
        filtered_apparitions = sorted([e[1] for e in filtered_tuples], reverse=True)
        tgt_f.write(str(stats.describe(filtered_apparitions)))
        tgt_f.write("\n")
        tgt_f.write("median: " + str(filtered_apparitions[round(len(filtered_apparitions)/2)]))
        tgt_f.write("\n")
        tgt_f.write("Showing " + str(args.tup_to_show) + " first tuples: ")
        tgt_f.write("\n")
        tgt_f.write(str(filtered_tuples[:args.tup_to_show]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, default='/local/gainondefor/work/lip6/data/bp_companies_dict_1stw.pkl')
    parser.add_argument("--tup_to_show", type=int, default=200)
    parser.add_argument("--blacklist_file", type=str, default="/local/gainondefor/work/lip6/data/company_word_blacklist.pkl")
    parser.add_argument("--tgt_file", type=str, default="/local/gainondefor/work/lip6/data/company_report.txt")
    args = parser.parse_args()
    main(args)

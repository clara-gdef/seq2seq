import argparse
from os import system, path
# import ipdb


def main(args):
    ref = path.join(args.DATA_DIR, args.ref)
    pred = path.join(args.DATA_DIR, args.pred)
    cmd_line = './multi-bleu.perl ' + ref + ' < ' + pred + ''
    #ipdb.set_trace()
    system(cmd_line)


def compute_bleu_score(pred, ref):
    cmd_line = './multi-bleu.perl ' + ref + ' < ' + pred + ''
    system(cmd_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--pred", type=str, default="mc_lj_words.txt")
    parser.add_argument("--ref", type=str, default="labels_lj.txt")
    args = parser.parse_args()
    main(args)


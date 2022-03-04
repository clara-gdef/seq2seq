import argparse
import os
import pickle as pkl
from datetime import datetime

import ipdb
import quiviz
import torch
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import DecoderWithFT
from sequence_generator import generate_sequence
from utils.bleu_score import compute_bleu_score


def main(args):
    suffix = str(args.model_type)[:2]
    xp_title = "FT " + args.model_type + " dec bs" + str(args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    print("Loading data...")
    data_test = []
    flag_err = False
    with open(os.path.join(args.DATA_DIR, "pkl/prof_rep2_lj_ft_" + suffix + "_test.pkl"), 'rb') as f:
        while not flag_err:
            try:
                data = pkl.load(f)
                data_test.append(data)
            except EOFError:
                flag_err = True
                continue
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)
    print("Data loaded.")

    hidden_size = args.dec_hidden_size
    num_layers = 1

    decoder = DecoderWithFT(int(args.emb_size), hidden_size, num_layers, len(index))

    dec_weights = os.path.join(args.model_dir, args.dec_model)
    decoder.load_state_dict(torch.load(dec_weights))
    dec = decoder.cuda()

    dictionary = main_for_one_split(args, dec, data_test, index)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/esann20/" + args.model_type + "_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, decoder, dataset_test, vocab_index):
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate,
                                 shuffle=True, num_workers=0, drop_last=True)

    pred_file, lab_file = evaluate(decoder, dataloader_test, vocab_index)
    compute_bleu_score(pred_file, lab_file)


def evaluate(decoder, dataloader_test, vocab_index):
    rev_index = {v: k for k, v in vocab_index.items()}
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_test, desc="Evaluating...") as pbar:
            for ids, profile, lj_indices in pbar:
                b_size = 1

                # initialize the hidden state of the decoder
                h_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                c_0 = torch.zeros(decoder.num_layer, b_size, decoder.hidden_size).cuda()
                decoder_hidden = (h_0, c_0)

                profile_tensor = torch.stack(profile).unsqueeze(1).cuda()

                token = vocab_index["SOT"]
                decoded_words = []
                if args.beam_search:
                    for i in range(len(lj_indices[0])):
                        tokens, nlls = generate_sequence(decoder, decoder_hidden, "ft", len(vocab_index), vocab_index["SOT"], vocab_index["EOD"],
                                                         search="greedy", size=1, sampling_topk=1, max_len=64,
                                                         hidden=profile_tensor, unk=5, unk_penalty=None)
                else:
                    for i in range(len(lj_indices[0])):
                        tok_tensor = torch.LongTensor(1, 1)
                        tok_tensor[:, 0] = token
                        output, decoder_hidden = decoder(profile_tensor, decoder_hidden, tok_tensor)
                        dec_word = output.argmax(-1).item()
                        decoded_words.append(dec_word)
                        token = dec_word

                pred_file = os.path.join(args.DATA_DIR, "pred_ft_" + args.model_type + "2.txt")
                with open(pred_file, 'a') as f:
                    for w in decoded_words:
                        f.write(rev_index[w] + ' ')
                    f.write("\n")

                lab_file = os.path.join(args.DATA_DIR, "label_ft_" + args.model_type + "2.txt")
                with open(lab_file, 'a') as f:
                    for w in lj_indices[0][1:]:
                        f.write(rev_index[w] + ' ')
                    f.write("\n")

        return pred_file, lab_file


def collate(batch):
    tmp = list(zip(*batch))
    ids, profiles, lj_ind = tmp[0], tmp[1], tmp[2]
    identifiers, prof, lj_indices = zip(*sorted(zip(ids, profiles, lj_ind), key=lambda item: item[2], reverse=True))
    return list(identifiers), list(prof), list(lj_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dec_hidden_size", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="pt")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--dec_model", type=str, default="pt_bs180_lr0.001_tf1_hs_256_max_ep_300_40k_dec_best_ep_214")
    parser.add_argument("--tf", type=float, default=1)
    parser.add_argument("--dpo", type=float, default=.5)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--beam_search", type=bool, default=False)
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20/job_dec')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

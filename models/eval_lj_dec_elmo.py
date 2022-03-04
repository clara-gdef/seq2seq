import argparse
import os
import pickle as pkl

import ipdb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import DecoderWithFT
from utils.bleu_score import compute_bleu_score


def main(args):
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Loading data...")
    with open(os.path.join(args.DATA_DIR, "test_lj.pkl"), "rb") as f:
        data_test = pkl.load(f)
    print("Data loaded.")

    hidden_size = args.dec_hidden_size
    num_layers = 1
    elmo_dimension = 1024

    with ipdb.launch_ipdb_on_exception():
        decoder = DecoderWithFT(elmo_dimension, hidden_size, num_layers, len(index))

        dec_weights = os.path.join(args.model_dir, args.dec_model)
        decoder.load_state_dict(torch.load(dec_weights))
        dec = decoder.cuda()

        main_for_one_split(dec, data_test, index)


def main_for_one_split(decoder, dataset_test, vocab_index):
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate,
                                 shuffle=True, num_workers=0, drop_last=True)

    pred_file, lab_file = evaluate(decoder, dataloader_test, vocab_index)
    compute_bleu_score(pred_file, lab_file)


def evaluate(dec, dataloader_test, index):
    rev_index = {v: k for k, v in index.items()}
    with ipdb.launch_ipdb_on_exception():
        for ids, tensors, indices, words in tqdm(dataloader_test, desc="Evaluating..."):
            b_size = len(ids)
            decoded_words = []

            # initialize the hidden state of the decoder
            h_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
            c_0 = torch.zeros(dec.num_layer, b_size, dec.hidden_size).cuda()
            decoder_hidden = (h_0, c_0)

            predictions = torch.cat(tensors).cuda()

            token = [[index["SOT"]]]

            for i in range(len(words[0][:-1])):
                output, decoder_hidden = dec(predictions, decoder_hidden, torch.LongTensor(token))
                dec_word = output.argmax(-1)
                token = [[dec_word]]
                decoded_words.append(dec_word)

            pred_file = os.path.join(args.DATA_DIR, "pred_lstm_elmo.txt")
            with open(pred_file, 'a') as f:
                for w in decoded_words:
                    f.write(rev_index[w.item()] + ' ')
                f.write("\n")

            lab_file = os.path.join(args.DATA_DIR, "label_lstm_elmo.txt")
            with open(lab_file, 'a') as f:
                # for example in range(b_size):
                    for w in words[0][1:]:
                        f.write(w + ' ')
                    f.write("\n")

        return pred_file, lab_file


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    indices = [e[2] for e in batch]
    words = [e[3] for e in batch]
    return ids, tensors, indices, words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/prof_ind_elmo_test_cpu.pkl")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--dec_hidden_size", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="elmo_lstm")
    parser.add_argument("--from_trained_model", type=bool, default=True)
    parser.add_argument("--dec_model", type=str, default="elmo_bs80_lr0.001_tf1_hs_256_max_ep_300_40k_dec_best_ep_14")
    parser.add_argument("--enc_model", type=str,
                        default="elmo_w2v_bs64_lr0.0001_tf0_hs_512_max_ep_300_encCareer_best_ep_185")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/esann20/job_dec')
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

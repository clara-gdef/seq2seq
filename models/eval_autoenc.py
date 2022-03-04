import argparse
import os
import pickle as pkl
from datetime import datetime

import ipdb
import quiviz
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes import DecoderLSTM, EncoderBiLSTM
from pre_proc.from_indices_to_words import decode_indices
from utils.JobDataset import JobDataset
from utils.Utils import collate_for_jobs, compute_crossentropy


def main(args):
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)


    print("Loading data...")
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    print("Data loaded.")

    dataset = JobDataset(datadict["test"])
    # dimension = embeddings.shape[1]

    dimension = 100

    hidden_size = 16
    num_layers = 1

    decoder = DecoderLSTM(embeddings, hidden_size, num_layers, dimension, embeddings.size(0), args.MAX_CAREER_LENGTH)
    encoder = EncoderBiLSTM(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)

    # # for efficient memory save
    del embeddings

    enc_weights = os.path.join(args.model_dir, args.enc_model)
    dec_weights = os.path.join(args.model_dir, args.dec_model)
    encoder.load_state_dict(torch.load(enc_weights))
    decoder.load_state_dict(torch.load(dec_weights))

    enc = encoder.cuda()
    dec = decoder.cuda()

    dictionary = main_for_one_split(args, enc, dec, dataset, index)
    day = str(datetime.now().day)
    month = str(datetime.now().month)
    tgt_file = os.path.join(args.DATA_DIR,
                            "results/splitless/" + args.model_type + "_eval_results_" + day + "_" + month)
    with open(tgt_file, "wb") as file:
        pkl.dump(dictionary, file)


def main_for_one_split(args, encoder, decoder, dataset, vocab_index):
    dataloader_test = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_for_jobs,
                                  shuffle=True, num_workers=3, drop_last=True)
    dico = main_for_one_epoch(args, encoder, decoder, vocab_index, dataloader_test)
    print(dico)
    return dico


@quiviz.log
def main_for_one_epoch(args, encoder, decoder, vocab_index, dataloader_test):

    test_loss = evaluate_perp(args, encoder, decoder, dataloader_test, vocab_index)

    return test_loss


def evaluate_perp(args, encoder, decoder, dataloader_test, vocab_index):
    rev_index = {v: k for k, v in vocab_index.items()}
    nb_tokens = 0
    enforce_sorted = False
    cross_entropy = []
    with ipdb.launch_ipdb_on_exception():
        with tqdm(dataloader_test, desc="Evaluating perplexity...") as pbar:
            for job, seq_length in pbar:
                b_size = 1
                if len(cross_entropy) % 1000 == 1:
                    print(torch.exp(torch.sum(torch.FloatTensor(cross_entropy) / float(nb_tokens))).item())

                job_tensor = torch.LongTensor(job).cuda()

                # train the encoder
                enc_output, attention, encoder_hidden = encoder(job_tensor, seq_length, enforce_sorted)

                # train the decoder
                decoder_hidden = (encoder_hidden[0].view(1, b_size, -1), encoder_hidden[1].view(1, b_size, -1))
                nb_tokens += seq_length[0]

                enc_o = enc_output.unsqueeze(1).expand(1, seq_length[0], -1)
                decoder_output, decoder_hidden = decoder(enc_o, decoder_hidden, job_tensor)

                cross_entropy.append(compute_crossentropy(decoder_output.transpose(2, 1), job_tensor).item())


                pred_file_overall = os.path.join(args.DATA_DIR, "results/eval_output_ae_splitless.txt")
                pred_text_overall = decode_indices(decoder_output.argmax(-1).squeeze(0), rev_index)
                pred_text_overall += "\n"
                with open(pred_file_overall, "a") as pf:
                    pf.write(pred_text_overall)
                label_file_overall = os.path.join(args.DATA_DIR, "results/eval_label_ae_splitless.txt")
                label_text_overall = decode_indices(job_tensor[0], rev_index)
                label_text_overall += "\n"
                with open(label_file_overall, "a") as lf:
                    lf.write(label_text_overall)

                pbar.update(1)
            # perplexity_title

    return {"perplexity": torch.exp(torch.sum(torch.FloatTensor(cross_entropy))/nb_tokens).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/indices_jobs.pkl")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--record_outputs", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="s2s")
    parser.add_argument("--emb_file", type=str, default="pkl/tensor_40k.pkl")
    parser.add_argument("--enc_model", type=str, default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_enc_best_ep_92")
    parser.add_argument("--dec_model", type=str, default="s2s_hard_decay_bs200_lr0.001_tf1_hs_16_max_ep_300_40k_dec_best_ep_92")
    parser.add_argument("--voc_size", type=str, default="40k")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/splitless')
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    args = parser.parse_args()
    main(args)

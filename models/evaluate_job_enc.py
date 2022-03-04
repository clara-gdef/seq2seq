import argparse
import quiviz
import os
import pickle as pkl
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader

from classes import DecoderGru, EncoderBiGru_old, EncoderBiGru
from utils.Utils import load_data, build_profile_embeddings_old, transform, collate, \
    indices_list_to_one_hot_tensor, pad_labels, pad_pred, compute_perplexity, build_profile_embeddings
from utils.LinkedInDataset import LinkedInDataset
import torch
from tqdm import tqdm

import ipdb


def eval(args):
    xp_title = "s2s separated"
    quiviz.name_xp(xp_title + " s#" + str(args.split))
    quiviz.register(LinePlotObs())
    with open(os.path.join(args.DATA_DIR, args.emb_file), "rb") as f:
        embeddings = pkl.load(f)
    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    # TODO: remove when moving onto new model
    if args.old_models:
        SOS = torch.zeros(1, 300) - 31
        EOS = torch.zeros(1, 300) + 31
        torch.cat((embeddings, SOS, EOS), dim=0)

    dimension = embeddings.shape[1]

    hidden_size = 64
    num_layers = 1

    data_tl, (trainit, valit, testit) = load_data(args)

    decoder = DecoderGru(embeddings, hidden_size, num_layers, dimension, embeddings.size(0), args.MAX_CAREER_LENGTH)
    if args.old_models:
        encoder = EncoderBiGru_old(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)
    else:
        encoder = EncoderBiGru(embeddings, dimension, hidden_size, num_layers, args.MAX_CAREER_LENGTH, args.batch_size)

    # # for efficient memory save
    # del embeddings

    dataset = LinkedInDataset(testit, data_tl, transform)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate,
            shuffle=True, num_workers=3, drop_last=False)

    if args.from_trained_model:
        enc_weights = os.path.join(args.model_dir + "/split" + str(args.split), args.enc_model)
        dec_weights = os.path.join(args.model_dir + "/split" + str(args.split), args.dec_model)
        encoder.load_state_dict(torch.load(enc_weights))
        decoder.load_state_dict(torch.load(dec_weights))

    enc = encoder.cuda()
    dec = decoder.cuda()

    dictionary = test(args, enc, dec, dataloader, index)
    print(dictionary)
    with open(os.path.join(args.DATA_DIR, "perp_old_model_s" + str(args.split) + ".pkl"), "wb") as f:
        pkl.dump({"perplexity for split "+str(args.split): dictionary}, f)


def test(args, enc, dec, dataloader_test, voc_index):
    whole_perp = []
    with tqdm(total=len(dataloader_test), desc="Evaluation...") as pbar:  
        for iteration, batch in tqdm(enumerate(dataloader_test)):
            profile = batch["profile"]
            seq_length = batch["seq_length"]
            b_size = len(batch["profile"])
            # last_jobs = batch["last_job"]

            if args.old_models:
                profile_id = build_profile_embeddings_old(b_size, args.MAX_SEQ_LENGTH, args.MAX_CAREER_LENGTH, profile,
                                                  seq_length, voc_index)
            else:
                profile_id = build_profile_embeddings(b_size, args.MAX_SEQ_LENGTH, args.MAX_CAREER_LENGTH, profile,
                                                  seq_length, voc_index)
            profile_index = profile_id.type(torch.LongTensor).cuda()
            tmp = torch.LongTensor(seq_length)

            # get the encoder output
            if args.old_models:
                perp_tensor = test_old_models(profile_index, b_size, enc, dec, voc_index, tmp)
                whole_perp.append(perp_tensor)
            else:
                perp_tensor = test_newer_models(profile_index, b_size, enc, dec, voc_index, tmp)
                whole_perp.append(perp_tensor)
        pbar.update(1)
        whole_perp_tensor = torch.mean(torch.FloatTensor(whole_perp))
    return {"eval_perp": whole_perp_tensor.item()}


def test_old_models(profile_index, b_size, enc, dec, voc_index, tmp):
    encoder_hidden = torch.autograd.Variable(enc.initHidden()).cuda()
    softmax = torch.nn.Softmax(dim=0)
    perplexity = []
    for i in range(profile_index.shape[2]):
        # we put 0 because the bacth size will always be 1
        target = profile_index[:, :, i]
        seq_len = tmp[:, i]
        if seq_len.tolist() != [0]:
            enc_output, attention, enc_hidden_out = enc(target, seq_len, encoder_hidden)
            encoder_hidden = enc_hidden_out

        tmp = torch.zeros(b_size) + voc_index["SOS"]
        tokens = torch.autograd.Variable(tmp.type(torch.LongTensor)).cuda()
        token = tokens.unsqueeze(1)
        decoder_hidden = encoder_hidden
        weighted_rep = enc_output

        counter = 0
        out_tokens = []
        end_of_seq_token = voc_index["EOS"]

        while token.item() != end_of_seq_token and counter < args.MAX_SEQ_LENGTH:
            # targets = torch.autograd.Variable(profile_index[:, :, i]).cuda()
            targets = None
            decoder_output, decoder_hidden = dec(weighted_rep, decoder_hidden, targets, token)
            output = softmax(decoder_output).argmax(-1)
            token = output.type(torch.LongTensor).cuda()
            counter += 1
            out_tokens.append(token.item())

        ipdb.set_trace()
        pred = indices_list_to_one_hot_tensor(out_tokens, voc_index)
        if len(target) < len(pred):
            target = pad_labels(target, len(pred))
        if len(target) > len(pred):
            pred = pad_pred(pred, len(target))
        perplexity.append(torch.nn.functional.cross_entropy(pred, target))
    perp_tensor = torch.mean(torch.FloatTensor(perplexity))
    return perp_tensor


def test_newer_models(profile_index, b_size, enc, dec, voc_index, lengths):
    encoder_hidden = torch.autograd.Variable(enc.initHidden()).cuda()
    softmax = torch.nn.Softmax(dim=-1)
    perplexity = []
    for i in range(profile_index.shape[1]):
        target = profile_index[:, i, :]
        seq_len = lengths[:, i]
        if seq_len.tolist() != [0] * b_size:
            enc_output, attention, enc_hidden_out = enc(target, seq_len, encoder_hidden)
            encoder_hidden = enc_hidden_out

            tmp = torch.zeros(b_size) + voc_index["SOT"]
            tokens = torch.autograd.Variable(tmp.type(torch.LongTensor)).cuda()
            token = tokens.unsqueeze(1)
            decoder_hidden = encoder_hidden
            weighted_rep = enc_output

            counter = 0
            out_tokens = []
            end_of_seq_token = voc_index["EOD"]

            while token.item() != end_of_seq_token and counter < args.MAX_SEQ_LENGTH:
                # targets = torch.autograd.Variable(profile_index[:, :, i]).cuda()
                targets = None
                decoder_output, decoder_hidden = dec(weighted_rep, decoder_hidden, targets, token)
                output = softmax(decoder_output).argmax(-1)
                token = output.type(torch.LongTensor).cuda()
                counter += 1
                out_tokens.append(token.item())

            ipdb.set_trace()
            pred = indices_list_to_one_hot_tensor(out_tokens, voc_index)
            if len(target) < len(pred):
                labels = pad_labels(target, len(pred))
            if len(target) > len(pred):
                pred = pad_pred(pred, len(target))

            perplexity.append(compute_perplexity(pred, labels))

        perp_tensor = torch.mean(torch.FloatTensor(perplexity))
    return perp_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--old_models", type=bool, default=False)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--emb_file", type=str, default="pkl/100_tensor_vocab_3j.pkl")
    parser.add_argument("--index_file", type=str, default="pkl/100_index_vocab_3j.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="s2s")
    parser.add_argument("--from_trained_model", type=bool, default=True)
    parser.add_argument("--enc_model", type=str, default=None)
    parser.add_argument("--dec_model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--MAX_CAREER_LENGTH", type=int, default=8)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=.5)
    args = parser.parse_args()
    eval(args)

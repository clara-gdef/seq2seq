import argparse
import itertools
import os
import pickle as pkl
import torch

from allennlp.modules.elmo import Elmo
from quiviz import quiviz
from quiviz.contrib import LinePlotObs
from torch.utils.data import DataLoader
from torch.autograd import gradcheck
from tqdm import tqdm

from classes import CareerEncoderLSTM, CareerDecoderLSTM, DecoderWithElmo
from utils import ProfileDatasetElmoIndices
from utils.Utils import collate_profiles_ind_elmo, save_best_model_elmo_w2v, \
    model_checkpoint_elmo_w2v, labels_to_indices
import ipdb


def init(args):
    # loading data
    print("Loading data...")
    train_file = os.path.join(args.DATA_DIR, args.train_file)
    with open(train_file, "rb") as f:
        datadict_train = pkl.load(f)
    print("Train file loaded.")

    valid_file = os.path.join(args.DATA_DIR, args.valid_file)
    with open(valid_file, "rb") as f:
        datadict_valid = pkl.load(f)
    print("Valid file loaded.")

    with open(os.path.join(args.DATA_DIR, args.index_file), "rb") as f:
        index = pkl.load(f)

    print("Data loaded.")

    dataset_train = ProfileDatasetElmoIndices(datadict_train["data"])
    dataset_valid = ProfileDatasetElmoIndices(datadict_valid["data"])
    del datadict_train, datadict_valid

    hidden_size = args.hidden_size
    elmo_size = 1024
    num_layers = 1

    encoder_career = CareerEncoderLSTM(elmo_size, hidden_size, 1, args.dpo, args.bidirectional).cuda()
    decoder_career = CareerDecoderLSTM(elmo_size, hidden_size * 2, 1, args.dpo).cuda()
    optim = torch.optim.SGD(itertools.chain(encoder_career.parameters(), decoder_career.parameters()), lr=args.lr)

    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("Initializing ELMo...")
    elmo = Elmo(options_file, weight_file, 2, requires_grad=False, dropout=0)
    print("ELMo ready.")

    job_dec_hs = str.split(args.job_dec_model, sep="_")[6]
    job_decoder = DecoderWithElmo(elmo, elmo_size, int(job_dec_hs), num_layers, len(index)).cuda()
    job_dec_weights = os.path.join("/net/big/gainondefor/work/trained_models/seq2seq/elmo/", args.job_dec_model)
    job_decoder.load_state_dict(torch.load(job_dec_weights))

    prev_epochs = 0

    if args.from_trained_model:
        prev_epochs = int(str.split(args.dec_model, sep='_')[-1])
        enc_weights = os.path.join(args.model_dir, args.enc_model)
        encoder_career.load_state_dict(torch.load(enc_weights))
        dec_weights = os.path.join(args.model_dir, args.dec_model)
        decoder_career.load_state_dict(torch.load(dec_weights))
        optim_weights = os.path.join(args.model_dir, args.optim)
        optim.load_state_dict(torch.load(optim_weights))

    return dataset_train, dataset_valid, encoder_career, decoder_career, job_decoder, optim, prev_epochs


def main(args):
    xp_title = "ELMo TEST CE lr" + str(args.lr) + " hs" + str(args.hidden_size) + " bs" + str(
        args.batch_size)
    quiviz.name_xp(xp_title)
    quiviz.register(LinePlotObs())

    dataset_train, dataset_valid, encoder_career, decoder_career, job_decoder, optim, prev_epochs = init(args)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_ind_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                                  collate_fn=collate_profiles_ind_elmo,
                                  shuffle=True, num_workers=0, drop_last=True)
    res_epoch = {}
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    best_val_loss = 1e+300

    for epoch in range(1, args.epoch + 1):
        dico = main_for_one_epoch(epoch, encoder_career, decoder_career, job_decoder, criterion,
                                  args, best_val_loss, optim, dataloader_train, dataloader_valid, prev_epochs)
        res_epoch[epoch] = {'train_loss': dico['train_loss'],
                            'valid_loss': dico['valid_loss']}
        best_val_loss = dico['best_val_loss']


@quiviz.log
def main_for_one_epoch(epoch, encoder_career, decoder_career, job_decoder, criterion,
                       args, best_val_loss, optim, dataloader_train, dataloader_valid, prev_epochs):
    epoch += prev_epochs
    print("Training and validating for epoch " + str(epoch))
    train_loss = train(args, encoder_career, decoder_career, job_decoder, dataloader_train, criterion, optim, epoch)
    valid_loss = valid(encoder_career, decoder_career, job_decoder, dataloader_valid, criterion)

    target_dir = args.model_dir
    if valid_loss['valid_loss'] < best_val_loss:
        if not args.DEBUG:
            save_best_model_elmo_w2v(args, epoch, target_dir, encoder_career, decoder_career, optim)
        best_val_loss = valid_loss['valid_loss']
    if args.save_last:
        if not args.DEBUG:
            model_checkpoint_elmo_w2v(args, epoch, target_dir, encoder_career, decoder_career, optim)
        dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(args, encoder_career, decoder_career, job_decoder, dataloader_train, criterion, optim, epoch):
    elmo_dimension = 1024
    loss_list = []
    nb_tokens = 0
    with ipdb.launch_ipdb_on_exception():
        for ids, jobs, career_len, jobs_indices, jobs_words, jobs_lengths in tqdm(dataloader_train):
            loss = 0
            optim.zero_grad()

            b_size = len(ids)

            max_seq_length = max(career_len)
            profile_tensor = torch.zeros(b_size, max_seq_length, elmo_dimension)

            ## We encode each career
            for person in range(b_size):
                profile_tensor[person, :len(jobs[person]), :] = torch.cat(jobs[person])
            prof_tensor = profile_tensor.cuda()
            z_people, hidden_state = encoder_career(prof_tensor, list(career_len), enforce_sorted=False)

            ## we decode each career

            prev_job = torch.zeros(1, 1, elmo_dimension).cuda()

            jobs = torch.zeros(b_size, max_seq_length, elmo_dimension).cuda()

            for i in range(b_size):
                h_0 = torch.zeros(decoder_career.num_layers, 1, decoder_career.hidden_size).cuda()
                c_0 = torch.zeros(decoder_career.num_layers, 1, decoder_career.hidden_size).cuda()
                decoder_hidden = (h_0, c_0)
                for j in range(max_seq_length):
                    next_job, decoder_hidden = decoder_career(z_people[i, :, :].unsqueeze(0), decoder_hidden, prev_job)
                    jobs[i, j, :] = next_job

            job_decoder_hidden = (
                torch.zeros(1, 1, job_decoder.hidden_size).cuda(), torch.zeros(1, 1, job_decoder.hidden_size).cuda())

            for i in range(b_size):
                for j in range(len(jobs_words[i])):
                    tmp = jobs[i, j, :].unsqueeze(0).unsqueeze(0)
                    expanded_job = tmp.expand(1, len(jobs_words[i][j]), jobs.shape[-1])
                    tokens = jobs_words[i][j]
                    decoder_output, decoder_hidden = job_decoder(expanded_job, job_decoder_hidden, [tokens])
                    loss += criterion(decoder_output.transpose(2, 1), torch.LongTensor(jobs_indices[i][j]).unsqueeze(0).cuda())

            nb_tokens += sum([e for i in jobs_lengths for e in i])

            loss_list.append(loss.item())
            loss.backward()

            optim.step()

    return {"train_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "train_perplexity": 2 ** (sum(loss_list) / nb_tokens)}


def valid(encoder_career, decoder_career, job_decoder, dataloader_valid, criterion):
    elmo_dimension = 1024
    loss_list = []
    nb_tokens = 0
    with ipdb.launch_ipdb_on_exception():
        for ids, jobs, career_len, jobs_indices, jobs_words, jobs_lengths in tqdm(dataloader_valid):
            loss = 0
            b_size = len(ids)

            max_seq_length = max(career_len)
            profile_tensor = torch.zeros(b_size, max_seq_length, elmo_dimension)

            ## We encode each career
            for person in range(b_size):
                profile_tensor[person, :len(jobs[person]), :] = torch.cat(jobs[person])
            prof_tensor = profile_tensor.cuda()
            z_people, hidden_state = encoder_career(prof_tensor, list(career_len), enforce_sorted=False)

            ## we decode each career

            prev_job = torch.zeros(1, 1, elmo_dimension).cuda()

            jobs = torch.zeros(b_size, max_seq_length, elmo_dimension).cuda()

            for i in range(b_size):
                h_0 = torch.zeros(decoder_career.num_layers, 1, decoder_career.hidden_size).cuda()
                c_0 = torch.zeros(decoder_career.num_layers, 1, decoder_career.hidden_size).cuda()
                decoder_hidden = (h_0, c_0)
                for j in range(max_seq_length):
                    next_job, decoder_hidden = decoder_career(z_people[i, :, :].unsqueeze(0), decoder_hidden, prev_job)
                    jobs[i, j, :] = next_job

            job_decoder_hidden = (
                torch.zeros(1, 1, job_decoder.hidden_size).cuda(), torch.zeros(1, 1, job_decoder.hidden_size).cuda())

            for i in range(b_size):
                for j in range(len(jobs_words[i])):
                    tmp = jobs[i, j, :].unsqueeze(0).unsqueeze(0)
                    expanded_job = tmp.expand(1, len(jobs_words[i][j]), jobs.shape[-1])
                    tokens = jobs_words[i][j]
                    decoder_output, decoder_hidden = job_decoder(expanded_job, job_decoder_hidden, [tokens])
                    loss += criterion(decoder_output.transpose(2, 1), torch.LongTensor(jobs_indices[i][j]).unsqueeze(0).cuda())

            nb_tokens += sum([e for i in jobs_lengths for e in i])

            loss_list.append(loss.item())

    return {"valid_loss": torch.mean(torch.FloatTensor(loss_list)).item(),
            "valid_perplexity": 2 ** (sum(loss_list) / nb_tokens)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="pkl/prof_ind_elmo_train_cpu.pkl")
    parser.add_argument("--valid_file", type=str, default="pkl/prof_ind_elmo_valid_cpu.pkl")
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--model_type", type=str, default="elmo_ce_test_")
    parser.add_argument("--from_trained_model", type=bool, default=False)
    parser.add_argument("--dec_model", type=str, default=None)
    parser.add_argument("--enc_model", type=str, default=None)
    parser.add_argument("--job_dec_model", type=str,
                        default="s2s_elmo_bs128_lr0.001_tf1_hs_256_max_ep_300_40k_dec_best_ep_19")
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--save_last", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--tf", type=float, default=0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/elmo_w2v')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--dpo", type=float, default=.0)
    parser.add_argument("--bidirectional", type=bool, default=True)
    args = parser.parse_args()
    main(args)


import torch
import os
import pickle as pkl
from tqdm import tqdm
import glob
import ipdb
from utils.fmtl import FMTL


MAX_SEQ_LENGTH = 64
MAX_CAREER_LENGTH = 8


def load_data(args):
    """ This method should get all the jobs of a user """
    data_file = os.path.join(args.DATA_DIR, args.input_file)
    with open(data_file, 'rb') as file:
        datadict = pkl.load(file)
    data_tl, (trainit, valit, testit) = FMTL_train_val_test(datadict["data"],
                                                            datadict["splits"],
                                                            args.split,
                                                            validation=0.5,
                                                            rows=datadict["rows"])
    data_tl = FMTL(list(x for x in tqdm(data_tl, desc="prebuilding")), rows=data_tl.rows)

    return data_tl, (trainit, valit, testit)


def FMTL_train_val_test(datatuples, splits, split_num=0, validation=0.5, rows=None):
    """
    Builds train/val/test indexes sets from tuple list and split list
    Validation set at 0.5 if n split is 5 gives an 80:10:10 split as usually used.
    """
    train, test = [], []

    for idx, split in tqdm(enumerate(splits), total=len(splits), desc="Building train/test of split #{}".format(split_num)):
        if split == split_num:
            test.append(idx)
        else:
            train.append(idx)

    if len(test) <= 0:
            raise IndexError("Test set is empty - split {} probably doesn't exist".format(split_num))

    if rows and type(rows) is tuple:
        rows = {v: k for k, v in enumerate(rows)}
        print("Tuples rows are the following:")
        print(rows)

    if validation > 0:

        if 0 < validation < 1:
            val_len = int(validation * len(test))

        validation = test[-val_len:]
        test = test[:-val_len]

    else:
        validation = []

    idxs = (train, test, validation)
    fmtl = FMTL(datatuples, rows)
    iters = idxs

    return (fmtl, iters)


def collate_for_jobs(batch):
    tmp = list(zip(*batch))
    sequences, lengths = tmp[0], tmp[1]
    jobs, seq_lengths = zip(*sorted(zip(sequences, lengths), key=lambda item: item[1], reverse=True))
    return list(jobs), list(seq_lengths)


def collate_for_jobs_elmo(batch):
    tmp = list(zip(*batch))
    sequences, lengths, indices = tmp[0], tmp[1], tmp[2]
    jobs, seq_lengths, ind = zip(*sorted(zip(sequences, lengths, indices), key=lambda item: item[1], reverse=True))
    return list(jobs), list(seq_lengths), list(ind)


def collate_for_profiles_elmo(batch):
    tmp = list(zip(*batch))
    identifier, jobs, lengths = tmp[0], tmp[1], tmp[2]
    return identifier, jobs, lengths


def collate_profiles_lj_elmo(batch):
    tmp = list(zip(*batch))
    ids, jobs_indices, jobs_lengths, lj_indices, lj_lengths = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]
    return ids, jobs_indices, jobs_lengths, lj_indices, lj_lengths


def collate_profiles_ind_elmo(batch):
    tmp = list(zip(*batch))
    ids, embs, career_lengths, jobs_indices, jobs_words, job_lengths = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]
    return ids, embs, career_lengths, jobs_indices, jobs_words, job_lengths


def collate_profiles_skth(batch):
    ids = [p[0][0] for p in batch]
    jobs = [p[0][1:][0] for p in batch]
    lengths = [p[1][1:][0] for p in batch]
    return ids, jobs, lengths


def transform_indices(person, lengths):
    # for person in people:
    jobs = person[1][1:]
    len_jobs = lengths[1][1:]
    last_job = person[1][0]
    len_last_job = lengths[1][0]
    identifier = person[0]
    return identifier, jobs, len_jobs, last_job, len_last_job


def transform_for_elmo(person, lengths):
    identifier = person[0]
    jobs = person[1][1:]
    return identifier, jobs, lengths - 1


def transform_for_elmo_lj(person, lengths):
    identifier = person[0]
    jobs = person[1]
    if len(person) > 2:
        last_job = person[3]
        last_job_len = person[4]
    else:
        last_job, last_job_len = None, None
    return identifier, jobs, lengths, last_job, last_job_len


def transform_for_elmo_indices(identifier, emb, indices, lengths):
    return identifier, emb, lengths, indices


def build_profile_embeddings_old(batch_size, MAX_SEQ_LENGTH, MAX_CAREER_LENGTH, profile, profile_length, vocab_index):
    """This method returns a list of indices corresponding to the words embeddings of a career"""
    assert MAX_SEQ_LENGTH >= max(max(profile_length))
    indices = torch.zeros(batch_size, MAX_SEQ_LENGTH, MAX_CAREER_LENGTH-1)
    for i in range(batch_size):
        for j, exp in enumerate(profile[i]):
            counter = 0
            for word in exp:
                if counter < MAX_SEQ_LENGTH and j < MAX_CAREER_LENGTH-1:
                    if word not in vocab_index.keys():
                        indices[i][counter][j] = vocab_index["UNK"]
                    else:
                        indices[i][counter][j] = vocab_index[word]
                    counter += 1
    return indices


def indices_list_to_one_hot_tensor(indice_list, vocab_index):
    indices = torch.zeros(len(indice_list), len(vocab_index))
    for i, ind in enumerate(indice_list):
        indices[i][ind] = 1
    return indices.cuda()


def labels_to_indices(labels, vocab_index):
    indices = torch.zeros(len(labels))
    for i, word in enumerate(labels):
        if word not in vocab_index.keys():
            indices[i] = vocab_index["UNK"]
        else:
            indices[i] = vocab_index[word]
    return indices.type(torch.LongTensor).cuda()


def pad_pred(pred, labels_len):
    new_tensor = torch.zeros(labels_len, pred.shape[-1], dtype=pred.dtype)
    new_tensor[:len(pred), :] = pred
    return new_tensor.cuda()


def pad_labels(labels, pred_len):
    new_tensor = torch.zeros(pred_len, dtype=labels.dtype)
    new_tensor[:len(labels)] = labels
    return new_tensor.cuda()


def compute_crossentropy(prediction, labels):
    return torch.nn.functional.cross_entropy(prediction.cuda(), labels.cuda(), ignore_index=0, reduction='sum')


def compute_nll_eval(prediction, labels):
    return torch.nn.functional.nll_loss(prediction.cuda(), labels.cuda(), ignore_index=0, reduction='none')


def save_best_model(args, epoch, target_dir, encoder, decoder, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) + \
                   '_' + str(args.voc_size) + '_'
    for e in range(epoch):
        prev_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_best_ep_' + str(e)))
        prev_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_best_ep_' + str(e)))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(e)))
        if len(prev_model_enc) > 0:
            os.remove(prev_model_enc[0])
        if len(prev_model_dec) > 0:
            os.remove(prev_model_dec[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir,  file_pattern + 'enc_best_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir,  file_pattern + 'dec_best_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(epoch)))


def save_best_model_elmo(args, epoch, target_dir, decoder, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.dec_hidden_size) + \
                   '_max_ep_' + str(args.epoch) + \
                   '_' + str(args.voc_size) + '_'
    for e in range(epoch):
        # prev_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_best_ep_' + str(e)))
        prev_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_best_ep_' + str(e)))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(e)))
        # if len(prev_model_enc) > 0:
        #     os.remove(prev_model_enc[0])
        if len(prev_model_dec) > 0:
            os.remove(prev_model_dec[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    # torch.save(encoder.state_dict(), os.path.join(target_dir, file_pattern + 'enc_best_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir,  file_pattern + 'dec_best_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(epoch)))


def save_best_model_elmo_w2v(args, epoch, target_dir, encoder, decoder, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) + '_'
    for e in range(epoch):
        prev_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'encCareer_best_ep_' + str(e)))
        prev_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'decCareer_best_ep_' + str(e)))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optimCareer_best_ep_' + str(e)))
        if len(prev_model_enc) > 0:
            os.remove(prev_model_enc[0])
        if len(prev_model_dec) > 0:
            os.remove(prev_model_dec[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir, file_pattern + 'encCareer_best_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir,  file_pattern + 'decCareer_best_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optimCareer_best_ep_' + str(epoch)))


def save_best_model_v2(args, epoch, target_dir, encoder, decoder, model, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) + \
                   '_' + str(args.voc_size) + '_'
    for e in range(epoch):
        prev_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_best_ep_' + str(e)))
        prev_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_best_ep_' + str(e)))
        prev_mlp = glob.glob(os.path.join(target_dir, file_pattern + 'mlp_best_ep_' + str(e)))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(e)))
        if len(prev_model_enc) > 0:
            os.remove(prev_model_enc[0])
        if len(prev_model_dec) > 0:
            os.remove(prev_model_dec[0])
        if len(prev_mlp) > 0:
            os.remove(prev_mlp[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir,  file_pattern + 'enc_best_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir,  file_pattern + 'dec_best_ep_' + str(epoch)))
    torch.save(model.state_dict(), os.path.join(target_dir,  file_pattern + 'mlp_best_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(epoch)))


def save_best_model_v3(args, epoch, target_dir, encoder, decoder, rnn, mlp, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_dpo' + str(args.dpo) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) + \
                   '_' + str(args.voc_size) + '_'
    for e in range(epoch):
        prev_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_best_ep_' + str(e)))
        prev_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_best_ep_' + str(e)))
        prev_rnn = glob.glob(os.path.join(target_dir, file_pattern + 'rnn_best_ep_' + str(e)))
        prev_mlp = glob.glob(os.path.join(target_dir, file_pattern + 'mlp_best_ep_' + str(e)))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(e)))
        if len(prev_model_enc) > 0:
            os.remove(prev_model_enc[0])
        if len(prev_model_dec) > 0:
            os.remove(prev_model_dec[0])
        if len(prev_rnn) > 0:
            os.remove(prev_rnn[0])
        if len(prev_mlp) > 0:
            os.remove(prev_mlp[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir,  file_pattern + 'enc_best_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir,  file_pattern + 'dec_best_ep_' + str(epoch)))
    torch.save(rnn.state_dict(), os.path.join(target_dir,  file_pattern + 'rnn_best_ep_' + str(epoch)))
    torch.save(mlp.state_dict(), os.path.join(target_dir,  file_pattern + 'mlp_best_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(epoch)))


def save_best_classifier(args, epoch, target_dir, classifier, optim, suffix):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_max_ep_' + str(args.epoch) + '_'
    for e in range(epoch):
        prev_classif = glob.glob(os.path.join(target_dir, file_pattern + '_best_ep_' + str(e) + suffix))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + '_optim_best_ep_' + str(e) + suffix))
        if len(prev_classif) > 0:
            os.remove(prev_classif[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    torch.save(classifier.state_dict(), os.path.join(target_dir,  file_pattern + '_best_ep_' + str(epoch) + suffix))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + '_optim_best_ep_' + str(epoch) + suffix))


def save_best_classifier_ft(args, epoch, target_dir, classifier, optim, suffix):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_max_ep_' + str(args.epoch) + '_'
    for e in range(epoch):
        prev_classif = glob.glob(os.path.join(target_dir, file_pattern + '_best_ep_' + str(e) + "_ft" + suffix))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + '_optim_best_ep_' + str(e) + "_ft" + suffix))
        if len(prev_classif) > 0:
            os.remove(prev_classif[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    torch.save(classifier.state_dict(), os.path.join(target_dir,  file_pattern + '_best_ep_' + str(epoch) + "_ft" + suffix))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + '_optim_best_ep_' + str(epoch) + "_ft" + suffix))


def save_best_model_mlp_rnn(args, epoch, target_dir, decoder, rnn, mlp, optim, dec_hs, dec_lr, dec_ep):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_dpo' + str(args.dpo) + \
                   '_max_ep_' + str(args.epoch) + \
                   '_dechs_' + str(dec_hs) + \
                   '_declr_' + str(dec_lr) + \
                   '_decep_' + str(dec_ep) + \
                   '_' + str(args.voc_size) + '_'
    for e in range(epoch):
        prev_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_best_ep_' + str(e)))
        prev_rnn = glob.glob(os.path.join(target_dir, file_pattern + 'rnn_best_ep_' + str(e)))
        prev_mlp = glob.glob(os.path.join(target_dir, file_pattern + 'mlp_best_ep_' + str(e)))
        prev_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(e)))
        if len(prev_model_dec) > 0:
            os.remove(prev_model_dec[0])
        if len(prev_rnn) > 0:
            os.remove(prev_rnn[0])
        if len(prev_mlp) > 0:
            os.remove(prev_mlp[0])
        if len(prev_optim) > 0:
            os.remove(prev_optim[0])
    torch.save(decoder.state_dict(), os.path.join(target_dir,  file_pattern + 'dec_best_ep_' + str(epoch)))
    torch.save(rnn.state_dict(), os.path.join(target_dir,  file_pattern + 'rnn_best_ep_' + str(epoch)))
    torch.save(mlp.state_dict(), os.path.join(target_dir,  file_pattern + 'mlp_best_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_best_ep_' + str(epoch)))


def word_seq_to_one_hot_tensor(word_seq, vocab_index):
    indices = torch.zeros(len(word_seq), len(vocab_index))
    indices = indices.type(dtype=torch.FloatTensor)
    for i, ind in enumerate(word_seq):
        indices[i][ind] = 1
    return indices


def model_checkpoint(args, epoch, target_dir, encoder, decoder, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) +\
                   '_' + str(args.voc_size) + '_'
    if epoch > 1:
        last_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch - 1)))
        last_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch - 1)))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch - 1)))
        if len(last_model_enc) > 0:
            os.remove(last_model_enc[0])
        if len(last_model_dec) > 0:
            os.remove(last_model_dec[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch)))


def model_checkpoint_elmo(args, epoch, target_dir, decoder, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.dec_hidden_size) + \
                   '_max_ep_' + str(args.epoch) +\
                   '_' + str(args.voc_size) + '_'
    if epoch > 1:
        # last_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch - 1)))
        last_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch - 1)))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch - 1)))
        # if len(last_model_enc) > 0:
        #     os.remove(last_model_enc[0])
        if len(last_model_dec) > 0:
            os.remove(last_model_dec[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    # torch.save(encoder.state_dict(), os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch)))


def model_checkpoint_elmo_w2v(args, epoch, target_dir, encoder, decoder, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) + '_'
    if epoch > 1:
        last_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'encCareer_last_ep_' + str(epoch - 1)))
        last_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'decCareer_last_ep_' + str(epoch - 1)))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optimCareer_last_ep_' + str(epoch - 1)))
        if len(last_model_enc) > 0:
            os.remove(last_model_enc[0])
        if len(last_model_dec) > 0:
            os.remove(last_model_dec[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir, file_pattern + 'encCareer_last_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir, file_pattern + 'decCareer_last_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optimCareer_last_ep_' + str(epoch)))


def model_checkpoint_classifier(args, epoch, target_dir, classifier, optim, suffix):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_max_ep_' + str(args.epoch) + '_'
    if epoch > 1:
        last_model = glob.glob(os.path.join(target_dir, file_pattern + 'last_ep_' + str(epoch - 1) + suffix))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch - 1) + suffix))
        if len(last_model) > 0:
            os.remove(last_model[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    torch.save(classifier.state_dict(), os.path.join(target_dir, file_pattern + 'last_ep_' + str(epoch) + suffix))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch) + suffix))


def model_checkpoint_classifier_ft(args, epoch, target_dir, classifier, optim, suffix):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_max_ep_' + str(args.epoch) + '_'
    if epoch > 1:
        last_model = glob.glob(os.path.join(target_dir, file_pattern + 'last_ep_' + str(epoch - 1) + "_ft" + suffix))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch - 1) + "_ft" + suffix))
        if len(last_model) > 0:
            os.remove(last_model[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    torch.save(classifier.state_dict(), os.path.join(target_dir, file_pattern + 'last_ep_' + str(epoch) + "_ft" + suffix))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch) + "_ft" + suffix))


def model_checkpoint_v2(args, epoch, target_dir, encoder, decoder, model, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_tf' + str(args.tf) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) +\
                   '_' + str(args.voc_size) + '_'
    if epoch > 1:
        last_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch - 1)))
        last_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch - 1)))
        last_model_mlp = glob.glob(os.path.join(target_dir, file_pattern + 'mlp_last_ep_' + str(epoch - 1)))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch - 1)))
        if len(last_model_enc) > 0:
            os.remove(last_model_enc[0])
        if len(last_model_dec) > 0:
            os.remove(last_model_dec[0])
        if len(last_model_mlp) > 0:
            os.remove(last_model_mlp[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch)))
    torch.save(model.state_dict(), os.path.join(target_dir, file_pattern + 'mlp_last_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch)))


def model_checkpoint_v3(args, epoch, target_dir, encoder, decoder, rnn, mlp, optim):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_dpo' + str(args.dpo) + \
                   '_hs_' + str(args.hidden_size) + \
                   '_max_ep_' + str(args.epoch) +\
                   '_' + str(args.voc_size) + '_'
    if epoch > 1:
        last_model_enc = glob.glob(os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch - 1)))
        last_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch - 1)))
        last_model_rnn = glob.glob(os.path.join(target_dir, file_pattern + 'rnn_last_ep_' + str(epoch - 1)))
        last_model_mlp = glob.glob(os.path.join(target_dir, file_pattern + 'mlp_last_ep_' + str(epoch - 1)))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch - 1)))
        if len(last_model_enc) > 0:
            os.remove(last_model_enc[0])
        if len(last_model_dec) > 0:
            os.remove(last_model_dec[0])
        if len(last_model_rnn) > 0:
            os.remove(last_model_rnn[0])
        if len(last_model_mlp) > 0:
            os.remove(last_model_mlp[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    torch.save(encoder.state_dict(), os.path.join(target_dir, file_pattern + 'enc_last_ep_' + str(epoch)))
    torch.save(decoder.state_dict(), os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch)))
    torch.save(rnn.state_dict(), os.path.join(target_dir, file_pattern + 'rnn_last_ep_' + str(epoch)))
    torch.save(mlp.state_dict(), os.path.join(target_dir, file_pattern + 'mlp_last_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch)))


def model_checkpoint_mlp_rnn(args, epoch, target_dir, decoder, rnn, mlp, optim, dec_hs, dec_lr, dec_ep):
    file_pattern = str(args.model_type) + \
                   '_bs' + str(args.batch_size) + \
                   '_lr' + str(args.lr) + \
                   '_dpo' + str(args.dpo) + \
                   '_max_ep_' + str(args.epoch) + \
                   '_dechs_' + str(dec_hs) + \
                   '_declr_' + str(dec_lr) + \
                   '_decep_' + str(dec_ep) + \
                   '_' + str(args.voc_size) + '_'
    if epoch > 1:
        last_model_dec = glob.glob(os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch - 1)))
        last_model_rnn = glob.glob(os.path.join(target_dir, file_pattern + 'rnn_last_ep_' + str(epoch - 1)))
        last_model_mlp = glob.glob(os.path.join(target_dir, file_pattern + 'mlp_last_ep_' + str(epoch - 1)))
        last_optim = glob.glob(os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch - 1)))
        if len(last_model_dec) > 0:
            os.remove(last_model_dec[0])
        if len(last_model_rnn) > 0:
            os.remove(last_model_rnn[0])
        if len(last_model_mlp) > 0:
            os.remove(last_model_mlp[0])
        if len(last_optim) > 0:
            os.remove(last_optim[0])
    torch.save(decoder.state_dict(), os.path.join(target_dir, file_pattern + 'dec_last_ep_' + str(epoch)))
    torch.save(rnn.state_dict(), os.path.join(target_dir, file_pattern + 'rnn_last_ep_' + str(epoch)))
    torch.save(mlp.state_dict(), os.path.join(target_dir, file_pattern + 'mlp_last_ep_' + str(epoch)))
    torch.save(optim.state_dict(), os.path.join(target_dir, file_pattern + 'optim_last_ep_' + str(epoch)))


def separate_title_from_desc(sequence, delimiter_token):
    title = []
    desc = []
    flag_end_of_title = False
    for tok in sequence:
        if not flag_end_of_title:
            title.append(tok)
            if tok == delimiter_token:
                flag_end_of_title = True
        else:
            desc.append(tok)
    return title, desc

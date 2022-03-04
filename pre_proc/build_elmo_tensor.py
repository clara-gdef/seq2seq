import os
import pickle as pkl
import argparse
import torch
from tqdm import tqdm
from allennlp.modules.elmo import Elmo


def main(args):
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    print("Loading word vectors...")
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    print("Word vectors loaded.")
    with open(os.path.join(args.DATA_DIR, "pkl/index_40k.pkl"), "wb") as f:
        vocab = pkl.load(f)

    build_index_and_tensor(vocab, elmo, args)


def build_index_and_tensor(word_list, ft, args):
    word_to_index = dict()
    print("Length of the vocabulary: " + str(len(word_list)))
    with tqdm(total=len(word_list), desc="Building tensors and index...") as pbar:
        tensor_updated, w2i_updated, num_tokens = build_special_tokens(word_to_index)
        for i, word in enumerate(word_list):
            if word is not '':
                character_ids = batch_to_ids(word)
                tensor_updated = torch.cat([tensor_updated, torch.FloatTensor(ft.get_word_vector(word)).view(1, -1)], dim=0)
                w2i_updated[word] = i + num_tokens
            pbar.update(1)
    print(len(word_to_index))
    with open(os.path.join(args.DATA_DIR, "pkl/tensor_elmo_40k.pkl"), "wb") as f:
        pkl.dump(tensor_updated, f)


def build_special_tokens(word_to_index):
    """
    SOT stands for 'start of title'
    EOT stands for 'end of title'
    SOD stands for 'start of description'
    EOD stands for 'end of description'
    PAD stands for 'padding inde
    UNK stands for 'unknown word'
    """
    SOT = torch.randn(1, 1024)
    EOT = torch.randn(1, 1024)
    SOD = torch.randn(1, 1024)
    EOD = torch.randn(1, 1024)
    PAD = torch.randn(1, 1024)
    UNK = torch.randn(1, 1024)
    word_to_index["PAD"] = 0
    tensor = PAD
    word_to_index["SOT"] = 1
    tensor = torch.cat([tensor, SOT], dim=0)
    word_to_index["EOT"] = 2
    tensor = torch.cat([tensor, EOT], dim=0)
    word_to_index["SOD"] = 3
    tensor = torch.cat([tensor, SOD], dim=0)
    word_to_index["EOD"] = 4
    tensor = torch.cat([tensor, EOD], dim=0)
    word_to_index["UNK"] = 5
    tensor = torch.cat([tensor, UNK], dim=0)
    return tensor, word_to_index, 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pkl/jobs.pkl")
    parser.add_argument("--model_version", type=str, default='s2s')
    parser.add_argument("--DATA_DIR", type=str, default='/local/gainondefor/work/data')
    parser.add_argument("--pre_trained_model", type=str, default='ft_pre_trained.bin')
    parser.add_argument("--max_voc_len", type=int, default=40000)
    parser.add_argument("--min_occurence", type=int, default=5)
    args = parser.parse_args()
    main(args)


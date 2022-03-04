import ipdb
import argparse
import pickle as pkl
from collections import Counter
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def main(args):
    with ipdb.launch_ipdb_on_exception():
        val_set_profiles, val_set_labels = get_data_to_plot()
        for iteration in range(10):
            plot_pca_tsne(val_set_profiles, val_set_labels, iteration)



def get_data_to_plot():
    suffix = args.ft_type

    if suffix != "elmo":
        with open(os.path.join(args.DATA_DIR, "valid_industry" + suffix + ".pkl"), "rb") as f:
            validset = pkl.load(f)
    else:
        # arguments
        emb_option = "jobs"
        result_directory_name = "industry_classification/results_jobs_mean"
        result_dir = "/local/gainondefor/work/scherer/" + str(result_directory_name) + "/"
        data_path = "/local/gainondefor/work/scherer/embeddings_clara/" + emb_option + "/"
        labels_path = "/local/gainondefor/work/scherer/embeddings_clara/labels.p"
        indus_path = "/local/gainondefor/work/scherer/embeddings_clara/industries_count.p"
        partition_path = "/local/gainondefor/work/scherer/embeddings_clara/partition.p"
        pickles_indices = "/local/gainondefor/work/scherer/embeddings_clara/pickles_indices/degree.p"
        labels = pkl.load(open(labels_path, "rb"))

        indus = pkl.load(open(indus_path, "rb"))

        # labels : {"ID": str_labels} for all data
        labels = pkl.load(open(labels_path, "rb"))

        partition = pkl.load(open(partition_path, "rb"))
        indices_ok = pkl.load(open(pickles_indices, "rb"))
        filenames = sorted(os.listdir(data_path))
        good_files = set([int(ind.rstrip('.pt')) for ind in filenames])
        labels_mapping = pkl.load(open(result_dir + "labels_mapping.p", "rb"))
        # pour virer les gens nuls
        partition['valid'] = set(indices_ok).intersection(partition['valid'])
        partition['valid'] = good_files.intersection(partition['valid'])

        validset = []

        l_part = list(partition['test'])
        for i in tqdm(range(0, len(l_part)), desc='Building dataset...'):
            ID = l_part[i]
            liste_jobs = torch.load(data_path + str(ID) + ".pt")
            t = torch.mean(torch.stack(liste_jobs), dim=0)
            validset.append((ID, t, labels_mapping[labels[ID]]))

    selected_classes = get_most_common_classes(validset)
    val_set_profiles = []
    val_set_labels = []
    for i in tqdm(validset, desc="buildind dataset..."):
        if len(val_set_labels) < args.subsample:
            if i[-1] in selected_classes:
                val_set_profiles.append(i[1].numpy())
                val_set_labels.append(i[-1])
    return val_set_profiles, val_set_labels

def plot_pca_tsne(val_set_profiles, val_set_labels, iteration):
    print("num of different labels : " + str(len(set(val_set_labels))))
    print("Running PCA on data...")
    pca = PCA(n_components=2)
    t0 = time()
    reduced_val_set = pca.fit_transform(val_set_profiles)
    t1 = time()
    print("PCA decomposition achieved.")

    print("Fitting tsne...")
    t2 = time()
    val_set_embedded_tsne = TSNE(n_components=2).fit_transform(reduced_val_set[:args.subsample])
    t3 = time()

    print("Profiles to plot : " + str(args.subsample))
    # Plot results
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("PCA & TSNE of %s profiles (%s) with %i points"
                 % (args.prof_type, args.ft_type, args.subsample), fontsize=14)
    ax = fig.add_subplot(211)
    ax.scatter(reduced_val_set[:args.subsample, 0], reduced_val_set[:args.subsample, 1],
               c=val_set_labels[:args.subsample], cmap=plt.cm.Spectral)
    ax.set_title("PCA over %s points (%.2g sec)" % (args.subsample, t1 - t0))
    ax.axis('tight')

    ax = fig.add_subplot(212)
    ax.scatter(val_set_embedded_tsne[:args.subsample, 0], val_set_embedded_tsne[:, 1],
               c=val_set_labels[:args.subsample], cmap=plt.cm.Spectral)
    ax.set_title("TSNE over %s points (%.2g sec)" % (args.subsample, t3 - t2))
    ax.axis('tight')
    plt.savefig('tsne_pca_' + str(args.subsample) + "_" + args.ft_type + "_" + str(iteration) + '.png')


def get_most_common_classes(dataset):
    cnt = Counter()
    for item in dataset:
        cnt[item[-1]] += 1
    mc_classes = [i[0] for i in cnt.most_common(10)]
    return mc_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/data/gainondefor/seq2seq/")
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--prof_type", type=str, default="jobs")
    parser.add_argument("--ft_type", type=str, default="fs")
    args = parser.parse_args()
    main(args)

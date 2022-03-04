import argparse
import os
import pickle as pkl

import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def main(args):

    enc_type = str.split(args.enc_model, sep="_")[1]

    if enc_type == "w2v":
        enc_hs = str.split(args.enc_model, sep="_")[8]
        enc_lr = str.split(args.enc_model, sep="_")[5]
        enc_ep = str.split(args.enc_model, sep="_")[-1]
    else:
        enc_hs = str.split(args.enc_model, sep="_")[9]
        enc_lr = str.split(args.enc_model, sep="_")[3]
        enc_ep = str.split(args.enc_model, sep="_")[-1]

    suffix = "_" + str(enc_type) + "_" + str(enc_lr) + "_" + str(enc_hs) + "_" + str(enc_ep)

    with open(os.path.join(args.DATA_DIR, args.classes_file), "rb") as f:
        classes = pkl.load(f)

    # with ipdb.launch_ipdb_on_exception():
        # if args.compute_rep:
        #     print("Loading data...")
        #     with open(os.path.join(args.DATA_DIR, "test_skills" + suffix + ".pkl"), 'rb') as f:
        #         data_test = pkl.load(f)
        #
        #     print("Data loaded.")
        #
        #     input_size = int(enc_hs) * 2
        #     classifier = SkillsPredictor(input_size, input_size, len(classes))
        #     weights = os.path.join(args.model_dir, args.sk_model)
        #     classifier.load_state_dict(torch.load(weights))
        #
        #     classifier = classifier.cuda()
        #
        #     dataloader_test = DataLoader(data_test, batch_size=args.batch_size, collate_fn=collate,
        #                               shuffle=True, num_workers=0, drop_last=True)
        #     counter = 0
        #     reps = []
        #     for ids, tensors, labels in tqdm(dataloader_test, desc="Building Representations..."):
        #         if counter <= args.num_people:
        #             rep = classifier.extract(torch.cat(tensors).cuda())
        #             reps.append((ids[0], rep, labels))
        #     with open(os.path.join(args.DATA_DIR, "rep_skills" + suffix + ".pkl"), 'wb') as f:
        #         pkl.dump(reps, f)
        # else:
        #     print("Loading data...")
        #     with open(os.path.join(args.DATA_DIR, "rep_skills" + suffix + ".pkl"), 'rb') as f:
        #         reps = pkl.load(f)
        #     print("Data loaded.")
    # cnt = Counter()
    # for p in tqdm(lab.values()):
    #     for i, j in enumerate(p):
    #         if p[i] > 0:
    #             cnt[i] += 1
    #
    # tmp = dict()
    # for i in range(len(classes)):
    #     tmp[classes[i]] = cnt[i]

    with open(os.path.join(args.DATA_DIR, "pkl/counted_skills.pkl"), "rb") as f:
        classes_counted = pkl.load(f)

    weights = os.path.join(args.model_dir, args.sk_model)
    w2 = torch.load(weights)["layer2.weight"]

    projection(w2, classes, suffix, classes_counted)


def projection(w2, classes, suffix, classes_counted):
    w_numpy = w2.cpu().numpy()

    labels = dict()
    for i in range(len(classes)):
        labels[i] = classes[i]

    rev_dict = {v: k for k, v in labels.items()}
    #
    # most_common = sorted([(v, k) for k, v in classes_counted.items()], reverse=True)[:52]
    # mc_labels = [e[1] for e in most_common]
    # least_common = sorted([(v, k) for k, v in classes_counted.items()], reverse=True)[-52:]
    # # lc_labels = [e[1] for e in least_common]
    #
    # colors_to_print = []
    # for skill in range(len(classes)):
    #     if labels[skill] in mc_labels:
    #         colors_to_print.append("b")
    #     # elif labels[skill] in lc_labels:
    #     #     colors_to_print.append("g")
    #     else:
    #         colors_to_print.append('0.75')
    #
    # # retained_indices = [rev_dict[s] for s in itertools.chain(mc_labels, lc_labels)]
    # retained_indices = [rev_dict[s] for s in mc_labels]


    lang = ["SQL", "PostgreSQL", "JavaScript", "Android", "Java", "C", "C++", "Python", "HTML", "CSS", "VBA", "Informatique"]
    mgmt = ["Marketing", "Stratégie marketing", "Marketing digital", "Communication marketing", "Management", "Agile Methodologies", "Scrum", "Gestion de projet logiciel", "Web Project Management"]
    droit = ["Droit international", "Rédaction juridique", "Recherche juridique", "Corporate Law"]
    graphism = ['Adobe Photoshop', 'Adobe Illustrator', 'Adobe Creative Suite', 'Graphisme', 'Photoshop', 'Web Design']
    office = ['Microsoft Excel', 'PowerPoint', 'Microsoft Word', 'Microsoft Office']
    langues = ["Français", "Anglais", "English", "Espagnol", "French"]
    misc = ["Art", "Tourism", "Music", "Teaching", "Sports", "Research", "Architecture", "Enseignement", "Vente", "Video", "Radio"]

    retained_skills = lang + mgmt + droit + graphism + office + langues + misc

    retained_indices = [rev_dict[i] for i in retained_skills]

    colors_to_print = []
    for skill in range(len(classes)):
        if labels[skill] in lang:
            colors_to_print.append("b")
        elif labels[skill] in mgmt:
            colors_to_print.append("g")
        elif labels[skill] in droit:
            colors_to_print.append('r')
        elif labels[skill] in graphism:
            colors_to_print.append("c")
        elif labels[skill] in office:
            colors_to_print.append("m")
        elif labels[skill] in langues:
            colors_to_print.append("y")
        elif labels[skill] in misc:
            colors_to_print.append("k")
        else:
            colors_to_print.append('0.75')

    pca = PCA(n_components=2)
    data = pca.fit_transform(w_numpy)
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])

    fig, ax = plt.subplots()

    for e, i in enumerate(range(len(classes))):
        ax.scatter(x[i], y[i], color=colors_to_print[e])

    for i in retained_indices:
        ax.scatter(x[i], y[i], color=colors_to_print[i])
        plt.text(x[i] + .01, y[i] + .01, labels[i])

    plt.title("PCA cat " + suffix)
    plt.savefig("img/PCA_cat_" + suffix + ".svg", format='svg')
    plt.show()


def collate(batch):
    ids = [e[0] for e in batch]
    tensors = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return ids, tensors, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/data")
    parser.add_argument("--index_file", type=str, default="pkl/index_40k.pkl")
    parser.add_argument("--classes_file", type=str, default="pkl/good_skills.p")
    parser.add_argument("--label_file", type=str, default="pkl/labels_skills.p")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/seq2seq/elmo/elmo_w2v')
    parser.add_argument("--enc_model", type=str, default="elmo_w2v_gradclip_sgd_bs64_lr0.0001_tf0_hs_512_max_ep_300_encCareer_best_ep_185")
    parser.add_argument("--sk_model", type=str, default="skills_bs128_lr0.001_max_ep_100__best_ep_97_w2v_lr0.0001_512_185")
    args = parser.parse_args()
    main(args)

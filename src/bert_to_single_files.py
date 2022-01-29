import collections
import glob
import json

import torch
from tqdm import tqdm

PT_BASE = "/disk1/sajad/datasets/sci/arxivL//bert-files/2048-segmented-intro1536-15-introConc-updated/-updated15/"
# for set in ["test", "val", "train"]:
# for set in ["test", "val", "train"]:
for set in ["test"]:
    paper_sects = collections.defaultdict(dict)
    paper_labels = collections.defaultdict(dict)
    for file in tqdm(glob.glob(PT_BASE + set + "*.pt"),
                     total=len(glob.glob(PT_BASE + set + "*.pt"))):
        instances = torch.load(file)

        for instance in instances:
            sent_numbers = instance['sent_numbers']
            sent_sect_labels = instance['sent_sect_labels']
            sent_labels = instance['sent_labels']
            paper_id = instance['paper_id'].split('___')[0]

            for idx, num in enumerate(sent_numbers):
                paper_sects[paper_id][num] = sent_sect_labels[idx]
                paper_labels[paper_id][num] = sent_labels[idx]

    for json_file in tqdm(glob.glob("/disk1/sajad/datasets/sci/arxivL/splits/" + set + "/*.json"),
                          total=len(glob.glob("/disk1/sajad/datasets/sci/arxivL/splits/" + set + "/*.json"))):
        paper = json.load(open(json_file))
        json_id = paper["id"]

        sect_labels = paper_sects[json_id]
        sent_labels = paper_labels[json_id]

        new_sents = []
        for idx, sent in enumerate(paper["sentences"]):
            try:
                # sent.append(sect_labels[idx])
                sent[-2] = sent_labels[idx]
            except:
                # sent[-1] = sent_labels[idx]
                sent[-2] = 0

                # sent.append(4)
            new_sents.append(sent)
        paper["sentences"] = new_sents
        json.dump(paper, open(json_file.replace("splits","splits-with-sections-introConc"), mode='w'))

import glob
import json
import os.path
from os.path import join as pjoin

from tqdm import tqdm

abs_summaries = {
    'train': {},
    'val': {},
    'test': {}
}

ABS_BASE_DIR = "/disk1/sajad/sci-trained-models/bart/saved_models/bart/bart-intro-arxivL-cnn/checkpoint-3000/"

for set in abs_summaries.keys():
    with open(pjoin(ABS_BASE_DIR, f'generated_predictions_{set}.txt')) as fRP, open(f'/disk1/sajad/datasets/sci/arxivL/intro_summary/{set}.json') as fRI:
        for pred, id in zip(fRP, fRI):
            id = json.loads(id.strip())['paper_id']
            abs_summaries[set][id] = pred

MY_FORMAT_SPLITS = "/disk1/sajad/datasets/sci/arxivL/splits/"
MY_FORMAT_SPLITS_WITH_INTRO_SUM = "/disk1/sajad/datasets/sci/arxivL/intro_summary/splits/"
for set in abs_summaries.keys():
    if not os.path.exists(pjoin(MY_FORMAT_SPLITS_WITH_INTRO_SUM, set)):
        os.makedirs(pjoin(MY_FORMAT_SPLITS_WITH_INTRO_SUM, set))
    not_found = 0

    for f in tqdm(glob.glob(pjoin(MY_FORMAT_SPLITS, set + "/*.json")), total=len(glob.glob(pjoin(MY_FORMAT_SPLITS, set + "/*.json")))):
        paper_ent = json.load(open(f))

        try:
            paper_ent['intro_summary'] = abs_summaries[set][paper_ent['id']]
        except:
            not_found+=1
            continue
        json.dump(paper_ent, open(pjoin(MY_FORMAT_SPLITS_WITH_INTRO_SUM, set + f'/{paper_ent["id"]}.json'), mode='w'))

    print(f'not found for {set}:  {not_found}')
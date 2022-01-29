import json
import os
import random
import sys
# sys.path.insert(0, '/home/sajad/packages/sum/introAbsGuided/src/')

from api_abstract_fetch import call_api


def _return_id_segments(machine, abstract_ids):
    if machine == 'barolo':
        return abstract_ids[:len(abstract_ids)//3]

    if machine == 'chianti':
        return abstract_ids[len(abstract_ids)//3:2*len(abstract_ids)//3]

    if machine == 'brunello':
        return abstract_ids[2 * len(abstract_ids) // 3:]

if __name__ == '__main__':
    BASE_DS_DIR = "/disk1/sajad/datasets/sci/arxivL/splits/"



    # for se in ['test', 'val']:
    for se in ['train']:

        if not os.path.exists(BASE_DS_DIR + f'abstracts_{se}_main.json'):
            paper_references = {}
        else:
            paper_references = json.load(open(BASE_DS_DIR + f'abstracts_{se}_main.json', mode='r'))
        abstract_ids = []

        for f in os.listdir(BASE_DS_DIR + se + '/'):
            abstract_ids.append(f.replace('.json', '').strip())

        random.seed(8888)
        random.shuffle(abstract_ids)

        # MACHINE = str(sys.argv[-1])
        # abstract_ids = _return_id_segments(MACHINE, abstract_ids)

        missing = [a for a in abstract_ids if a not in paper_references.keys()]


        call_api(missing, wr_file_dir=BASE_DS_DIR + f'abstracts_{se}_main.json', paper_references=paper_references)
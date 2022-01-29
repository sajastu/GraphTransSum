import glob
import json
import os
import random
import re
from multiprocessing import Pool

from tqdm import tqdm

import rouge_score_utils as rg

import scispacy
import spacy
nlp = spacy.load("en_core_sci_md")


def _ref_abstracts(paper_abstract, refs):
    ref_abstracts = []
    paper_abstract_str = ' '.join(' '.join(t) for t in paper_abstract)
    for ref in refs:
        ref_abstract = ref['citedPaper']['abstract']
        if ref_abstract is not None:

            # ref_abstract = re.sub(r'(?<=[ ][[]{2}).*?(?=[]]{2}[ .!?])|\d+|\b[a-zA-Z0-9{+-}]*\([a-zA-Z0-9{+-}]*\)|\b[a-zA-Z]\b', '', ref_abstract)
            r1, r2, rL = rg.evaluate_rouge([ref_abstract], [paper_abstract_str])
            r_mean = (r2 + rL) / 2
            ref_abstracts.append((ref_abstract, r_mean))
    # sort ref abstracts
    ref_abstracts = sorted(ref_abstracts, key=lambda x:x[1], reverse=True)[:5]
    mode_ref_abstracts = []
    for ref_abs in ref_abstracts:
        abs_nlp = nlp(ref_abs[0]).sents
        mod_abs = ''
        for sent in abs_nlp:
            if len(sent) > 6:
                mod_abs += ' '
                mod_abs += sent.text
        mod_abs = str(mod_abs.strip().replace('\n', ' ').encode('ascii', 'ignore'))
        mod_abs = re.sub(r' +', ' ', mod_abs)
        mode_ref_abstracts.append(mod_abs)
    return mode_ref_abstracts



def _non_ref_abstracts(paper_abstract, all_abstracts):

    paper_abstract_str = ' '.join(' '.join(t) for t in paper_abstract)
    ref_abstracts = []

    all_abs_list = list(all_abstracts.items())

    random.shuffle(all_abs_list)
    all_abs_list = all_abs_list[:100]

    for paper_id, all_abs in all_abs_list:
        all_abs_str = ' '.join(' '.join(t) for t in all_abs)
        r1, r2, rL = rg.evaluate_rouge([all_abs_str], [paper_abstract_str])
        r_mean = (r2 + rL) / 2
        ref_abstracts.append((all_abs_str, r_mean))

    ref_abstracts = sorted(ref_abstracts, key=lambda x:x[1], reverse=True)[:5]
    ref_abstracts = [r[0] for r in ref_abstracts]

    return ref_abstracts

def _mp_abstract_fetch(params):

    if params[-1]:

        paper_id, abstract, refs, _ = params
        abstract, abstract_all = abstract
        ref_abstracts = _ref_abstracts(abstract, refs)
        ref_abstracts_rest = _non_ref_abstracts(abstract, abstract_all)
        ref_abstracts = [r.encode('ascii', 'ignore').decode('utf-8')[2:-2] for r in ref_abstracts] + ref_abstracts_rest[:5-len(ref_abstracts)]


    else:
        paper_id, abstract, ds_abstracts, _ = params
        ref_abstracts = _non_ref_abstracts(abstract, ds_abstracts)

    return {'paper_id': paper_id, 'ref_abstracts': ref_abstracts}

if __name__ == '__main__':
    BASE_DIR="/disk1/sajad/datasets/sci/arxivL/splits/"
    paper_ids = []
    arxivL_abstracts = {}
    # ref_abstarcts = json.load(open(f'/disk1/sajad/datasets/sci/arxivL/splits/all_ref_abstracts_train.json'))
    # import pdb;pdb.set_trace()
    for se in ['train']:
        for f in os.listdir(BASE_DIR + se  + '/'):
            paper_ids.append(f.replace('.json', ''))
            arxivL_abstracts[f.replace('.json', '')] = json.load(open(BASE_DIR + se + '/' + f))['gold']
        # read in abstracts
        abstracts = {}

        for f in glob.glob(BASE_DIR + 'abstracts_train_main.json'):
            abs = json.load(open(f))
            abstracts.update(abs)

        mp_all_abstracts = []
        all_ref_abstracts = {}
        for paper_id, abstract in arxivL_abstracts.items():
            try:
                # import pdb;pdb.set_trace()
                refs = abstracts[paper_id]
                if len([r for r in refs if r['citedPaper']['abstract'] is not None]) > 1 and len([r for r in refs if r['citedPaper']['abstract'] is not None]) < 5:
                    mp_all_abstracts.append((paper_id, (abstract, arxivL_abstracts), refs, True))
                    # pass
                else:
                    pass
                    # mp_all_abstracts.append((paper_id, abstract, arxivL_abstracts, False))

                # ref_abstracts = _ref_abstracts(abstract, refs)
            except:
                pass
                # mp_all_abstracts.append((paper_id, abstract, arxivL_abstracts, False))

        ## DEBUGGING
        # for a in mp_all_abstracts:
        #     if  a[-1]:
        #         _mp_abstract_fetch(a)

        pool = Pool(12)
        for out in tqdm(pool.imap_unordered(_mp_abstract_fetch, mp_all_abstracts), total=len(mp_all_abstracts)):
            all_ref_abstracts[out['paper_id']] = out['ref_abstracts']

            # try:
        #         if len(all_ref_abstracts) % 1000==0:
        #             with open(BASE_DIR + 'all_ref_abstracts_non.json', mode='w') as fW:
        #                     for id, refs in all_ref_abstracts.items():
        #                         try:
        #                             json.dump({
        #                                 'paper_id': id,
        #                                 'ref_abstracts': refs
        #                             },
        #                                 fW)
        #                             fW.write('\n')
        #                         except:
        #                             import pdb;pdb.set_trace()
        #     except:
        #         pass
        # with open(BASE_DIR + 'all_ref_abstracts_non.json', mode='w') as fW:
        #     try:
        #         for id, refs in all_ref_abstracts.items():
        #             json.dump({
        #                 'paper_id': id,
        #                 'ref_abstracts': refs
        #             },
        #             fW)
        #             fW.write('\n')
        #     except:
        #         with open('not_gotten_ref_abstracts.txt', mode='a') as fs:
        #             fs.write(id)
        #             fs.write('\n')

        json.dump(all_ref_abstracts, open(BASE_DIR +'all_ref_abstracts_non_2.json', mode='w'), indent=2)
    #
    # #
    #
    #
    #
    #

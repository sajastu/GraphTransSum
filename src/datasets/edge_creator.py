import glob
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import torch as torch
from tqdm import tqdm
from rouge_utils import _get_word_ngrams, cal_rouge
import dill

def _cal_rouge(sents, abstract):

    evaluated_1grams = [_get_word_ngrams(1, [sent.split()]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract[0].split()])
    evaluated_2grams = [_get_word_ngrams(2, [sent.split()]) for sent in sents]
    reference_2grams = _get_word_ngrams(2,  [abstract[0].split()])

    candidates_1 = [evaluated_1grams[idx] for idx in range(len(evaluated_1grams))]
    candidates_1 = set.union(*map(set, candidates_1))
    candidates_2 = [evaluated_2grams[idx] for idx in range(len(evaluated_2grams))]
    candidates_2 = set.union(*map(set, candidates_2))

    rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
    rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']


    return (rouge_1 + rouge_2) / 2

def _mp_rouge(params):
    i, j ,p_id, src_sent_1, src_sent_2, tgt = params
    rg_sent_1 = _cal_rouge([src_sent_1], [tgt])
    rg_sent_2 = _cal_rouge([src_sent_1] + [src_sent_2], [tgt])
    return {
        'index': (i, j),
        'rg_diff': rg_sent_2 - rg_sent_1,
        'paper_id': p_id
    }

if __name__ == '__main__':
    BASE_DIR = os.getenv('BERT_DIR')

    if not os.path.exists(BASE_DIR[:-1] + '-graph'):
        os.makedirs(BASE_DIR[:-1] + '-graph')

    BASE_DIR_WR = BASE_DIR[:-1] + '-graph'


    for se in ['train', 'val', 'test']:
    # for se in ['train']:
        prepare_array = []
        print('populating the ROUGE list...')

        for f in tqdm(glob.glob(BASE_DIR + '/' + se + '*.pt'), total=len(glob.glob(BASE_DIR + '/' + se + '*.pt'))):
            datasets = torch.load(f)


            for d in datasets:
                src_sents = d['src_tokens']
                if len(src_sents) == len(d['src_sent_labels']):
                    src_sents = [' '.join(s) for s in src_sents]
                    tgt = d['tgt_txt']
                    p_id = d['paper_id']

                    for i, src_sent_1 in enumerate(src_sents):
                        for j, src_sent_2 in enumerate(src_sents):
                            if j > i:
                                prepare_array.append((i, j, p_id, src_sent_1, src_sent_2, tgt))
                            # _mp_rouge(prepare_array[-1])
                else:
                    import pdb;pdb.set_trace()


        paper_id_edges =  defaultdict(lambda: defaultdict(dict))
        pool = Pool(15)

        # compute pair-wise ROUGE

        for out in tqdm(pool.imap_unordered(_mp_rouge, prepare_array), total=len(prepare_array)):
            paper_id_edges[out['paper_id']][out['index'][0]][out['index'][1]] = out['rg_diff']

        pool.close()
        pool.join()

        print(f'Now saving {se} graph...')

        torch.save(
            obj=paper_id_edges,
            f=f'/disk1/sajad/datasets/sci/arxivL/rouge_edge/{se}.pkl',
            pickle_module=dill
        )

        print(f'Now re-saving {se} torch files...')

        for f in tqdm(glob.glob(BASE_DIR + '/' + se + '*.pt'), total=len(glob.glob(BASE_DIR + '/' + se + '*.pt'))):
            datasets = torch.load(f)
            new_datasets = []

            for d in datasets:
                edge_weights = []
                src_sents = d['src_tokens']
                p_id = d['paper_id']

                for i, src_sent_1 in enumerate(src_sents):
                    for j, src_sent_2 in enumerate(src_sents):
                        if i == j:
                            edge_weights.append(0.0)

                        elif i > j:
                            edge_weight = paper_id_edges[p_id][j][i]
                            edge_weights.append(edge_weight)

                        elif i < j:
                            edge_weight = paper_id_edges[p_id][i][j]
                            edge_weights.append(edge_weight)

                d['edge_weights'] = edge_weights
                new_datasets.append(d)

            torch.save(new_datasets, f.replace('new', 'new-graph'))







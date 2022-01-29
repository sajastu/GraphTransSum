import argparse
import glob
import re
from multiprocessing import Pool
import torch
from tqdm import tqdm

from utils.rouge_score import evaluate_rouge
from utils.rouge_utils import _get_word_ngrams, cal_rouge

import scispacy
import spacy
nlp = spacy.load("en_core_sci_md")
nlp.disable_pipe("parser")
nlp.disable_pipe("ner")

parser = argparse.ArgumentParser()
parser.add_argument("-pt_dirs_src", default='')
parser.add_argument("-set", default='test')

args = parser.parse_args()


PT_DIRS = args.pt_dirs_src


def greedy_selection_score_based(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # import pdb;pdb.set_trace()
    max_score = 0.0

    selected = []


    for s in range(summary_size):
        cur_max_score = max_score
        cur_id = -1
        for i in range(len(doc_sent_list)):
            if (i in selected):
                continue
            candidate_str = ' '.join([doc_sent_list[idx] for idx in selected] + [doc_sent_list[i]])

            rouge_1, rouge_2, rouge_l = evaluate_rouge([candidate_str], [abstract_sent_list])
            score = (rouge_1 + rouge_l + rouge_2) / 3
            if score > cur_max_score:
                cur_max_score = score
                cur_id = i
        if (cur_id == -1):
            continue
        selected.append(cur_id)
        max_score = 0

    # now construct binary array
    sent_labels = [0] * len(doc_sent_list)

    for s in selected:
        sent_labels[s] = 1

    return sent_labels

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # import pdb;pdb.set_trace()
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    # abstract = _rouge_clean(' '.join(abstract)).split()
    # sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in doc_sent_list]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in doc_sent_list]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(doc_sent_list)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = (rouge_1 + rouge_2) / 2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            # return selected
            continue
        selected.append(cur_id)
        max_rouge = 0

    sent_labels = [0] * len(doc_sent_list)

    for s in selected:
        sent_labels[s] = 1

    return sent_labels



def _mp_change_oracle(params):
    sett, file = params
    datasets = torch.load(file)


    try:
        for j, d in tqdm(enumerate(datasets), total=len(datasets), desc=f'processing {file}'):
            src_sents = d['src_tokens']
            tgt_txt = d['tgt_tokens']
            sentence_labels = greedy_selection(src_sents, tgt_txt, summary_size=10)

            datasets[j]['src_sent_labels'] = sentence_labels
    except:
        print(file)
    torch.save(datasets, file)
    print(f'Saved ... {file}')

bert_files = []
for sett in ['test', 'val', 'train']:
    for f in glob.glob(f'{PT_DIRS}/{sett}*.pt'):
        bert_files.append((sett, f))
        # _mp_change_oracle(bert_files[-1])


pool = Pool(12)

for out in tqdm(pool.imap_unordered(_mp_change_oracle, bert_files), total=len(bert_files)):
    pass

pool.close()
pool.join()
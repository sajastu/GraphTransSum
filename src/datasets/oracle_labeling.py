import glob
import json
import operator
import os.path
import re
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from rouge_score_utils import rouge_scorer

BASE_DS_DIR = "/disk1/sajad/datasets/sci/arxivL/intro_summary/"
N_SENTS = 15

WR_DIR_NEW = f'{BASE_DS_DIR}/splits-normal2-{N_SENTS}/'
if not os.path.exists(WR_DIR_NEW):
    os.makedirs(WR_DIR_NEW)

def evaluate_rouge(hypotheses, references, type='f'):
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    results = {"rouge1_f": [], "rouge1_r": [], "rouge1_p": [], "rouge2_f": [],
               "rouge2_r": [], "rouge2_p": [], "rougeL_f": [], "rougeL_r": [], "rougeL_p": []}
    results_avg = {}

    if len(hypotheses) < len(references):
        print("Warning number of papers in submission file is smaller than ground truth file", file=sys.stderr)
    # import pdb;pdb.set_trace()
    hypotheses = list(hypotheses)
    references = list(references)
    for j, hyp in enumerate(hypotheses):
        submission_summary = hyp.replace('<q>', ' ')

        scores = scorer.score(references[j].strip(), submission_summary.strip())

        for metric in metrics:
            results[metric + "_f"].append(scores[metric].fmeasure)
            results[metric + "_r"].append(scores[metric].recall)
            results[metric + "_p"].append(scores[metric].precision)

        for rouge_metric, rouge_scores in results.items():
            results_avg[rouge_metric] = np.average(rouge_scores)

    return results_avg['rouge1_' + type], results_avg['rouge2_'+ type], results_avg['rougeL_'+ type]


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def _mp_write(ent):
    with open(ST_DIR + f'/{ent["id"]}.json', mode='w') as fW:
        json.dump(ent, fW)






def greedy_selection(doc_sent_list, abstract_sent_list, summary_size, sent_txt=None, gold_txt=None, paper_id=None, instance_number=None):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            binary_labels = [0] * len(doc_sent_list)
            for sel in selected:
                binary_labels[int(sel)] = 1
            return binary_labels
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    binary_labels = [0] * len(doc_sent_list)
    for sel in selected:
        binary_labels[int(sel)] = 1

    return binary_labels


def greedy_selection_fullfil(doc_sent_list, abstract_sent_list, summary_size, sent_txt=None, gold_txt=None, paper_id=None, instance_number=None):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])
    selected = []
    sentence_sorted_scores = {}
    summary_sent_number = 0
    cur_max_rouge = max_rouge
    trials = 1
    trials_from_scratch = 0
    selected_history = []
    break_less_than_15 = False
    while summary_sent_number < N_SENTS:
        cur_id = -1
        sentence_sorted_scores[summary_sent_number] = []
        for i in range(len(sents)):
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
            sentence_sorted_scores[summary_sent_number].append((i, rouge_score))

            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        try:
            if (cur_id == -1) and summary_sent_number < N_SENTS:
                selected_history.append((len(selected), cur_max_rouge, selected))
                # backward and try another sentence...
                sys.stdout.write("{}   \r".format(selected))
                sys.stdout.flush()
                prev1_SS = sorted(sentence_sorted_scores[summary_sent_number-1], key=operator.itemgetter(1), reverse=True)
                prev2_SS = sorted(sentence_sorted_scores[summary_sent_number-2], key=operator.itemgetter(1), reverse=True)

            # see if we add the new sentence either it improves the ROUGE
                if prev1_SS[trials][1] > prev2_SS[0][1]:
                    # if it improves the ROUGE, then it is the right path to pursue!
                    # substitute the specified index to the selected array
                    trials += 1
                    selected[-1] = sentence_sorted_scores[summary_sent_number-1][trials][0]
                    continue

                # if the tried ID from prev1_SS does not improve ROUGE...
                # then it means that we must also change -2 selected backward
                # --we should repeat the process until we find a different new selection that highers the ROUGE score
                else:
                    sentence_sorted_scores.pop(summary_sent_number)
                    selected = selected[:-1]
                    summary_sent_number -= 1
                    trials = 1
                    continue
        except:
            try:
                trials_from_scratch += 1
                selected = [sorted(sentence_sorted_scores[summary_sent_number - 1], key=operator.itemgetter(1), reverse=True)[trials_from_scratch][0]]
                cur_max_rouge = sorted(sentence_sorted_scores[summary_sent_number - 1], key=operator.itemgetter(1), reverse=True)[trials_from_scratch][1]
                continue
            except:
                selected= sorted(selected_history, key=lambda element: (element[0], element[1]), reverse=True)[0][2]
                break_less_than_15 = True
                break
                # selected = sorted(selected_history, key=operator.itemgetter(0), reverse=True)[1]
                # not 15 label summary can be created -- return the highest from selected history

        if break_less_than_15:
            break
        # if cur_id == -1:
        #     import pdb;pdb.set_trace()
        if cur_id != -1:
            selected.append(cur_id)
            summary_sent_number += 1

        if len(selected) == N_SENTS:
            break
    # if paper_id == "1004.4714":
    #     import pdb;pdb.set_trace()

    binary_labels = [0] * len(doc_sent_list)
    # print(f'{instance_number}. Paper_id: {paper_id} ==> Picked {len(selected)}: selected {selected} / cur_max_rouge: {cur_max_rouge}')
    # try:
    for sel in selected:
        binary_labels[int(sel)] = 1
    # except:
    #     print('hum')
    #     print(selected)
    #     os._exit(3)

    return binary_labels


def find_next_oracle_id(cur_id, cur_max_rouge, evaluated_1grams, evaluated_2grams, num_of_selected, reference_1grams,
                        reference_2grams, selected, sentence_sorted_scores, sents):
    for i in range(len(sents)):
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
        sentence_sorted_scores[num_of_selected].append((i, rouge_score))

        if rouge_score > cur_max_rouge:
            cur_max_rouge = rouge_score
            cur_id = i
    return cur_id



def _mp_label(file):
    file, inst_number = file
    paper = json.load(open(file))
    paper_sents = [s[0] for s in paper['sentences']]
    tgt_str = ' '.join([' '.join(t) for t in paper['gold']]).replace(' _ _', '')


    sent_bin_labels = greedy_selection(paper_sents, paper['gold'], summary_size=N_SENTS, sent_txt=[s[-7] for s in paper['sentences']], gold_txt=tgt_str, paper_id=paper['id'], instance_number=inst_number)

    if sum(sent_bin_labels) < 15:
        print('Normal labeling finds lt 15 sents...')
        sent_bin_labels = greedy_selection_fullfil(paper_sents, paper['gold'], summary_size=N_SENTS, sent_txt=[s[-7] for s in paper['sentences']], gold_txt=tgt_str, paper_id=paper['id'], instance_number=inst_number)

    if sum(sent_bin_labels) < 15:
        print(f'-- sent labels is now {sum(sent_bin_labels)} which is less than 15 -- ')
        with open('lt15_val.txt', mode='a') as fW:
            fW.write(f'{inst_number}\t{paper["id"]}\n')
    new_sents = []
    oracle_str = ''

    for j, sent in enumerate(paper['sentences']):
        sent[-6] = sent_bin_labels[j]
        if sent[-6] == 1:
            oracle_str += sent[-7]
            oracle_str += ' '
        new_sents.append(sent)
    r1, r2, rl = evaluate_rouge([oracle_str.strip()], [tgt_str.strip()])
    # if paper['id'] == "1004.4714":
    #     import pdb;pdb.set_trace()
    paper['sentences'] = new_sents

    with open(ST_DIR + f'/{paper["id"]}.json', mode='w') as fW:
        json.dump(paper, fW)

    return paper, {'r1': r1, 'r2': r2, 'rl': rl}


for st in ['val', 'test', 'train']:

    files = []

    for ins_number, f in enumerate(glob.glob("{}/splits-with-introRg-BertScore/{}/*.json".format(BASE_DS_DIR, st))):
        files.append((f, ins_number))
        _mp_label((f, ins_number))

    pool = Pool(10)
    ents = []

    ST_DIR = WR_DIR_NEW+ f'/{st}'
    if not os.path.exists(ST_DIR):
        os.makedirs(ST_DIR)

    rouge_scores = {'r1': [], 'r2':[], 'rl':[]}
    for out in tqdm(pool.imap_unordered(_mp_label, files), total=len(files)):
        ents.append(out[0])

        for m in ['r1', 'r2', 'rl']:
            rouge_scores[m].append(out[1][m])

    pool.close()
    pool.join()

    print(f'Oracle scores for {st} set with {N_SENTS}...')

    print(f'RG-1 {np.mean(rouge_scores["r1"])}')
    print(f'RG-2 {np.mean(rouge_scores["r2"])}')
    print(f'RG-L {np.mean(rouge_scores["rl"])}')


    # writing
    # pool_n = Pool(10)
    #
    # for _ in tqdm(pool_n.imap_unordered(_mp_write, ents), total=len(ents)):
    #     pass
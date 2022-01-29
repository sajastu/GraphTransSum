
"""

Check to see if we bring up the sentences which has the most rouge overlap with the intro summmary, how will the oracle be?

"""
import glob
import json
from multiprocessing import Pool

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
from datasets import load_metric


import sys


sys.path.append(r'/home/sajad/packages/summarization/introAbsGuided/src/')


bertscore = load_metric("bertscore")


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

def _mp_rouge(params):
    sents, intro_gold = params
    rg_sents = []
    for sent in sentences:
        # _, rg2, rgL = evaluate_rouge([sent], [intro_summ])
        # bertscore = metric.add(prediction=sent, reference=intro_summ)
        # scores = bertscore.compute(lang="en")

        results = bertscore.compute(predictions=[sent], references=[intro_summ], lang="en")
        rg_sents.append((results['f1'][0], sent))

    rg_sents_sorted = [s[1] for s in sorted(rg_sents, key=lambda x: x[0], reverse=True)]
    oralce_summary = ' '.join(rg_sents_sorted[:15])

    final_rouge = evaluate_rouge([oralce_summary], [gold])

    return final_rouge

for set in ['val', 'test']:

    inputs = []
    rg1_all = []
    rg2_all = []
    rgL_all = []

    for f in tqdm(glob.glob(f'/disk1/sajad/datasets/sci/arxivL/intro_summary/splits/{set}/*.json'), total=len(glob.glob(f'/disk1/sajad/datasets/sci/arxivL/intro_summary/splits/{set}/*.json'))):

        ent = json.load(open(f))
        gold = ' '.join([' '.join(s) for s in ent['gold']])
        sentences = [ ' '.join(s[0]) for s in ent['sentences']]
        intro_summ = ent['intro_summary']
        inputs.append((sentences, intro_summ))
        rg = _mp_rouge((sentences, intro_summ))
        rg1_all.append(rg[0])
        rg2_all.append(rg[1])
        rgL_all.append(rg[2])
    # pool = Pool(12)


    # for rg in tqdm(pool.imap_unordered(_mp_rouge, inputs), total=len(inputs)):
    #     rg1_all.append(rg[0])
    #     rg2_all.append(rg[1])
    #     rgL_all.append(rg[2])

    print(f'oracle for {set}: {np.mean(np.asarray(rg1_all))}  /  {np.mean(np.asarray(rg2_all))}  / {np.mean(np.asarray(rgL_all))}')




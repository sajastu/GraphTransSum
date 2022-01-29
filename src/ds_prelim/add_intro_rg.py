import glob
import json
import os.path
import re
from multiprocessing import Pool
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm


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


def _mp_intro_add(paper):
    new_sents = []
    for sent in paper['sentences']:
        try:
            r1, r2, rL = evaluate_rouge([sent[-3]], [re.sub(' +', ' ', paper['intro_summary'].replace('\n',''))])
            intro_rg = (r2 + rL) / 2
        except:
            # import pdb;pdb.set_trace()
            print(paper['id'])

        r1, r2, rL = evaluate_rouge([sent[-3]], [' '.join([' '.join(g) for g in paper['gold']])])
        gold_rg = (r2 + rL) / 2

        new_sents.append(sent + [intro_rg, gold_rg])
    paper['sentences'] = new_sents

    return paper




for set in ['train']:
    papers = []

    for f in glob.glob(f'/disk1/sajad/datasets/sci/arxivL/intro_summary/splits/{set}/*.json'):
        ent = json.load(open(f))
        papers.append(ent)
        # if 'astro-ph9806123' in f:
        #     _mp_intro_add(ent)

    new_papers = []
    pool = Pool(12)

    for out in tqdm(pool.imap_unordered(_mp_intro_add, papers), total=len(papers), desc=f'{set}'):
        new_papers.append(out)

    WR_PATH = f'/disk1/sajad/datasets/sci/arxivL/intro_summary/splits-with-introRg/{set}'
    if not os.path.exists(WR_PATH):
        os.makedirs(WR_PATH)


    for npaper in new_papers:
        with open(WR_PATH + f'/{npaper["id"]}.json', mode='w') as fW:
            json.dump(npaper, fW)

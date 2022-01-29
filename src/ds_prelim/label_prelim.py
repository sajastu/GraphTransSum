import glob
import json
import math
import pickle
import statistics
from collections import defaultdict
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy import trunc
from tqdm import tqdm
from rouge_score import rouge_scorer
from pylab import rcParams
from datasets import load_metric
from matplotlib.ticker import FormatStrFormatter
from bert_score import score

bertscore = load_metric("bertscore")

metrics = ['rouge1', 'rouge2', 'rougeL']
LABELS = [0,1]
global metric
# metric = [metrics[0],  metrics[1], metrics[2]]
metric = [metrics[1], metrics[2]]
# metric = ['bertscore']

def evaluate_rouge(hypotheses, references, type='f'):
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

def _mp_rg(params):
    id, sent, gold, intro_summary, label = params
    r1, r2, rL = evaluate_rouge([sent], [gold])
    scores_gold = {'rouge1': r1, 'rouge2': r2, 'rougeL': rL}
    r1, r2, rL = evaluate_rouge([sent], [intro_summary])
    scores_intro = {'rouge1': r1, 'rouge2': r2, 'rougeL': rL}
    returned_scores = []

    scores = [scores_gold, scores_intro]

    for score in scores:
        sc = 0
        for m in metric:
            sc = score[m]
        sc = sc / len(metric)
        returned_scores.append(sc)

    returned_scores.extend([id, label])
    return returned_scores

def _mp_rg_paper(paper):
    sentences = paper['sentences']
    labels = {
        '0': {
            'intro':[],
            'gold': []
        },
        '1':{
            'intro':[],
            'gold': []
        }
    }
    intro_all_max = max([s[-2] for s in sentences])
    gold_all_max = max([s[-1] for s in sentences])

    intro_all_min = min([s[-2] for s in sentences])
    gold_all_min = min([s[-1] for s in sentences])

    intro_mean = statistics.mean([s[-2] for s in sentences])
    gold_mean = statistics.mean([s[-1] for s in sentences])

    intro_std = statistics.stdev([s[-2] for s in sentences])
    gold_std = statistics.stdev([s[-1] for s in sentences])

    # for sent in sentences:
    #     labels[str(sent[-4])]['intro'].append(
    #         sent[-2] - intro_all_min / intro_all_max - intro_all_min
    #     )
    #     labels[str(sent[-4])]['gold'].append(
    #         sent[-1] - gold_all_min / gold_all_max - gold_all_min
    #     )

    oracle_from_intro = ''
    for sent in sentences:
        if sent[-3] != 0 or sent[-3] != 1:
            if sent[-4] == 1:
                oracle_from_intro += sent[-5]
                oracle_from_intro += ' '

    oracle_from_intro = oracle_from_intro.strip()




    # Z-score on labels
    for sent in sentences:
        if sent[-3] != 0 or sent[-3] != 1:
            labels[str(sent[-4])]['intro'].append(
                sent[-2]
            )
            labels[str(sent[-4])]['gold'].append(
                sent[-1]
            )

    # min-max normalization
    stat = {
        'min' : {'intro': 0, 'gold': 0},
        'max' : {'intro': 0, 'gold': 0},
        'mean': {'intro': 0, 'gold': 0},
        'std': {'intro': 0, 'gold': 0},
    }

    for type in ['intro', 'gold']:
        arrays = []
        for l in ['0', '1']:
            arrays.extend(labels[str(l)][type])
        min_all = min(arrays)
        max_all = max(arrays)
        mean_all = statistics.mean(arrays)
        std_all = statistics.stdev(arrays)
        stat['min'][type] = min_all
        stat['max'][type] = max_all
        stat['mean'][type] = mean_all
        stat['std'][type] = std_all

    new_labels = {
        '0': {
            'intro': [],
            'gold': []
        },
        '1': {
            'intro': [],
            'gold': []
        }
    }

    for type in ['intro', 'gold']:
        for l in ['0', '1']:
            arrays = labels[l][type]

            for a in arrays:
                new_labels[l][type].append(
                    (( (a - stat['mean'][type]) / stat['std'][type]) - stat['min'][type]) / ( (stat['max'][type] - stat['min'][type]) )
                )



    ###### ###### ###### labels 2-3 ###### ###### ######



    # tan h estimator
    # for sent in sentences:
    #     labels[str(sent[-4])]['intro'].append(
    #         0.5 * (math.tanh(0.01 * ((sent[-2] - intro_mean) / intro_std)) + 1.0 )
    #     )
    #     labels[str(sent[-4])]['gold'].append(
    #         0.5 * (math.tanh(0.01 * ((sent[-1] - gold_mean) / gold_std)) + 1.0 )
    #     )


    return new_labels


def _mp_rg_paper_section(paper):
    sentences = paper['sentences']
    labels = {
        '0':{
            '0': {
                'intro':[],
                'gold': []
            },
            '1':{
                'intro':[],
                'gold': []
            }
        },
        '1': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '2': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '3': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '4': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
    }
    intro_all_max = max([s[-2] for s in sentences])
    gold_all_max = max([s[-1] for s in sentences])

    intro_all_min = min([s[-2] for s in sentences])
    gold_all_min = min([s[-1] for s in sentences])

    intro_mean = statistics.mean([s[-2] for s in sentences])
    gold_mean = statistics.mean([s[-1] for s in sentences])

    intro_std = statistics.stdev([s[-2] for s in sentences])
    gold_std = statistics.stdev([s[-1] for s in sentences])

    # Z-score on labels
    for sent in sentences:
        if sent[-3] != 0 or sent[-3] != 1:

            labels[str(sent[-3])][str(sent[-4])]['intro'].append(
                sent[-2]
            )
            labels[str(sent[-3])][str(sent[-4])]['gold'].append(
                sent[-1]
            )

    # min-max normalization
    stat = {
        'min' : {'intro': 0, 'gold': 0},
        'max' : {'intro': 0, 'gold': 0},
        'mean': {'intro': 0, 'gold': 0},
        'std': {'intro': 0, 'gold': 0},
    }

    for type in ['intro', 'gold']:
        arrays = []
        for l in ['0', '1']:
            for section in ['0', '1', '2', '3', '4']:
                arrays.extend(labels[section][str(l)][type])
        min_all = min(arrays)
        max_all = max(arrays)
        mean_all = statistics.mean(arrays)
        std_all = statistics.stdev(arrays)
        stat['min'][type] = min_all
        stat['max'][type] = max_all
        stat['mean'][type] = mean_all
        stat['std'][type] = std_all

    new_labels = {
        '0':{
            '0': {
                'intro':[],
                'gold': []
            },
            '1':{
                'intro':[],
                'gold': []
            }
        },
        '1': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '2': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '3': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '4': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
    }

    for section in ['0','1', '2', '3', '4']:
        for type in ['intro', 'gold']:
            for l in ['0', '1']:
                arrays = labels[section][l][type]

                for a in arrays:
                    new_labels[section][l][type].append(
                        (( (a - stat['mean'][type]) / stat['std'][type]) - stat['min'][type]) / ( (stat['max'][type] - stat['min'][type]) )
                    )


    return new_labels


def _mp_bertscore(params):
    out = []
    sent, gold, intro_summary = params
    results = bertscore.compute(predictions=[sent], references=[gold], lang="en")
    out.append(results['f1'][0])

    results = bertscore.compute(predictions=[sent], references=[intro_summary], lang="en")
    out.append(results['f1'][0])


    return out


def _rg_scatter(gold_summary_x, intro_summary_y, filename='rg-mesh.pdf'):
    rcParams['figure.figsize'] = 40, 40
    # plt.yticks(np.arange(0.0, 1.01, 0.01))
    # plt.xticks(np.arange(0, 1, 0.01))
    # plt.xticks(np.arange(0.0, 1.01, 0.01).round(decimals=2))
    # min_starting = min(min(gold_summary_x), min(intro_summary_y))
    fig, ax = plt.subplots()

    plt.scatter(gold_summary_x, intro_summary_y, s=10)
    plt.xticks(np.arange(0.0, 1.0, 0.01))
    plt.yticks(np.arange(0.0, 1.0, 0.01))
    plt.xticks(rotation=90)  # Rotates X-Axis Ticks by 45-degrees

    plt.xlabel('Gold RG', labelpad=10)
    plt.ylabel('IntroSummary RG', labelpad=10)
    plt.savefig(filename, bbox_inches='tight')

def _rg_scatter_2(label_scores, filename='rg-mesh.pdf'):
    label0, label1 = label_scores['0'], label_scores['1']

    rcParams['figure.figsize'] = 40, 40
    # plt.yticks(np.arange(0.0, 1.01, 0.01))
    # plt.xticks(np.arange(0, 1, 0.01))
    # plt.xticks(np.arange(0.0, 1.01, 0.01).round(decimals=2))
    # min_starting = min(min(gold_summary_x), min(intro_summary_y))
    fig, ax = plt.subplots()


    plt.scatter(label0['gold'], label0['intro'], s=10)
    plt.scatter(label1['gold'], label1['intro'], s=10)
    plt.xticks(np.arange(min(label0['gold']), max(max(label0['gold']), max(label1['gold'])), 0.1))
    plt.yticks(np.arange(min(label1['intro']), max(max(label1['intro']), max(label0['intro'])), 0.1))
    plt.xticks(rotation=90)  # Rotates X-Axis Ticks by 45-degrees

    plt.xlabel('Gold RG', labelpad=10)
    plt.ylabel('IntroSummary RG', labelpad=10)



    plt.savefig(filename, bbox_inches='tight')

def _rg_scatter_2_section(label_scores, filename='rg-mesh.pdf'):
    # label0, label1 = label_scores['0'], label_scores['1']

    rcParams['figure.figsize'] = 40, 40
    # plt.yticks(np.arange(0.0, 1.01, 0.01))
    # plt.xticks(np.arange(0, 1, 0.01))
    # plt.xticks(np.arange(0.0, 1.01, 0.01).round(decimals=2))
    # min_starting = min(min(gold_summary_x), min(intro_summary_y))
    fig, ax = plt.subplots()
    cmaps = []
    for section, labels in label_scores.items():
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        plt.scatter(labels['0']['gold'], labels['0']['intro'], s=10)
        plt.scatter(labels['1']['gold'], labels['1']['intro'], s=10)



    # plt.xticks(np.arange(min(label0['gold']), max(max(label0['gold']), max(label1['gold'])), 0.1))
    # plt.yticks(np.arange(min(label1['intro']), max(max(label1['intro']), max(label0['intro'])), 0.1))
    plt.xticks(rotation=90)  # Rotates X-Axis Ticks by 45-degrees
    plt.legend()

    plt.xlabel('Gold RG', labelpad=10)
    plt.ylabel('IntroSummary RG', labelpad=10)



    plt.savefig(filename, bbox_inches='tight')




def _rg_mesh(gold_summary_x, intro_summary_y, filename='rg-mesh.pdf'):
    # generate 2 2d grids for the x & y bounds
    heatmap, intro_summary_y, gold_summary_x = np.histogram2d(intro_summary_y, gold_summary_x, bins=(200, 200))

    # y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 1))
    # z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    # z = z[:-1, :-1]
    # z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    z = heatmap
    # z = heatmap[:-1, :-1]
    z_min, z_max = np.abs(z).min(), np.abs(z).max()

    fig, ax = plt.subplots()

    # rcParams["axes.titlepad"] = 22

    def NonLinCdict(steps, hexcol_array):
        cdict = {'red': (), 'green': (), 'blue': ()}
        for s, hexcol in zip(steps, hexcol_array):
            rgb = matplotlib.colors.hex2color(hexcol)
            cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
            cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
            cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
        return cdict

    hc = ['#e5e5f2', '#b2b2d8', '#6666b2', '#19198c']
    thss = [0, 0.33, 0.66, 1]

    cdict = NonLinCdict(thss, hc)
    cm = LinearSegmentedColormap('test', cdict)

    c = ax.pcolormesh(gold_summary_x, intro_summary_y, z, cmap=cm, vmin=z_min, vmax=z_max)
    # ax.set_title('Oracle sentence\'s importance over the relative source position')
    # set the limits of the plot to the limits of the data
    ax.axis([gold_summary_x.min(), gold_summary_x.max(), intro_summary_y.min(), intro_summary_y.max()])
    fig.colorbar(c, ax=ax)

    if 'abstractiveness' in filename or 'novel' in filename:
        plt.xlabel('n', labelpad=6)
    else:
        plt.xlabel('Gold RG', labelpad=10)

    plt.ylabel('IntroSummary RG', labelpad=10)

    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    # subprocess.call(['gupload', 'rg-mesh.pdf'])



# for set in ["test", "val", "train"]:
for set in ["val"]:
    # for LABEL in [0, 1]:
    scores_labels = {
        '0': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '1': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '2': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '3': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
        '4': {
            '0': {
                'intro': [],
                'gold': []
            },
            '1': {
                'intro': [],
                'gold': []
            }
        },
    }
    sent_gold_introS = []
    sent_gold_rg = []
    sent_introS_rg = []
    labels = []
    lenn = len(glob.glob(f"/disk1/sajad/datasets/sci/arxivL/intro_summary/splits-with-introRg/{set}/*.json"))
    for f in glob.glob(f"/disk1/sajad/datasets/sci/arxivL/intro_summary/splits-with-introRg/{set}/*.json"):
        ent = json.load(open(f))
        # id = ent['paper_id']
        sent_gold_introS.append(ent)
        # _mp_rg_paper(ent)
            # for sent in ent['sentences']:
            #     sent_gold_introS.append((id, sent[-3], ' '.join([' '.join(t) for t in ent['gold']]), ent['intro_summary'], sent[-2]))


        # P_gold, R_gold, sent_gold_rg = score([s[0] for s in sent_gold_introS], [s[1] for s in sent_gold_introS], lang="en", verbose=True, batch_size=220)
        # P_intro, R_intro, sent_introS_rg = score([s[0] for s in sent_gold_introS], [s[2] for s in sent_gold_introS], lang="en", verbose=True, batch_size=220)

        # P_gold, R_gold, sent_gold_rg = score([s[0] for s in sent_gold_introS], [s[1] for s in sent_gold_introS], model_type="allenai/longformer-large-4096", verbose=True, num_layers=12, batch_size=20)
        # P_intro, R_intro, sent_introS_rg = score([s[0] for s in sent_gold_introS], [s[2] for s in sent_gold_introS],model_type="allenai/longformer-large-4096", verbose=True, num_layers=12, batch_size=20)
    pool = Pool(12)


    # for out in tqdm(pool.imap_unordered(_mp_rg, sent_gold_introS), total=len(sent_gold_introS)):
    #     sent_gold_rg.append(out[0])
    #     sent_introS_rg.append(out[1])
    #     labels.append(out[-1])
    #     if out[-2] not in all_rg[str(out[-1])]:
    #         all_rg[str(out[-1])] = relative
        # print(out)

    # scores_labels[str(LABEL)]['gold'] = sent_gold_rg
    # scores_labels[str(LABEL)]['intro'] = sent_introS_rg

    for out in tqdm(pool.imap_unordered(_mp_rg_paper_section, sent_gold_introS), total=len(sent_gold_introS)):
        for section in ['0', '1', '2', '3', '4']:
            for label in ['0', '1']:
                for type in ['gold', 'intro']:
                    scores_labels[section][label][type].extend(out[section][label][type])


    print('pickling...')
    metric_str = '-'.join(['%s' % (m) for m in metric])

    with open(f'pkl-label{"0, 1"}-rel.pkl', mode='wb') as fW:
        pickle.dump(scores_labels, fW)


        # _rg_scatter(sent_gold_rg, sent_introS_rg, filename=f'{set}-label-{LABEL}-scatter-{metric_str}.pdf')
        # _rg_scatter(sent_gold_rg.cpu().numpy().tolist(), sent_introS_rg.cpu().numpy().tolist(), filename=f'{set}-label-{LABEL}-scatter-{metric_str}.pdf')
    metric_str = '-'.join(['%s' % (m) for m in metric])
    # _rg_scatter_2(scores_labels, filename=f'{set}-label-{"0, 1"}-scatter-{metric_str}.pdf')
    _rg_scatter_2_section(scores_labels, filename=f'{set}-label-{"0, 1"}-scatter-{metric_str}-section.pdf')
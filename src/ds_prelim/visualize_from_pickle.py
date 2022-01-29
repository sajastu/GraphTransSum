import pickle

import numpy as np
from pylab import rcParams
import matplotlib
import matplotlib.pyplot as plt

def _rg_scatter_2(label_scores, filename='rg-mesh.pdf'):
    label0, label1 = label_scores['0'], label_scores['1']

    rcParams['figure.figsize'] = 40, 40
    # plt.yticks(np.arange(0.0, 1.01, 0.01))
    # plt.xticks(np.arange(0, 1, 0.01))
    # plt.xticks(np.arange(0.0, 1.01, 0.01).round(decimals=2))
    # min_starting = min(min(gold_summary_x), min(intro_summary_y))
    plt.scatter(np.array(label0['gold'])-np.array(label0['intro']), label0['gold'], s=10)
    plt.scatter(np.array(label1['gold'])-np.array(label1['intro']), label1['gold'], s=10)
    plt.xticks(np.arange(-1, 1.0, 0.02))
    plt.yticks(np.arange(0, 1.0, 0.02))
    plt.xticks(rotation=90)  # Rotates X-Axis Ticks by 45-degrees

    plt.xlabel('Difference RG', labelpad=10)
    plt.ylabel('GOLD RG', labelpad=10)

    plt.savefig(filename, bbox_inches='tight')


LABELS = [0, 1]
scores_labels = {'0': {'gold': [], 'intro': []}, '1': {'gold': [], 'intro': []}}

for label in LABELS:
    with open(f'pkl-label{label}.pkl', mode='rb') as fR:
        scores = pickle.load(fR)
    scores_labels[str(label)]['gold'] = scores['gold_rg']
    scores_labels[str(label)]['intro'] = scores['into_rg']

_rg_scatter_2(scores_labels, filename='val_combined.pdf')


import glob
import json
import os.path
import random
import re
from multiprocessing import Pool
# from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
from bert_score import BERTScorer
from transformers import AutoTokenizer

from others.tokenization import BertTokenizer
import torch

scorer = BERTScorer(model_type='allenai/scibert_scivocab_cased', lang='en-sci', rescale_with_baseline=False)

# tokenizer = BertTokenizer.from_pretrained(, do_lower_case=True)
BSZ = 128

def get_tokenizer(model_type, use_fast=False):
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=use_fast)
    return tokenizer

tokenizer = get_tokenizer('allenai/scibert_scivocab_uncased')


def _mp_intro_add(paper):
    cands = [' '.join(s[0]) if len(tokenizer.encode(' '.join(s[0]))) < 450 else ' '.join(tokenizer.convert_ids_to_tokens([x for x in tokenizer.encode(' '.join(s[0]))[:450]])).replace(' ##', '') for s in paper['sentences']]

    # refs_intro = [re.sub(' +', ' ', paper['intro_summary'].replace('\n',''))] * len(cands)
    refs_gold = [' '.join([' '.join(g[0]) for g in paper['gold']])] * len(cands)
    try:
        P, R, F1_gold = scorer.score(cands, refs_gold, batch_size = BSZ)
        # P, R, F1_intro = scorer.score(cands, refs_intro, batch_size = BSZ)
        new_sents = [s + [ f1_gold.item()] for s, f1_gold in zip(paper['sentences'], F1_gold)]
        paper['sentences'] = new_sents
    except:
        try:

            if len(tokenizer.encode(refs_gold[0])) > 450:
                chunk_num = 0
                total_chunks = (len(tokenizer.encode(refs_gold[0])) // 450) + 1 if len(tokenizer.encode(refs_gold[0])) % 450 !=0 else (len(tokenizer.encode(refs_gold[0])) // 450)
                F1_gold_avg = torch.zeros(len(cands))
                while chunk_num < total_chunks:
                    try:
                        chunK_text = ' '.join(tokenizer.convert_ids_to_tokens(tokenizer.encode(refs_gold[0])[chunk_num*450:(chunk_num+1)*450])).replace(' ##', '')
                    except:
                        chunK_text = ' '.join(tokenizer.convert_ids_to_tokens(tokenizer.encode(refs_gold[0])[chunk_num*450:])).replace(' ##', '')

                    # refs_intro = [re.sub(' +', ' ', paper['intro_summary'].replace('\n', ''))] * len(cands)
                    refs_gold_chunk = [chunK_text] * len(cands)


                    P, R, F1_gold = scorer.score(cands, refs_gold_chunk, batch_size = BSZ)
                    F1_gold_avg += F1_gold
                    chunk_num+=1
                    # P, R, F1_intro = scorer.score(cands, refs_intro)

                F1_gold = F1_gold_avg / torch.Tensor([chunk_num])
                # P, R, F1_intro = scorer.score(cands, refs_intro, batch_size = BSZ)

                new_sents = [s + [f1_gold.item()] for s, f1_gold in zip(paper['sentences'], F1_gold)]
                paper['sentences'] = new_sents
                paper['id'] = paper['filename']
                del paper['filename']


                return paper, True
        except Exception as e:

            return {
                'paper_id': paper['id'],
                'len_gold': len(tokenizer.tokenize(refs_gold[0])),
                'error': str(e)
            }, False

    paper['id'] = paper['filename']
    del paper['filename']
    return paper, True




for set in ['val']:
    papers = []
    new_papers = []
    count = 0
    last_saved_file = {}
    WR_PATH = f'/disk1/sajad/datasets/sci/pubmedL/my-format-splits-BertScore/{set}'
    if not os.path.exists(WR_PATH):
        os.makedirs(WR_PATH)

    parsed_files =[f.replace('-BertScore','') for f in glob.glob(f'/disk1/sajad/datasets/sci/pubmedL/my-format-splits-BertScore/{set}/*.json')]
    remaining_files = [f for f in glob.glob(f'/disk1/sajad/datasets/sci/pubmedL/my-format-splits/{set}/*.json') if f not in parsed_files]

    # remaining_dict = json.load(open(f'bertScore_not_taken_{set}_r2.json', mode='r'))
    # remaining_files = [f'/disk1/sajad/datasets/sci/arxivL/intro_summary/splits-with-introRg/{set}/' + l + '.json' for l in list(remaining_dict.keys())]


    # files = glob.glob(f'/disk1/sajad/datasets/sci/arxivL/intro_summary/splits-with-introRg/{set}/*.json')
    files = remaining_files

    random.seed(8888)
    random.shuffle(files)

    for f in tqdm(files, total=len(files)):
        ent = json.load(open(f))
        papers.append(ent)
        out = _mp_intro_add(ent)
        if out[1]:
            new_papers.append(out[0])
            if not os.path.exists(WR_PATH + f'/{out[0]["id"]}.json'):
                with open(WR_PATH + f'/{out[0]["id"]}.json', mode='w') as fW:
                    json.dump(out[0], fW)
        else:
            count +=1
            wrN = open(f'bertScore_not_taken_{set}_r0_pL.json', mode='w')
            last_saved_file[out[0]['paper_id']] = out[0]
            json.dump(last_saved_file, wrN, indent=4)

    print(f'{count} papers not scored...')
    # pool = Pool(6)
    # for out in tqdm(pool.imap_unordered(_mp_intro_add, papers), total=len(papers), desc=f'{set}'):
    #     if out[1]:
    #         new_papers.append(out)
    #     else:
    #         import pdb;pdb.set_trace()
    #         if os.path.exists('bertScore_not_taken.json'):
    #             last_saved_file = json.load(open('bertScore_not_taken.json'))
    #             last_saved_file[out[0]['paper_id']] = out[0]
    #             json.dump(last_saved_file, wrN, indent=4)
    #         else:
    #             wrN = open('bertScore_not_taken.json', mode='w')
    #             last_saved_file = {}
    #             last_saved_file[out[0]['paper_id']] = out[0]
    #             json.dump(last_saved_file, wrN, indent=4)


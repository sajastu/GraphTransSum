import gc
import glob
import json
import os
from multiprocessing import Pool

import torch
from tqdm import tqdm

from others.logging import logger
from others.tokenization import BertTokenizer
from prepro.FIXED_KEYS import *
from prepro.data_builder import check_path_existence
from os.path import join as pjoin


class BertDataOriginal():
    def __init__(self, args):
        self.args = args

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src_sents, tgt, sent_labels, src_tokens=None, use_bert_basic_tokenizer=True, is_test=False):

        if ((not is_test) and len(src_sents) == 0):
            return None
        original_src_txt = src_sents

        idxs = [i for i, s in enumerate(src_sents) if (len(s.split()) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = sent_labels
        sent_labels = [_sent_labels[i] for i in idxs]

        src_sents = src_sents[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src_sents) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src_sents]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        # segment embeddings
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]


        # clss-ids
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] ' + ' '.join(self.tokenizer.tokenize(tgt)) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = tgt
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert_simple(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']

    check_path_existence(args.save_path)
    for corpus_type in datasets:
        a_lst = []
        c = 0
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            c += 1
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('.json', '.bert.pt')),  1))
        print("Number of files: " + str(c))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        # for a in a_lst:
        #     _format_to_bert_original(a)

        # single
        # json_f = args.raw_path + '/test.0.json'
        # _format_to_bert(('test', str(json_f), args, pjoin(args.save_path, str(json_f).replace('.json', '.bert.pt')), bart,
        #          sent_numbers, 40))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        pool = Pool(args.n_cpus)
        print('Processing {} set with {} json files...'.format(corpus_type, len(a_lst)))
        for _ in tqdm(pool.imap(_format_to_bert_original, a_lst), total=len(a_lst), desc=''):
            pass

        pool.close()
        pool.join()

def _format_to_bert_original(params):
    corpus_type, json_file, args, save_file, debug_idx = params

    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertDataOriginal(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        id, src_sents, src_labels, tgt = d[ID_KEY], d[SRC_SENTS_KEY], d[SRC_SENTS_LABELS], d[GOLD_KEY]

        b_data = bert.preprocess(src_sents, tgt, src_labels, use_bert_basic_tokenizer=True, is_test=is_test)

        if (b_data is None):
            print('rrrrrr')
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
        #                "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
        #                'src_txt': src_txt, "tgt_txt": tgt_txt}
        b_data_dict = {
            "src": src_subtoken_idxs,
            "tgt": tgt_subtoken_idxs,
            "src_sent_labels": sent_labels.copy(),
            "segs": segments_ids,
            'clss': cls_ids,
            'src_txt': src_txt,
            "tgt_txt": tgt_txt,
            "paper_id": d['id'],
       }

        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

import gc
import glob
import hashlib
import json
import os
import os.path
import pickle
import re
import statistics
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
from multiprocess import Pool
from tqdm import tqdm

from others.logging import logger
from others.tokenization import BertTokenizer, LongformerTokenizerMine
from prepro.utils import _get_word_ngrams, get_negative_samples, get_positive_samples_form_abstracts, remove_ack
from utils.graph_utils import Node, Graph
from utils.rouge_score import evaluate_rouge
from datetime import datetime
from uuid import uuid4
import pdb

from utils.rouge_utils import cal_rouge
import scispacy
import spacy

nlp = spacy.load("en_core_sci_md")
stopwords = nlp.Defaults.stop_words


nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]

INTRO_KWs_STR = "[introduction, introduction and motivation, motivation, motivations, basics and motivations, introduction., [sec:intro]introduction, *introduction*, i. introduction, [sec:level1]introduction, introduction and motivation, introduction[sec:intro], [intro]introduction, introduction and main results, introduction[intro], introduction and summary, [sec:introduction]introduction, overview, 1. introduction, [sec:intro] introduction, introduction[sec:introduction], introduction and results, introduction and background, [sec1]introduction, [introduction]introduction, introduction and statement of results, introduction[introduction], introduction and overview, introduction:, [intro] introduction, [sec:1]introduction, introduction and main result, introduction[sec1], [sec:level1] introduction, motivations, outline, introductory remarks, [sec1] introduction, introduction and motivations, 1.introduction, introduction and definitions, introduction and notation, introduction and statement of the results, i.introduction, introduction[s1], [sec:level1]introduction +,  introduction., introduction[s:intro], [i]introduction, [sec:int]introduction, introduction and observations, [introduction] introduction, [sec:1] introduction, **introduction**, [seci]introduction,, **introduction**, [seci]introduction, introduction and summary of results, introduction and outline, preliminary remarks, general introduction, [sec:intr]introduction, [s1]introduction, introduction[sec_intro], introduction and statement of main results, scientific motivation, [sec:sec1]introduction, *questions*, introduction and the model, intoduction, challenges, introduction[sec-intro], introduction and result, inroduction, [sec:intro]introduction +, introdution, 1 introduction, brief summary, motivation and introduction, [1]introduction, introduction and related work, [sec:one]introduction, [section1]introduction, [sect:intro]introduction]"

INTRO_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in INTRO_KWs_STR[1:-1].split(',')]) + "]"
INTRO_KWs = eval(INTRO_KWs_STR)[0]

CONC_KWs_STR = "[conclusion, conclusions, conclusion and future work, conclusions and future work, conclusion & future work, extensions, future work, related work and discussion, discussion and related work, conclusion and future directions, summary and future work, limitations and future work, future directions, conclusion and outlook, conclusions and future directions, conclusions and discussion, discussion and future directions, conclusions and discussions, conclusion and future direction, conclusions and future research, conclusion and future works, future plans, concluding remarks, conclusions and outlook, summary and outlook, final remarks, outlook, conclusion and outlook, conclusions and future work, summary and discussions, conclusion and future work, conclusions and perspectives, summary and concluding remarks, future work, conclusions., discussion and outlook, discussion & conclusions, open problems, remarks, conclusions[sec:conclusions], conclusion and perspectives, summary and future work, conclusion., summary & conclusions, closing remarks, final comments, future prospects, open questions, *conclusions*, [sec:conclusions]conclusions, conclusions and summary, comments, conclusion[sec:conclusion], perspectives, [sec:conclusion]conclusion, conclusions and future directions, summary & discussion, conclusions and remarks, conclusions and prospects, discussions and summary, future directions, conclusions and final remarks, the future, concluding comments, conclusions and open problems, summary[sec:summary], conclusions and future prospects, summary and remarks, conclusions and further work, conclusions[conclusions], [sec:summary]summary, comments and conclusions, summary and future prospects, [conclusion]conclusion, conclusion and remarks, concluding remark, further remarks, prospects, conclusion and open problems, conclusion and summary, v. conclusions, iv. conclusions,  summary and conclusions, summary and prospects, conclusions:, conclusion[conclusion], summary and final remarks, summary and future directions, summary & conclusion, [summary]summary, iv. conclusion, further questions, conclusion and future directions,  concluding remarks, further work, [conclusions]conclusions, outlook and conclusions, v. conclusion, *summary*, concluding remarks and open problems, conclusions and future works, future, [sec:conclusions] conclusions, [sec:concl]conclusions, remarks and conclusions, concluding remarks., conclusion and future works, summary., 4. conclusions, discussion and open problems, summary and comments, final remarks and conclusions, summary and conclusions., [sec:conc]conclusions, summary[summary], conclusions and open questions, [sec:conclusion]conclusions, further directions, conclusions and implications, conclusions & outlook, review, [sec:level1]conclusion, future developments, [sec:conc] conclusions, conclusions[sec:concl], conclusions and future perspectives, summary, conclusions and outlook, conclusions & discussion, [conclusions] conclusions, future research, concluding remarks and outlook, conclusions and future research, conclusion & outlook, discussion and future directions, conclusions[sec:conc], summary & outlook, vi. conclusions, future plans, [sec:summary] summary, conclusions and comments, conclusion and further work, conclusion and open questions, conclusions & future work, 5. conclusions, [sec:conclusion] conclusion, *concluding remarks*, iv. summary, conclusions[conc], conclusion:, [concl]conclusions, summary and perspective, conclusions[sec:conclusion], [sec:level1]conclusions, open issues, [sec:conc]conclusion, [sec:concl]conclusion, [sec:sum]summary, summary of the results, implications and conclusions, conclusions[conclusion], some remarks, conclusions[concl], conclusion and future research, conclusion remarks, vi. conclusion, perspective, conclusions and future developments, [conc]conclusion, general remarks, summary and conclusions[sec:summary], summary and open questions, 4. conclusion, conclusion and future prospects, concluding remarks and perspectives, remarks and questions, remarks and questions, [conclusion] conclusion, summary and implications, conclusive remarks, comments and conclusions, summary of conclusions, [conclusion]conclusions, conclusion and perspective, conclusion[sec:conc], [sec:summary]summary and conclusions, [sec:level1]summary, [sec:con]conclusion, [sec:level4]conclusion, conclusions and outlook., [summary]summary and conclusions, conclusion[sec:concl], 5. conclusion, [conc]conclusions, outlook and conclusion, remarks and conclusion,  summary and conclusion, conlusions, conclusion and final remarks, v. summary, future outlook, future improvements, summary and open problems, conclusion[concl], summary]"

CONC_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in CONC_KWs_STR[1:-1].split(',')]) + "]"
CONC_KWs = eval(CONC_KWs_STR)[0]

ABSTRACT_KWs_STR = "[abstract, 0 abstract]"
ABSTRACT_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in ABSTRACT_KWs_STR[1:-1].split(',')]) + "]"
ABS_KWs = eval(ABSTRACT_KWs_STR)[0]


# Tokenization classes

class LongformerData():
    def __init__(self, args=None):
        if args is not None:
            self.args = args
        self.CHUNK_LIMIT = 2046

        self.tokenizer = LongformerTokenizerMine.from_pretrained('longformer-based-uncased', do_lower_case=True)

        self.sep_token = '</s>'
        self.cls_token = '<s>'
        self.pad_token = '<pad>'
        self.tgt_bos = 'madeupword0000'
        self.tgt_eos = 'madeupword0001'
        self.tgt_sent_split = 'madeupword0002'

        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def cal_token_len(self, src):
        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [sent[0] for sent in src]
        src_txt_tmp = [[self.cls_token] + tokens + [self.sep_token] for tokens in src_txt]
        src_txt = []
        for s in src_txt_tmp:
            for ss in s:
                src_txt.append(ss)

        # text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(src_txt)

        return len(src_subtokens)

    def cal_token_len_prep(self, src):
        # idxs = [i for i, s in enumerate(src)]
        # src = [src[i] for i in idxs]
        src_txt = [sent[0] for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)


    def make_chunks(self, src_sents_tokens, sent_labels=None, sent_rg_scores=None, sent_rg_scores_intro=None, chunk_size=2046, paper_idx=0, intro_summary=None, gold_abstract=None):

        idxs = [i for i, s in enumerate(src_sents_tokens) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        src_sents_tokens = [src_sents_tokens[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        _sent_labels = [0] * len(src_sents_tokens)
        for l in sent_labels:
            _sent_labels[l] = 1
        sent_labels = [_sent_labels[i] for i in idxs]
        sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        sent_rg_intro = [sent_rg_scores_intro[i] for i in idxs]

        src_sents_tokens = src_sents_tokens[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]
        sent_rg_intro = sent_rg_intro[:self.args.max_src_nsents]

        src_txt = [sent[0] for sent in src_sents_tokens]
        src_txt_tmp = [[self.cls_token] + tokens + [self.sep_token] for tokens in src_txt]
        src_txt = []
        for s in src_txt_tmp:
            for ss in s:
                src_txt.append(ss)
        # text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(src_txt)

        # src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]
        sent_rg_intro = sent_rg_intro[:len(cls_ids)]

        out_sents_labels = []
        out_sents_rg_scores = []
        out_sents_rg_intro = []
        cur_len = 0
        out_src = []
        rg_score = 0
        j = 0

        last_chunk = False
        while j < len(cls_ids):
            if j == len(cls_ids) - 1:
                out_src1 = out_src.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_sents_rg_intro1 = out_sents_rg_intro.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_sents_rg_scores.clear()
                out_sents_rg_intro.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = True
                if len(out_src1) == 0:
                    j += 1
                    continue
                yield out_src1, out_sents_labels1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, out_sents_rg_intro1
            else:
                if cur_len < chunk_size:
                    out_src.append((src_sents_tokens[j][0], src_sents_tokens[j][1], src_sents_tokens[j][2]))
                    out_sents_labels.append(sent_labels[j])
                    out_sents_rg_scores.append(sent_rg_scores[j])
                    out_sents_rg_intro.append(sent_rg_intro[j])
                    if j != 0:
                        try:
                            cur_len += len(src_subtokens[cls_ids[j]:cls_ids[j+1]])
                        except:
                            print(j)
                            print(len(cls_ids))
                            os._exit(-1)
                    else:
                        cur_len += len(src_subtokens[cls_ids[0]: cls_ids[1]])
                    j += 1

                else:
                    cur_len -= len(src_subtokens[cls_ids[j - 1]:cls_ids[j]])
                    j = j - 1
                    out_src = out_src[:-1]

                    src_txt = [o[0] for o in out_src]
                    src_txt_tmp = [[self.cls_token] + tokens + [self.sep_token] for tokens in src_txt]
                    src_txt = []
                    for s in src_txt_tmp:
                        for ss in s:
                            src_txt.append(ss)


                    # src_subtokens = self.tokenizer.tokenize(src_txt)
                    # src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
                    # src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
                    out_sents_labels = out_sents_labels[:-1]
                    out_sents_rg_scores = out_sents_rg_scores[:-1]
                    out_sents_rg_intro = out_sents_rg_intro[:-1]
                    out_src1 = out_src.copy()
                    out_sents_labels1 = out_sents_labels.copy()
                    out_sents_rg_scores1 = out_sents_rg_scores.copy()
                    out_sents_rg_intro1 = out_sents_rg_intro.copy()
                    out_src.clear()
                    out_sents_labels.clear()
                    cur_len1 = cur_len
                    cur_len = 0
                    last_chunk = False
                    if len(out_src1) == 0:
                        j += 1
                        continue


                    yield out_src1, out_sents_labels1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, out_sents_rg_intro1


    def normal_tokenize_new(self, src_tokens, tgt_tokens):
        tmp_lemma_tokens = []
        tmp_lemma_tokens_tgt = []
        new_src_tokens = []
        for idx_sent, sent in enumerate(src_tokens):
            tmp_lemma_tokens.append([])
            new_src_tokens.append([])
            sent_txt = ' '.join(sent)
            sent_tokens = nlp(sent_txt)
            for idx_token, token in enumerate(sent_tokens):
                tmp_lemma_tokens[idx_sent].append(token.lemma_)
                new_src_tokens[idx_sent].append(token.text)

        for idx_sent, sent in enumerate(tgt_tokens):
            tmp_lemma_tokens_tgt.append([])
            sent_txt = ' '.join(sent)
            sent_tokens = nlp(sent_txt)
            for idx_token, token in enumerate(sent_tokens):
                tmp_lemma_tokens_tgt[idx_sent].append(token.lemma_)

        return new_src_tokens, tmp_lemma_tokens, tmp_lemma_tokens_tgt

    def _annotate(self, src_tokens, src_lemmas, tgt_tokens, tgt_lemmas, src_bert_nodes_len, tgt_bert_nodes_len, id, include_tgt=False, is_test=False):

        """

        :param src_tokens: tokens of source
        :param tgt_tokens: tokens of target
        :param src_lemmas: lemmas of source words
        :param tgt_lemmas: lemmas of target words
        :param src_bert_nodes_len: size of bert embeddings of source (will be used to create the graph nodes)
        :param tgt_bert_nodes_len: size of bert embeddings of tokens (will be used to create the graph nodes)
        :param id: paper_id for debugging
        :return: a graph from Graph() class

        IMPORTANT NOTE: should check if graph's edge doesn't violate node dimensionality...
        """

        def _find_common_words(s_token_idx, query, search_in, idx_sent, idx_sent_end):
            commonalities = []
            idx_sent_query = idx_sent
            idx_sent_search = idx_sent + 1
            for sent in search_in:
                for idx, token in enumerate(sent):
                    if query in stopwords:
                        return []
                    if query == token and len(query) > 2:
                        commonalities.append(
                            Node(idx_sent_query, s_token_idx, idx_sent_search, idx, query, token)
                        )

                idx_sent_search += 1

            return commonalities

        graph = Graph()
        src_lemma_tokens = src_lemmas


        if include_tgt and not is_test:
            tgt_subgraph = Graph()

        """""""""""""""""""""""""""""""""""
        1. connect intra-sentence tokens
        """""""""""""""""""""""""""""""""""
        # add to src subgraph
        for idx_sent, sent in enumerate(src_lemma_tokens):
            for idx_token, token in enumerate(sent):
                common_node = _find_common_words(idx_token, token, src_lemma_tokens[idx_sent+1: ], idx_sent, len(src_lemma_tokens))
                if len(common_node) > 0:
                    graph.add_edge_batch(common_node)


        if include_tgt and not is_test:
            # add to tgt subgraph
            for idx_sent, sent in enumerate(tgt_lemmas):
                for idx_token, token in enumerate(sent):
                    common_node = _find_common_words(idx_token, token, tgt_lemmas[idx_sent+1: ], idx_sent, len(tgt_lemmas))
                    if len(common_node) > 0:
                        tgt_subgraph.add_edge_batch(common_node)

        """""""""""""""""""""""""""""""""""
        2. connect tokens to the associated sentence
        """""""""""""""""""""""""""""""""""
        # add to src subgraph
        for idx_sent, sent in enumerate(src_tokens):
            for idx_token, token in enumerate(sent):
                graph.add_edge(Node(idx_sent, idx_token, s_node_txt=token, e_node_txt=' '.join(sent), s_node_lemma=src_lemma_tokens[idx_sent][idx_token]))

        if include_tgt and not is_test:
            # add to tgt subgraph
            for idx_sent, sent in enumerate(tgt_tokens):
                for idx_token, token in enumerate(sent):
                    tgt_subgraph.add_edge(Node(idx_sent, idx_token, s_node_txt=token, e_node_txt=' '.join(sent), s_node_lemma=tgt_lemmas[idx_sent][idx_token]))


        """""""""""""""""""""""""""""""""""
        3. Tokenizing using LM tokenizer
        """""""""""""""""""""""""""""""""""

        src_subtokens = []
        src_subtokens_flat = []
        lm_src_sent_len = []
        for idx_sent, sent in enumerate(src_tokens):
            sent_subtokens = [['<s>']] + self.tokenizer.tokenize_2d(sent) + [['</s>']]
            src_subtokens.append(sent_subtokens)
            for subtokens in sent_subtokens:
                src_subtokens_flat.extend(subtokens)
            lm_src_sent_len.append(sum([len(t) for t in sent_subtokens]))

        if include_tgt and not is_test:
            tgt_subtokens = []
            tgt_subtokens_flat = []
            lm_tgt_sent_len = []
            for idx_sent, sent in enumerate(tgt_tokens):
                sent_subtokens = [['<s>']] + self.tokenizer.tokenize_2d(sent) + [['</s>']]
                tgt_subtokens.append(sent_subtokens)
                for subtokens in sent_subtokens:
                    tgt_subtokens_flat.extend(subtokens)
                lm_tgt_sent_len.append(sum([len(t) for t in sent_subtokens]))

        """""""""""""""""""""""""""""""""""
        4. Replacing Graph IDs according to LM tokens
        """""""""""""""""""""""""""""""""""
        # replacing source sub-graph
        for node in graph:
            try:
                # sent idxs
                ss_idx = node.ss_idx # start
                es_idx = node.es_idx # end

                # token idxs
                st_idx = node.st_idx + 1 # start  # + 1 b/c of <s>
                et_idx = node.et_idx + 1 if node.et_idx is not None else None # end

                # retrieve associated tokens from LM
                lm_tokens_start = src_subtokens[ss_idx][st_idx]

                start_token_idx = (
                    (
                            ( ( (sum(lm_src_sent_len[:ss_idx]))) + ( sum([len(t) for t in src_subtokens[ss_idx][:st_idx]])) )
                    )
                )


                lm_tokens_end = []
                if et_idx is not None:
                    lm_tokens_end = src_subtokens[es_idx][et_idx]
                    end_token_idx = (((sum(lm_src_sent_len[:es_idx]))) + ( (sum([len(t) for t in src_subtokens[es_idx][:et_idx]])) ))

                # src_subtokens_flat

                if len(lm_tokens_start) == 1:
                    node.st_idx = (start_token_idx, )

                elif len(lm_tokens_start) > 1:

                    populater = len(lm_tokens_start)-1
                    tuple = (start_token_idx, )

                    # print(src_subtokens_flat[tuple[0]])
                    while populater != 0:
                        tuple = tuple + (tuple[-1] + 1, )
                        populater -= 1
                    node.st_idx = tuple


                if len(lm_tokens_end) == 1:
                    # print(node)
                    node.et_idx = (end_token_idx, )


                elif len(lm_tokens_end) > 1:
                    populater = len(lm_tokens_end) - 1
                    tuple = (end_token_idx,)
                    while populater != 0:
                        tuple = tuple + (tuple[-1] + 1,)
                        populater -= 1
                    node.et_idx = tuple



                if len(lm_tokens_end) == 0:
                    node.ss_idx = ((sum(lm_src_sent_len[:ss_idx])),)

            except Exception as e:
                print(e)
                print('error')
                os._exit(-1)
                # import pdb;pdb.set_trace()
        graph.set_nodes_and_edges(src_bert_nodes_len)

        if include_tgt and not is_test:
            # replacing target sub-graph
            for node in tgt_subgraph:
                try:
                    # sent idxs
                    ss_idx = node.ss_idx # start
                    es_idx = node.es_idx # end

                    # token idxs
                    st_idx = node.st_idx + 1 # start  # + 1 b/c of <s>
                    et_idx = node.et_idx + 1 if node.et_idx is not None else None # end

                    # retrieve associated tokens from LM
                    lm_tokens_start = tgt_subtokens[ss_idx][st_idx]

                    start_token_idx = (
                        (
                                ( ( (sum(lm_tgt_sent_len[:ss_idx]))) + ( sum([len(t) for t in tgt_subtokens[ss_idx][:st_idx]])) )
                        )
                    )


                    lm_tokens_end = []
                    if et_idx is not None:
                        lm_tokens_end = tgt_subtokens[es_idx][et_idx]
                        end_token_idx = (((sum(lm_tgt_sent_len[:es_idx]))) + ( (sum([len(t) for t in tgt_subtokens[es_idx][:et_idx]])) ))

                    # src_subtokens_flat

                    if len(lm_tokens_start) == 1:
                        node.st_idx = (start_token_idx, )

                    elif len(lm_tokens_start) > 1:

                        populater = len(lm_tokens_start)-1
                        tuple = (start_token_idx, )

                        # print(src_subtokens_flat[tuple[0]])
                        while populater != 0:
                            tuple = tuple + (tuple[-1] + 1, )
                            populater -= 1
                        node.st_idx = tuple


                    if len(lm_tokens_end) == 1:
                        # print(node)
                        node.et_idx = (end_token_idx, )


                    elif len(lm_tokens_end) > 1:
                        populater = len(lm_tokens_end) - 1
                        tuple = (end_token_idx,)
                        while populater != 0:
                            tuple = tuple + (tuple[-1] + 1,)
                            populater -= 1
                        node.et_idx = tuple



                    if len(lm_tokens_end) == 0:
                        node.ss_idx = ((sum(lm_src_sent_len[:ss_idx])),)

                except Exception as e:
                    print(e)
                    print('error')
                    os._exit(-1)
                    # import pdb;pdb.set_trace()
            tgt_subgraph.set_nodes_and_edges(tgt_bert_nodes_len)
            graph.add_connections_with_tgt(tgt_subgraph)

        return graph


    def _construct_graph(self, src_tokens, src_lemmas, tgt_tokens, tgt_lemmas, src_bert_nodes_len, tgt_bert_nodes_len, id, is_test=False):
        return self._annotate(src_tokens, src_lemmas, tgt_tokens, tgt_lemmas, src_bert_nodes_len, tgt_bert_nodes_len, id, include_tgt=True, is_test=False)


    def preprocess_single(self, src, tgt, sent_rg_scores=None, sent_rg_scores_intro=None, sent_labels=None,
                          use_bert_basic_tokenizer=False, is_test=False, intro_summary=None, low_sents=None, positive_samples=None, id=None):

        neg_sents_cls_indxes, pos_sents_cls_indxes, neg_subtokens_idxs, pos_sents_subtokens_idxs = None, None, None, None
        # low_sents = [l[1] for l in low_sents]

        ##### LOW SENTS --FROM THE PAPER ITSELF
        if low_sents is not None:
            low_sents = [l[1] for l in low_sents]


        if ((not is_test) and len(src) == 0):
            return None


        # find eligible indices to keep!!
        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]


        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [sent_labels[i] for i in idxs]

        original_src_txt_str = [' '.join(s[0]) for s in src]

        src_txt_tokens, src_lemmas, tgt_lemmas = self.normal_tokenize_new([s[0] for s in src], tgt)

        src = [(src_txt_tokens[j], ) + tuple(s[1:]) for j, s in enumerate(src)]

        _sent_labels = [0] * len(src)

        if low_sents is not None:
            idxs_positive_samples = [i for i, s in enumerate(positive_samples) if (len(re.sub(' +', ' ', re.sub(r'[^\w]', ' ', ' '.join(s))).split()) > 4)]
            idxs_low_samples = [i for i, s in enumerate(low_sents) if (len(re.sub(' +', ' ', re.sub(r'[^\w]', ' ', ' '.join(s))).split()) > 4)]

        try:

            ##### LOW SENTS --- FROM PAPER ITSELF
            if low_sents is not None:
                low_sentences = [low_sents[i][:self.args.max_src_ntokens_per_sent] for i in idxs_low_samples]
                positive_sentences = [positive_samples[i][:self.args.max_src_ntokens_per_sent] for i in idxs_positive_samples]

            if sent_rg_scores is not None:
                sent_rg_scores = [sent_rg_scores[i] for i in idxs]

            if sent_rg_scores_intro is not None:
                sent_rg_scores_intro = [sent_rg_scores_intro[i] for i in idxs]

            if sent_rg_scores is not None:
                sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

            if sent_rg_scores_intro is not None:
                sent_rg_scores_intro = sent_rg_scores_intro[:self.args.max_src_nsents]

            if ((not is_test) and len(src) < self.args.min_src_nsents):
                return None


            src_sent_token_count = [self.cal_token_len([(sent[0], 'test')]) for sent in src]

            src_sents_number = [sent[2] for sent in src]

            src_txt = [sent[0] for sent in src]
            src_txt_tmp = [[self.cls_token] + tokens + [self.sep_token] for tokens in src_txt]
            src_txt = []
            for s in src_txt_tmp:
                for ss in s:
                    src_txt.append(ss)
            src_subtokens = self.tokenizer.tokenize(src_txt)
            src_subtoken_ids = self.tokenizer.convert_tokens_to_ids(src_subtokens)

            ##### LOW SENTS --- FROM PAPER ITSELF
            if low_sents is not None:
                low_sentences_txt = [' '.join(sent).replace(' _','')  for sent in low_sentences]
                positive_sentences_txt = [' '.join(sent).replace(' _','')  for sent in positive_sentences]
                neg_txt = ' {} {} '.format(self.sep_token, self.cls_token).join(low_sentences_txt)
                # low_txt = [' {} {} '.format(self.sep_token, self.cls_token).join(low_sent_txt) for low_sent_txt in low_sents_txt]
                pos_txt = ' {} {} '.format(self.sep_token, self.cls_token).join(positive_sentences_txt)
                pos_sents_subtokens = self.tokenizer.tokenize(pos_txt)
                pos_sents_subtokens = [self.cls_token] + pos_sents_subtokens + [self.sep_token]
                ##### LOW SENTS --- FROM PAPER ITSELF
                neg_sents_subtokens = self.tokenizer.tokenize(neg_txt)
                neg_sents_subtokens = [self.cls_token] + neg_sents_subtokens + [self.sep_token]
                pos_sents_subtokens_idxs = self.tokenizer.convert_tokens_to_ids(pos_sents_subtokens)
                neg_subtokens_idxs = self.tokenizer.convert_tokens_to_ids(neg_sents_subtokens)
                pos_sents_cls_indxes = [i for i, t in enumerate(pos_sents_subtokens_idxs) if t == self.cls_vid]
                neg_sents_cls_indxes = [i for i, t in enumerate(neg_subtokens_idxs) if t == self.cls_vid]

                # low_sents_subtokens = [[self.cls_token] + self.tokenizer.tokenize(low_t) + [self.sep_token] for low_t in low_txt]

            intro_summary_subtokens = self.tokenizer.tokenize(intro_summary.split())
            intro_summary_subtokens = [self.cls_token] + intro_summary_subtokens + [self.sep_token]
            t_txt = [' '.join(sent).replace(' _','') for sent in tgt]
            tgt_txt = ' {} {} '.format(self.sep_token, self.cls_token).join(t_txt)
            tgt_subtokens = self.tokenizer.tokenize(tgt_txt.split())
            tgt_subtokens = [self.cls_token] + tgt_subtokens + [self.sep_token]

            tgt_subtokens_ids = self.tokenizer.convert_tokens_to_ids(tgt_subtokens)
            intro_summary_subtokens_ids = self.tokenizer.convert_tokens_to_ids(intro_summary_subtokens)

            _segs = [-1] + [i for i, t in enumerate(src_subtoken_ids) if t == self.sep_vid]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
            segments_ids = []

            for i, s in enumerate(segs):
                if (i % 2 == 0):
                    segments_ids += s * [0]
                else:
                    segments_ids += s * [1]

            _segs_intro_summary = [-1] + [i for i, t in enumerate(intro_summary_subtokens_ids) if t == self.sep_vid]
            segs_intro_summary = [_segs_intro_summary[i] - _segs_intro_summary[i - 1] for i in
                                  range(1, len(_segs_intro_summary))]
            segments_ids_intro = []
            for i, s in enumerate(segs_intro_summary):
                if (i % 2 == 0):
                    segments_ids_intro += s * [0]
                else:
                    segments_ids_intro += s * [1]

            _segs_tgt = [-1] + [i for i, t in enumerate(tgt_subtokens_ids) if t == self.sep_vid]
            segs_tgt = [_segs_tgt[i] - _segs_tgt[i - 1] for i in
                                  range(1, len(_segs_tgt))]
            segments_ids_tgt = []
            for i, s in enumerate(segs_tgt):
                if (i % 2 == 0):
                    segments_ids_tgt += s * [0]
                else:
                    segments_ids_tgt += s * [1]

            cls_indxes = [i for i, t in enumerate(src_subtoken_ids) if t == self.cls_vid]
            # low_sents_cls_indxes = [[i for i, t in enumerate(low_sents_subtokens_idx) if t == self.cls_vid] for low_sents_subtokens_idx  in low_sents_subtokens_idxs]

            tgt_cls_indxes = [i for i, t in enumerate(tgt_subtokens_ids) if t == self.cls_vid]
            sent_labels = sent_labels[:len(cls_indxes)]


            if sent_rg_scores is not None:
                sent_rg_scores = sent_rg_scores[:len(cls_indxes)]

            if sent_rg_scores_intro is not None:
                sent_rg_scores_intro = sent_rg_scores_intro[:len(cls_indxes)]

            # tgt_subtokens_str = 'madeupword0000 ' + ' madeupword0002 '.join(
            #     [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
            #      in tgt]) + ' madeupword0001'
            # tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

            if ((not is_test) and len(tgt_subtokens) < self.args.min_tgt_ntokens):
                return None

            # tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
            tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
            tgt_tokens = tgt
            src_txt_tokens = [s[0] for s in src]

            # if debug:
            # import pdb;pdb.set_trace()
        except Exception as e:
            print(e)

        ## constructing graph
        graph = self._construct_graph(src_txt_tokens, src_lemmas, tgt_tokens, tgt_lemmas, len(src_subtoken_ids), len(tgt_subtokens_ids), id, is_test)


        # import pdb;pdb.set_trace()

        return src_subtoken_ids, intro_summary_subtokens_ids, tgt_subtokens_ids, neg_subtokens_idxs, pos_sents_subtokens_idxs, sent_rg_scores, sent_labels, segments_ids,\
               segments_ids_intro, segments_ids_tgt, cls_indxes, neg_sents_cls_indxes, pos_sents_cls_indxes, tgt_cls_indxes, original_src_txt_str, tgt_txt, src_sents_number, src_sent_token_count, \
               sent_rg_scores_intro, src_txt_tokens, tgt_tokens, graph

# bert-function
def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'val', 'test']


    if len(args.sent_numbers_file) > 0:
        sent_numbers = pickle.load(open(args.sent_numbers_file, "rb"))
    else:
        sent_numbers = None

    abstracts = None
    ref_abstarcts = None

    bart = args.bart
    check_path_existence(args.save_path)
    a_lst = []
    for corpus_type in datasets:
        c = 0
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            c += 1
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('.json', '.bert.pt')), bart,
                 sent_numbers, abstracts, ref_abstarcts,  1))
        print("Number of files: " + str(c))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        # for a in a_lst:
        #     _format_to_bert(a)

        # single
        # json_f = args.raw_path + '/test.0.json'
        # _format_to_bert(('test', str(json_f), args, pjoin(args.save_path, str(json_f).replace('.json', '.bert.pt')), bart,
        #          sent_numbers, 40))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        print('Will process {} set with {} json files...'.format(corpus_type, len(a_lst)))
    pool = Pool(args.n_cpus)
    for _ in tqdm(pool.imap(_format_to_bert, a_lst), total=len(a_lst), desc=''):
        pass

    pool.close()
    pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file, bart, sent_numbers_whole, abstracts_dict, ref_abstarcts, debug_idx = params

    abstracts_dict_pos = abstracts_dict.copy() if abstracts_dict is not None else None

    papers_ids = set()
    intro_labels_count = []
    intro_labels_len_count = []

    is_test = corpus_type == 'test'

    CHUNK_SIZE_CONST=-1
    if args.model_name == 'longformer':
        bert = LongformerData(args)
        CHUNK_SIZE_CONST = 2048

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []

    for j, data in tqdm(enumerate(jobs[debug_idx-1:]), total=len(jobs[debug_idx-1:])):
        try:
            paper_id, data_src, data_tgt, data_summary_intro = data['id'], data['src'], data['tgt'], data['intro_summary']
            if not isinstance(data_src[0][0], int):
                data_src = [[idx] + s for idx, s in enumerate(data_src)]

            data_src = remove_ack(data_src)
        except:
            import pdb;pdb.set_trace()
            print("NP: " + save_file + ' idx: ' + str(j) + '\n')
            with open('np_parsing.txt', mode='a') as F:
                F.write(save_file + ' idx: ' + str(j) + '\n')

        if len(data_src) < 5:  continue

        if sent_numbers_whole is not None:
            data_src = [d for d in data_src if d[0] in sent_numbers_whole[paper_id]]

        data_src = [s for s in data_src if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]
        sent_labels = [i for i, s in enumerate(data_src) if s[-6] == 1]
        sent_rg_scores = [s[-3] for i, s in enumerate(data_src) if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]
        sent_rg_scores_intro = [s[-4] for i, s in enumerate(data_src) if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]

        sents_bertScore_gold = [s[-1] for i, s in enumerate(data_src) if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]

        if abstracts_dict is None:
            positive_samples = None
            negative_samples = None
        else:
            positive_samples = get_positive_samples_form_abstracts(paper_id, abstracts_dict_pos)
            negative_samples = get_negative_samples(data_src, sents_bertScore_gold, sampling_len=24, tokenizer=bert)

        if (args.lower):
            source_sents = [([tkn.lower() for tkn in s[1]], s[2], s[0]) for s in data_src]
            data_tgt = [[tkn.lower() for tkn in s] for s in data_tgt]  # arxiv ––non-pubmed
            tkn_len = bert.cal_token_len(source_sents)

        else:
            source_sents = [(s[1], s[2], s[0]) for s in data_src]
            tkn_len = bert.cal_token_len(source_sents)
            debug = False

        if tkn_len > CHUNK_SIZE_CONST:
            try:
                for chunk_num, chunk in enumerate(bert.make_chunks(source_sents, sent_labels=sent_labels,sent_rg_scores=sent_rg_scores, sent_rg_scores_intro=sent_rg_scores_intro,
                                                                   chunk_size=CHUNK_SIZE_CONST, gold_abstract=data_tgt, intro_summary=data_summary_intro)):

                    if chunk_num > 0:
                        break


                    src_chunk, sent_labels_chunk, sent_rg_scores_chunk, curlen, last_chunk, rg_score, sent_rg_scores_intro_chunk = chunk
                    b_data = bert.preprocess_single(src_chunk, data_tgt, sent_labels=sent_labels_chunk,
                                                    sent_rg_scores=sent_rg_scores_chunk,
                                                    sent_rg_scores_intro=sent_rg_scores_intro_chunk,
                                                    use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                    is_test=is_test, intro_summary=data_summary_intro, low_sents=negative_samples, positive_samples=positive_samples, id=paper_id)

                    if (b_data is None):
                        # with open('not_parsed_chunk_multi_processing.txt', mode='a') as F:
                        #     F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' +str(chunk_num)+ '\n')
                        # print(paper_id)
                        continue

                    src_subtoken_idxs, intro_summary_subtoken_idxs, tgt_subtoken_idxs, low_sents_txt_subtokens_idxs, pos_sents_txt_subtoken_idxs, sent_rg_scores, sent_labels_chunk,  \
                    segments_ids, intro_segment_ids, tgt_segment_ids, cls_ids, cls_ids_low, cls_ids_pos, cls_ids_tgt, src_txt, tgt_txt, src_sent_number, src_sent_token_count, \
                    sent_rg_scores_intro, src_txt_tokens, tgt_tokens, graph = b_data

                    # sent_labels_chunk = [0] * len(src_txt)
                    if len(sent_labels_chunk) != len(cls_ids):
                        print(len(sent_labels_chunk))
                        print('Cls length should equal sent lables: {}'.format(paper_id))

                    b_data_dict = {
                        "src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                        "src_sent_rg": sent_rg_scores,
                        "src_sent_rg_intro": sent_rg_scores_intro,
                        "src_sent_labels": sent_labels_chunk.copy(),
                        "segs": segments_ids,
                        'clss': cls_ids,
                        'clss_low': cls_ids_low,
                        'clss_pos': cls_ids_pos,
                        'clss_tgt': cls_ids_tgt,
                        'src_txt': src_txt, "tgt_txt": tgt_txt,
                        'src_tokens': src_txt_tokens, "tgt_tokens": tgt_tokens,
                        "paper_id": paper_id + '___' + str(chunk_num) + '___' + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()),
                        "intro_summary": intro_summary_subtoken_idxs,
                        "low_sents": low_sents_txt_subtokens_idxs,
                        "pos_sents": pos_sents_txt_subtoken_idxs,
                        "segs_intro_summary": intro_segment_ids,
                        "segs_tgt": tgt_segment_ids,
                        "intro_summary_text": data_summary_intro,
                        "graph": graph,
                    }
                    papers_ids.add(paper_id.split('___')[0])
                    # if paper_id.split('___')[0] == '1109.4225':
                    #     import pdb;pdb.set_trace()
                    datasets.append(b_data_dict)

            except Exception:
                with open('not_parsed_chunks_function.txt', mode='a') as F:
                    F.write(
                        save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' + '\n')


        else:

            for chunk_num, chunk in enumerate(
                    bert.make_chunks(source_sents, sent_labels=sent_labels,sent_rg_scores=sent_rg_scores, sent_rg_scores_intro=sent_rg_scores_intro,
                                                               chunk_size=CHUNK_SIZE_CONST, gold_abstract=data_tgt, intro_summary=data_summary_intro)):

                src_chunk, sent_labels_chunk, sent_rg_scores_chunk, curlen, last_chunk, rg_score, sent_rg_scores_intro_chunk = chunk

                b_data = bert.preprocess_single(src_chunk, data_tgt, sent_labels=sent_labels_chunk,
                                                sent_rg_scores=sent_rg_scores_chunk,
                                                sent_rg_scores_intro=sent_rg_scores_intro_chunk,
                                                use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                is_test=is_test, intro_summary=data_summary_intro, low_sents=negative_samples, positive_samples=positive_samples, id=paper_id)

                if b_data == None:
                    with open('not_parsed_processing.txt', mode='a') as F:
                        F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '\n')
                    print(paper_id)
                    continue
                src_subtoken_idxs, intro_summary_subtoken_idxs, tgt_subtoken_idxs, low_sents_txt_subtokens_idxs, pos_sents_txt_subtoken_idxs, sent_rg_scores, sent_labels_chunk,  segments_ids, intro_segment_ids, tgt_segment_ids, \
                cls_ids, cls_ids_low, cls_ids_tgt, cls_ids_pos,  src_txt, tgt_txt, src_sent_number, src_sent_token_count, sent_rg_scores_intro, src_txt_tokens, tgt_tokens, graph = b_data

                if len(sent_labels_chunk) != len(cls_ids):
                    print(len(sent_labels_chunk))
                    print('Cls length should equal sent lables: {}'.format(paper_id))

                b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs, "src_sent_rg": sent_rg_scores,
                               "src_sent_rg_intro": sent_rg_scores_intro,
                               "src_sent_labels": sent_labels_chunk.copy(), "segs": segments_ids,
                               'clss': cls_ids,
                               'clss_low': cls_ids_low,
                               'clss_pos': cls_ids_pos,
                               'clss_tgt': cls_ids_tgt,
                               'src_txt': src_txt,
                               "tgt_txt": tgt_txt, 'src_tokens': src_txt_tokens,
                               "tgt_tokens": tgt_tokens,
                               "paper_id": paper_id + '___' + str(chunk_num) + '___' + datetime.now().strftime(
                                   '%Y%m-%d%H-%M%S-') + str(uuid4()),
                               "intro_summary": intro_summary_subtoken_idxs,
                               "low_sents": low_sents_txt_subtokens_idxs,
                               "pos_sents": pos_sents_txt_subtoken_idxs,
                               "segs_intro_summary": intro_segment_ids, "intro_summary_text": data_summary_intro,
                               "segs_tgt": tgt_segment_ids,
                               "graph": graph,

                               }
                papers_ids.add(paper_id.split('___')[0])
                datasets.append(b_data_dict)

    print('Processed instances %d data' % len(datasets))
    print('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    with open('papers_' + args.model_name + '_' +corpus_type +'.txt', mode='a') as F:
        for p in papers_ids:
            F.write(str(p))
            F.write('\n')

    datasets = []
    gc.collect()
    return save_file, papers_ids, len(papers_ids), intro_labels_count, intro_labels_len_count



################################### LINE FUNCTIONS #############################
# line function
def format_to_lines(args):
    if args.dataset != '':
        corpuses_type = [args.dataset]
    else:
        corpuses_type = ['train', 'val', 'test']

    sections = {}
    for corpus_type in corpuses_type:
        files = []
        for f in glob.glob(args.raw_path +'/*.json'):
            files.append(f)

        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = [(f, args.keep_sect_num) for f in corpora[corpus_type]]
            pool = Pool(args.n_cpus)
            dataset = []
            p_ct = 0
            all_papers_count = 0
            curr_paper_count = 0
            check_path_existence(args.save_path)

            ##########################
            ###### <DEBUGGING> #######
            ##########################

            # for a in tqdm(a_lst, total=len(a_lst)):
            #     d = _format_to_lines(a)
            #     if d is not None:
            #         # dataset.extend(d[0])
            #         dataset.append(d)
            #         if (len(dataset) > args.shard_size):
            #             pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #             check_path_existence(args.save_path)
            #             print(pt_file)
            #             with open(pt_file, 'w') as save:
            #                 # save.write('\n'.join(dataset))
            #                 save.write(json.dumps(dataset))
            #                 print('data len: {}'.format(len(dataset)))
            #                 p_ct += 1
            #                 all_papers_count += len(dataset)
            #                 dataset = []
            # if (len(dataset) > 0):
            #
            #     pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #     print(pt_file)
            #     with open(pt_file, 'w') as save:
            #         # save.write('\n'.join(dataset))
            #         save.write(json.dumps(dataset))
            #         p_ct += 1
            #         all_papers_count += len(dataset)
            #         dataset = []
            #
            # print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))
            ###########################
            ###### </DEBUGGING> #######
            ###########################

            # for d in tqdm(pool.imap_unordered(_format_longsum_to_lines_section_based, a_lst), total=len(a_lst)):
            for d in tqdm(pool.imap_unordered(_format_to_lines, a_lst), total=len(a_lst)):
                # d_1 = d[1]
                if d is not None:
                    all_papers_count+=1
                    curr_paper_count+=1

                    # dataset.extend(d[0])
                    dataset.append(d)
                    # import pdb;pdb.set_trace()
                    # if (len(dataset) > args.shard_size):
                    if (curr_paper_count > args.shard_size):
                        pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                        print(pt_file)
                        with open(pt_file, 'w') as save:
                            # save.write('\n'.join(dataset))
                            save.write(json.dumps(dataset))
                            print('data len: {}'.format(len(dataset)))
                            p_ct += 1
                            dataset = []
                        curr_paper_count = 0


            pool.close()
            pool.join()

            if (len(dataset) > 0):
                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                print(pt_file)
                # all_papers_count += len(dataset)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1

                    dataset = []
            print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))


    # sections = sorted(sections.items(), key=lambda x: x[1], reverse=True)
    # sections = dict(sections)
    # with open('sect_stat.txt', mode='a') as F:
    #     for s, c in sections.items():
    #         F.write(s + ' --> '+ str(c))
    #         F.write('\n')

def _format_to_lines(params):
    src_path, keep_sect_num = params

    def load_json(src_json):
        paper = json.load(open(src_json))
        try:
            id = paper['filename']
        except:
            id = paper['id']

        return paper['sentences'], paper['gold'], id, paper['intro_summary']
    paper_sents, paper_tgt, id, intro_summary = load_json(src_path)
    if paper_sents == -1:
        return None

    return {'id': id, 'src': paper_sents, 'tgt': paper_tgt, 'intro_summary': intro_summary}

## Other utils
def count_dots(txt):
    result = 0
    for char in txt:
        if char == '.':
            result += 1
    return result


def check_path_existence(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)

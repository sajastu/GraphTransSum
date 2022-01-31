import bisect
import gc
import glob
import math
import pdb
import random

import torch

from others.logging import logger
from sect_infos import get_sect_kws
import numpy as np


class Batch(object):
    def _pad(self, data, pad_id=-1, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]

        return rtn_data

    def _pad_2d(self, data, pad_id=-1, width=-1):
        try:
            if (width == -1):
                width = [max([max(len(d) for d in dd) for dd in data])] * len(data)
            rtn_data = [[d + [pad_id] * (width[j] - len(d)) for d in dd] for j, dd in enumerate(data)]
            rtn_data_out = []
            max_1d = max([len(d) for d in rtn_data])
            for r in rtn_data:
                rtd_d = []
                rtd_d.extend(r)
                # if max_1d - len(r) > 0:
                #     import pdb;pdb.set_trace()
                for _ in range(max_1d - len(r)):
                    rtd_d.append(width[1] * [-1])
                rtn_data_out.append(rtd_d)


            return rtn_data_out
        except:
            import pdb;pdb.set_trace()

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        # if is_test:
        self.PAD_ID = -1

        if data is not None:
            self.batch_size = len(data['src'])
            pre_src = [x for x in data['src']]
            pre_tgt = [x for x in data['tgt']]
            graph = [x for x in data['graph']]


            pre_segs = [x for x in data['segment_ids']]
            pre_clss = [x for x in data['cls_ids']]
            pre_clss_tgt = [x for x in data['clss_tgt']]
            pre_src_sent_rg = [x for x in data['src_sent_rg']]
            pre_sent_labels = [x for x in data['sent_labels']]
            paper_id = [x for x in data['paper_id']]


            src = torch.tensor(self._pad(pre_src, -1))
            tgt = torch.tensor(self._pad(pre_tgt, -1))

            segs = torch.tensor(self._pad(pre_segs, 0))
            clss = torch.tensor(self._pad(pre_clss, -1))
            clss_tgt = torch.tensor(self._pad(pre_clss_tgt, -1))
            src_sent_rg = torch.tensor(self._pad(pre_src_sent_rg, 0))

            sent_labels = torch.tensor(self._pad(pre_sent_labels, 0))


            mask_src = ~(src == -1)
            mask_tgt = ~(tgt == -1)
            mask_cls = ~(clss == -1)
            mask_cls_tgt = ~(clss_tgt == -1)

            src[src == -1] = 0
            tgt[tgt == -1] = 0
            clss[clss == -1] = 0
            clss_tgt[clss_tgt == -1] = 0

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'clss_tgt', clss_tgt.to(device))

            setattr(self, 'mask_cls', mask_cls.to(device))

            setattr(self, 'src_sent_rg', src_sent_rg.to(device))
            setattr(self, 'src_sent_labels', sent_labels.to(device))
            # setattr(self, 'section_rg', section_rg.to(device))

            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))

            # for int identifier
            # just for debugging
            setattr(self, 'paper_id', paper_id)
            setattr(self, 'graph', graph)

            if (is_test):
                src_str = [x for x in data['src_txt']]
                setattr(self, 'src_str', src_str)
                setattr(self, 'paper_id', paper_id)
                tgt_str = [x for x in data['tgt_txt']]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """

    # assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        if corpus_type=='val':
            dataset = torch.load(pt_file)
        else:

            dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    if corpus_type == 'val' or corpus_type == 'test':
        pts = sorted(glob.glob(args.bert_data_path + '/' + corpus_type + '.*.pt'), reverse=True)
        try:
            pts = [(int(f.split('.')[-2]), f) for f in pts]
        except:
            pts = [(int(f.split('.')[-3]), f) for f in pts]

        pts = sorted(pts, key=lambda tup: tup[0], reverse=False)
        pts = [p[1] for p in pts]


    else:
        pts = sorted(glob.glob(args.bert_data_path + '/' + corpus_type + '.*.pt'), reverse=True)
        try:
            pts = [(int(f.split('.')[-2]), f) for f in pts]
        except:
            # set.num.bert.pt
            pts = [(int(f.split('.')[-3]), f) for f in pts]

        pts = sorted(pts, key=lambda tup: tup[0], reverse=False)
        pts = [p[1] for p in pts]

        # import random
        # random.seed(888)
        # random.shuffle(pts)
        # pts = pts[:40]
        # pts = glob.glob(args.bert_data_path + '/' + corpus_type + '.6.bert.pt')

    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '/' + corpus_type + '.0' + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    src, labels = new[2], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        # data_len = sum([len(d) for d in self.datasets])
        # import pdb;pdb.set_trace()
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args,
                            dataset=self.cur_dataset, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        graph = ex['graph']


        tgt = ex['tgt']
        src_sent_rg = [round(e, 4) for e in ex['src_sent_rg']]
        sent_labels = ex['src_sent_labels']

        segs = ex['segs']
        clss = ex['clss']
        clss_tgt = ex['clss_tgt']

        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        paper_id = ex['paper_id']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        tgt = tgt[:-1][:self.args.max_pos_intro - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)

        src_sent_rg = src_sent_rg[:max_sent_id]
        sent_labels = sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        src_txt = src_txt[:max_sent_id]

        if (is_test):
            return list({
                'paper_id': paper_id,
                'sent_labels': sent_labels,
                'src': src,
                'tgt': tgt,
                'segment_ids': segs,
                'cls_ids': clss,
                'src_sent_rg': src_sent_rg,
                'src_txt': src_txt,
                'tgt_txt': tgt_txt,
                'graph': graph,
                'clss_tgt': clss_tgt,
            }.values())

        else:
            return list({
                'paper_id': paper_id,
                'sent_labels': sent_labels,
                'src': src,
                'tgt': tgt,
                'segment_ids': segs,
                'cls_ids': clss,
                'src_sent_rg': src_sent_rg,
                'graph': graph,
                'clss_tgt': clss_tgt,
            }.values())

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):


            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = buffer

            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if not self.is_test:
                    minibatch = {
                        'paper_id': [m[0] for m in minibatch],
                        'sent_labels': [m[1] for m in minibatch],
                        'src': [m[2] for m in minibatch],
                        'tgt': [m[3] for m in minibatch],
                        'segment_ids': [m[4] for m in minibatch],
                        'cls_ids': [m[5] for m in minibatch],
                        'src_sent_rg': [m[6] for m in minibatch],
                        'graph': [m[7] for m in minibatch],
                        'clss_tgt': [m[8] for m in minibatch],
                    }
                else:
                    minibatch = {
                        'paper_id':  [m[0] for m in minibatch],
                        'sent_labels':  [m[1] for m in minibatch],
                        'src':  [m[2] for m in minibatch],
                        'tgt': [m[3] for m in minibatch],
                        'segment_ids': [m[4] for m in minibatch],
                        'cls_ids': [m[5] for m in minibatch],
                        'src_sent_rg': [m[6] for m in minibatch],
                        'src_txt': [m[7] for m in minibatch],
                        'tgt_txt': [m[8] for m in minibatch],
                        'graph': [m[9] for m in minibatch],
                        'clss_tgt': [m[10] for m in minibatch],
                    }

                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return


class TextDataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        sent_labels = ex['sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        sent_labels = sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels, sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = batch(p_batch, self.batch_size)

            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch,  round(len(batch) / data_len_arXivL, 4)
            return

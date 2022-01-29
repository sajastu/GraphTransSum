"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
# from apex import amp
from torch.nn import CrossEntropyLoss

from models.reporter import Statistics


def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0):
    compute = NMTLossCompute(
        generator, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute



class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id



    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, output)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output,
                              shard_size, normalization, optim):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        shard_state = self._make_shard_state(batch, output)
        for shard in shards(shard_state, shard_size):
            # out = torch.rand(size=(1, 999, 768))
            # out[:, :992, :] = shard['output'].repeat(1, 999 // 8, 1)
            #
            # out[:, 992:, :] = shard['output'][:, :7, :]
            # shard['output'] = out
            import pdb;pdb.set_trace()

            loss, stats = self._compute_loss(batch, **shard)

            # with amp.scale_loss(loss, optim.optimizer) as scaled_loss:
            loss.div(float(normalization)).backward()

            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output):
        return {
            "output": output,
            "target": batch.sent_labels,
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)

        import pdb;pdb.set_trace()

        scores = self.generator(bottled_output)
        gtruth =target.contiguous().view(-1)

        loss = self.criterion(scores, gtruth)

        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


def greedy_cos_idf(ref_embedding, ref_masks, hyp_embedding, hyp_masks, all_layers=False):
    """
    Compute greedy matching based on cosine similarity.
    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `ref_lens` (list of int): list of reference sentence length.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the reference setence
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_lens` (list of int): list of candidate sentence length.
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
        - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the candidate setence
    """
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    if all_layers:
        B, _, L, D = hyp_embedding.size()
        hyp_embedding = hyp_embedding.transpose(1, 2).transpose(0, 1).contiguous().view(L * B, hyp_embedding.size(1), D)
        ref_embedding = ref_embedding.transpose(1, 2).transpose(0, 1).contiguous().view(L * B, ref_embedding.size(1), D)
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))

    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    if all_layers:
        masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
    else:
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks
    import pdb;pdb.set_trace()

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    # hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    # ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    # precision_scale = hyp_idf.to(word_precision.device)
    # recall_scale = ref_idf.to(word_recall.device)
    # if all_layers:
    #     precision_scale = precision_scale.unsqueeze(0).expand(L, B, -1).contiguous().view_as(word_precision)
        # recall_scale = recall_scale.unsqueeze(0).expand(L, B, -1).contiguous().view_as(word_recall)
    P = (word_precision).sum(dim=1)
    R = (word_recall).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
    ref_zero_mask = ref_masks.sum(dim=1).eq(2)

    if all_layers:
        P = P.view(L, B)
        R = R.view(L, B)
        F = F.view(L, B)

    if torch.any(hyp_zero_mask):
        print(
            "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.", file=sys.stderr,
        )
        P = P.masked_fill(hyp_zero_mask, 0.0)
        R = R.masked_fill(hyp_zero_mask, 0.0)

    if torch.any(ref_zero_mask):
        print("Warning: Empty reference sentence detected; setting raw BERTScores to 0.", file=sys.stderr)
        P = P.masked_fill(ref_zero_mask, 0.0)
        R = R.masked_fill(ref_zero_mask, 0.0)

    F = F.masked_fill(torch.isnan(F), 0.0)

    return P, R, F

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y, dim=3)
    COSINE_DIM4 = lambda x, y: 1 - F.cosine_similarity(x, y, dim=4)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    # BERTSCORE:

class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:
    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).
    Margin is an important hyperparameter and needs to be tuned respectively.
    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss
    :param model: SentenceTransformerModel
    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.
    Example::
        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)
    """
    def __init__(self, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin


    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'triplet_margin': self.triplet_margin}

    def forward(self, rep_anchor, rep_negative=None, rep_pos_oracle=None, rep_pos_gold=None,
                anchor_mask=None, gold_mask=None, oracle_pos_mask=None, negative_mask=None, src_labels=None, mask_cls_low=None):

        if rep_pos_gold is not None and rep_pos_oracle is not None:
            # distance_pos_1 = self.distance_metric(rep_anchor, rep_pos_oracle.repeat(1, rep_anchor.size(1), 1))
            # distance_pos_2 = self.distance_metric(rep_anchor, rep_pos_gold.repeat(1, rep_anchor.size(1), 1))
            # distance_neg = self.distance_metric(rep_anchor, rep_negative.repeat(1, rep_anchor.size(1), 1))
            # losses = F.relu(((distance_pos_1 + 2*distance_pos_2) / 3) - distance_neg + self.triplet_margin)

            # import pdb;pdb.set_trace()

            # BERTSOCRE of gold with anchor
            # greedy_cos_idf(rep_pos_gold, gold_mask, rep_anchor, hyp_masks=anchor_mask)

            distance_pos_1 = self.distance_metric(rep_anchor.unsqueeze(2).repeat(1, 1, rep_pos_gold.size(1), 1),
                                                  rep_pos_gold.unsqueeze(1).repeat(1, rep_anchor.size(1), 1, 1))
            distance_pos_1 = torch.min(distance_pos_1, dim=2).values

            distance_pos_2 = self.distance_metric(rep_anchor.unsqueeze(2).repeat(1, 1, rep_pos_oracle.size(1), 1),
                                                  rep_pos_oracle.unsqueeze(1).repeat(1, rep_anchor.size(1), 1, 1))
            distance_pos_2 = torch.min(distance_pos_2, dim=2).values

            distance_neg = self.distance_metric(rep_anchor, rep_negative.unsqueeze(1).repeat(1, rep_anchor.size(1), 1, 1))
            distance_neg = torch.min(distance_neg, dim=2).values

            losses = F.relu( ((distance_pos_1 + distance_pos_2) / 2) - distance_neg + self.triplet_margin)

        elif rep_pos_gold is not None or rep_pos_oracle is not None:

            if rep_pos_gold is None:

                # distance_pos_1 = self.distance_metric(rep_anchor, rep_pos_gold.repeat(1, rep_anchor.size(1), 1))
                distance_pos_1 = self.distance_metric(rep_anchor.unsqueeze(2).repeat(1, 1, rep_pos_oracle.size(1), 1), rep_pos_oracle.unsqueeze(1).repeat(1, rep_anchor.size(1), 1, 1))
                distance_pos_1 = torch.min(distance_pos_1, dim=2).values

            elif rep_pos_oracle is None:

                # distance_pos_1 = self.distance_metric(rep_anchor, rep_pos_gold.repeat(1, rep_anchor.size(1), 1))
                distance_pos_1 = self.distance_metric(rep_anchor.unsqueeze(2).repeat(1, 1, rep_pos_gold.size(1), 1), rep_pos_gold.unsqueeze(1).repeat(1, rep_anchor.size(1), 1, 1))

                # distance_pos_1 = torch.min(distance_pos_1, dim=2).values
                distance_pos_1 = (torch.sum(distance_pos_1*gold_mask.unsqueeze(1).repeat(1, distance_pos_1.size(1),1), dim=2) * anchor_mask) / ((gold_mask).sum(dim=1).unsqueeze(1).repeat(1, rep_anchor.size(1)))

            # repeat each sentence for neg_sent's size
            # repeat each neg_sample for each of the sentences in the anchor
            # distance_neg = self.distance_metric(rep_anchor.unsqueeze(2).repeat(1, 1, rep_negative.size(1), 1), rep_negative.unsqueeze(1).repeat(1, rep_anchor.size(1), 1, 1))
            # distance_neg = (torch.sum(distance_neg*mask_cls_low.unsqueeze(1).repeat(1, distance_neg.size(1),1), dim=2) * anchor_mask) / ((mask_cls_low).sum(dim=1).unsqueeze(1).repeat(1, rep_anchor.size(1)))

            # distance_neg = torch.min(distance_neg, dim=2).values



            NEG_SENTENCE_SIZE = rep_negative.size(1)
            ANCHOR_SENTENCE_SIZE = rep_anchor.size(1)
            BATCH_SIZE = rep_anchor.size(0)

            rep_negative = rep_negative.view(BATCH_SIZE, 5, NEG_SENTENCE_SIZE, -1)
            rep_anchor_mod = torch.cat(5 * [rep_anchor.unsqueeze(1)], dim=1)
            # rep_anchor_mod = torch.cat(rep_negative.size(1) * [rep_anchor_mod.unsqueeze(2)], dim=2)
            # rep_anchor_mod = torch.cat()
            # rep_anchor_mod = rep_anchor_mod.unsqueeze(2).repeat(1, 1, rep_negative.size(1), 1)
            # rep_anchor = rep_anchor_mod
            # rep_negative_mod = torch.cat(rep_anchor.size(1) * [rep_negative.unsqueeze(1)], dim=1)
            rep_anchor_mod = rep_anchor_mod.unsqueeze(3).repeat(1, 1, 1, NEG_SENTENCE_SIZE, 1)
            rep_negative_mod = rep_negative.unsqueeze(2).repeat(1,1, ANCHOR_SENTENCE_SIZE,1,1)
            # distance_neg = self.distance_metric(rep_anchor, rep_negative.repeat(1, rep_anchor.size(1), 1))
            distance_neg = TripletDistanceMetric.COSINE_DIM4(rep_anchor_mod, rep_negative_mod)
            expanded_clss_low_mask = mask_cls_low.unsqueeze(2).repeat(1, 1, distance_neg.size(2), 1)
            distance_neg = torch.where(expanded_clss_low_mask==False, torch.empty_like(distance_neg).fill_(10000000), distance_neg)
            # import pdb;pdb.set_trace()
            # mean
            distance_neg = torch.sum(distance_neg*expanded_clss_low_mask, dim=3) / expanded_clss_low_mask.sum(dim=3)
            # expanded_anchor_mask = anchor_mask.unsqueeze(1).repeat(1, 5, 1)
            # distance_neg = torch.sum(distance_neg * expanded_anchor_mask,  dim=1) / expanded_anchor_mask.sum(dim=1)
            # distance_neg = torch.min(distance_neg, dim=3).values
            expanded_anchor_mask = anchor_mask.unsqueeze(1).repeat(1, 5, 1)
            distance_neg = torch.where(expanded_anchor_mask==False, torch.empty_like(distance_neg).fill_(0.01), distance_neg)
            distance_neg = torch.sum(distance_neg*expanded_anchor_mask, dim=1) / (distance_neg != 0).sum(dim=1)

            # distance_neg[distance_neg != distance_neg] = 0
            # distance_neg = torch.mean(distance_neg[distance_neg>-1], dim=1).unsqueeze(0)
            # import pdb;pdb.set_trace()
            losses = F.relu(distance_pos_1 - distance_neg + self.triplet_margin)
            losses = losses * anchor_mask.float()

        return losses, distance_neg, distance_pos_1


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.special import lambertw


class SuperLoss(nn.Module):

    def __init__(self, tau=1.5, lam=0.9, batch_size=1):
        super(SuperLoss, self).__init__()
        self.tau = tau
        self.lam = lam
        self.loss_fct = torch.nn.BCELoss(reduction='none')
        self.batch_size = batch_size
        self.processed_docs = 0
        self.accumulative_loss = 0

    def set_update_tau(self):
        self.tau = self.accumulative_loss / self.processed_docs

    def forward(self, logits, targets):
        l_i = self.loss_fct(logits, targets).detach()
        sigma = self.sigma(l_i)
        loss = (self.loss_fct(logits, targets) - self.tau) * sigma + self.lam * (torch.log(sigma) ** 2)
        loss = loss.sum() / self.batch_size

        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().detach().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma
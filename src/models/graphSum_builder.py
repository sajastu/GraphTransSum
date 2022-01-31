import copy

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder, GraphEncoder
from models.optimizers import Optimizer
from transformers import LongformerModel

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)
        else:
            self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, mask_src, clss, segs, id=None):
        if(self.finetune):
            global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
            global_mask[:, :, clss.long()] = 1
            global_mask = global_mask.squeeze(0)

            top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)['last_hidden_state']
            # import pdb;pdb.set_trace()

        else:
            self.eval()
            with torch.no_grad():
                global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                global_mask[:, :, clss.long()] = 1
                global_mask = global_mask.squeeze(0)
                top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)[
                    'last_hidden_state']
        return top_vec

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(768 * 3, 768 * 6),
      nn.ReLU(),
      nn.Linear(768 * 6, 768),
      nn.ReLU(),
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.mlp_combiner = MLP()
        # self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
        #                                        args.ext_dropout, args.ext_layers)

        self.graph_encoder = GraphEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                                       args.ext_dropout, args.ext_layers)


        # for BertSum baseline
        self.wo = nn.Linear(768, 1, bias=True)
        self.sigmoid = nn.Sigmoid()


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.graph_encoder.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)

                # for BertSum Baseline
                for p in self.wo.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)

            if args.param_init_glorot:
                # for p in self.graph_encoder.parameters():
                #     if p.dim() > 1:
                #         xavier_uniform_(p)

                # for BertSum Baseline
                for p in self.wo.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)


        self.to(device)

    def forward(self, src, tgt, src_clss, tgt_clss, src_mask, tgt_mask, src_mask_cls, src_segs, id, graph=None):

        top_vec_src = self.bert(src, src_mask, src_clss, src_segs, id)
        # sents_vec = top_vec_src[torch.arange(top_vec_src.size(0)).unsqueeze(1), src_clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        # sent_scores, attn_nodes, attn_edge = self.ext_layer(sents_vec, mask_cls, edge_w)
        # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        # sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)

        top_vec_tgt = None
        if graph[0] is not None and graph[0].tgt_subgraph is not None:
            top_vec_tgt = self.bert(tgt, tgt_mask, tgt_clss, src_segs, id)


        # for BertSum baseline
        # top_vec_sents = top_vec_src[torch.arange(top_vec_src.size(0)).unsqueeze(1), src_clss]
        # sent_scores = self.sigmoid(self.wo(top_vec_sents)).squeeze(-1)


        graph_nodes, graph_clss_indxs, virtual_node = self.graph_encoder(top_vec_src, top_vec_tgt, src_clss, tgt_clss, src_mask_cls, src_mask, src, id, graph=graph)
        # graph_nodes = graph_nodes.unsqueeze(0)
        graph_sents_nodes = graph_nodes.unsqueeze(0)[torch.arange(graph_nodes.unsqueeze(0).size(0)).unsqueeze(1), graph_clss_indxs]
        sents_vec = top_vec_src[torch.arange(top_vec_src.size(0)).unsqueeze(1), src_clss]

        # combine embeddings of [interaction node] and [graph's source sentences nodes] with top_vec_src

        combined_sents_vector = torch.cat((graph_sents_nodes, sents_vec, virtual_node.unsqueeze(1).repeat(1, src_clss.size(1), 1)), dim=-1)
        top_sents_informed_vecs = self.mlp_combiner(combined_sents_vector)
        sent_scores = self.sigmoid(self.wo(top_sents_informed_vecs))
        sent_scores = sent_scores.squeeze(-1)  # * mask.float()

        return sent_scores, src_mask_cls
        # return sent_scores, mask_cls, None, None


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
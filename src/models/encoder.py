import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from models.GraphConvo import GraphConvolution
from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, ChebConv  # noqa


class GraphEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=2):
        super(GraphEncoder, self).__init__()

        self.conv1 = GATConv(768, 96, heads=8, dropout=0.6)
        self.conv2 = GATConv(96 * 8, 768, heads=1, concat=False,
                             dropout=0.1)

        self.graph_virtual_node = nn.Embedding(1, 768)

        # self.conv1 = GCNConv(768, 1024)
        # self.conv2 = GCNConv(1024, 768)

        # self.gc1 = GraphConvolution(768, 768)
        # self.gc2 = GraphConvolution(768, 768)
        # self.dropout = dropout


        # self.embedding_e = nn.Linear(1, 32)


    def forward(self, src_bert_embeds, tgt_bert_embeds, src_clss, tgt_clss, src_mask_clss, mask_src, src, id, graph=None):
        """

        :param src_bert_embeds: [B, N, DIM]  (t_j)
        :param mask: [B, N] (m_j)
        :param graph: [B, 1] Graph representation
        :return:

        """
        graph = graph[0] # for batch of 1

        graph_embeds, clss_ids_graph, clss_ids_tgt_graph, adj = graph._prepare_graph(
            src_bert_embeds, tgt_bert_embeds, src_clss, tgt_clss, src_mask_clss, mask_src, src, id
        )
        # 1. add virtual node amd update graph embeds
        graph_token_feature = self.graph_virtual_node.weight.unsqueeze(0)
        graph_embeds = torch.cat([graph_embeds, graph_token_feature], dim=1)

        # 2. update adjacency matrix w.r.t source and target (train) sentence ids
        virtual_node_connections = np.zeros((1, adj.shape[1]), dtype=np.long)
        adj = np.concatenate((adj, virtual_node_connections), axis=0)

        virtual_node_connections = np.zeros((1, adj.shape[1]+1), dtype=np.long)
        virtual_node_connections[:, clss_ids_graph.cpu().numpy().astype(int)] = 1
        virtual_node_connections[:, clss_ids_tgt_graph.cpu().numpy().astype(int)] = 1
        adj = np.concatenate((adj, virtual_node_connections.transpose()), axis=1)

        adj = torch.from_numpy(adj)


        # add one new dimension to adjacency matrix

        adj = adj.type(torch.FloatTensor).cuda()
        # x = x.squeeze(0)
        # # import pdb;pdb.set_trace()
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # x = F.log_softmax(x, dim=1)

        ##################################

        # adj_t = adj_t.unsqueeze(1)
        edge_index = (adj > 0).nonzero().t()
        # row, col = edge_index
        # edge_weight = adj[row, col]

        # ensure tensors are in the proper type and device...
        # edge_weight = edge_weight.type(torch.FloatTensor).cuda()


        # ensure tensors are in cuda device
        edge_index = edge_index.cuda()
        # edge_weight = edge_weight

        graph_embeds = graph_embeds.squeeze(0) # [node_size, feat]

        # graph_embeds = F.relu(self.conv1(graph_embeds, edge_index=edge_index, edge_weight=edge_weight))
        # graph_embeds = F.dropout(graph_embeds, training=self.training)
        # graph_embeds = self.conv2(graph_embeds, edge_index, edge_weight)
        # graph_embeds = F.log_softmax(graph_embeds, dim=1)
        graph_embeds = F.dropout(graph_embeds, p=0.1, training=self.training)
        graph_embeds = F.elu(self.conv1(graph_embeds, edge_index))
        graph_embeds = F.dropout(graph_embeds, p=0.1, training=self.training)
        graph_embeds = self.conv2(graph_embeds, edge_index)


        ##################################

        # src_bert_embeds = src_bert_embeds.unsqueeze(0)
        clss_ids_graph = clss_ids_graph.type(torch.LongTensor).cuda()
        # graph_src_sents_embeds = graph_embeds[torch.arange(graph_embeds.size(0)).unsqueeze(1), clss_ids_graph]

        # sent_scores = self.sigmoid(self.wo(src_bert_embeds))
        # import pdb;pdb.set_trace()

        # sent_scores = sent_scores.squeeze(-1) #* mask.float()
        return graph_embeds, clss_ids_graph, graph_embeds[-1, :][None,:]





#
# class ExtTransformerPredictor(nn.Module):
#     def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
#         super(ExtTransformerPredictor, self).__init__()
#         self.d_model = d_model
#         self.num_inter_layers = num_inter_layers
#         self.pos_emb = PositionalEncoding(dropout, d_model)
#         self.transformer_inter = nn.ModuleList(
#             [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
#              for _ in range(num_inter_layers)])
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#         self.wo = nn.Linear(d_model, 4, bias=True)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, top_vecs, mask):
#         """ See :obj:`EncoderBase.forward()`"""
#
#         batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
#         pos_emb = self.pos_emb.pe[:, :n_sents]
#         x = top_vecs * mask[:, :, None].float()
#         x = x + pos_emb
#
#         for i in range(self.num_inter_layers):
#             x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim
#
#         x = self.layer_norm(x)
#         sent_section_scores = self.softmax(self.wo(x))
#         sent_section_scores = sent_section_scores * mask.float().unsqueeze(-1).repeat(1, 1, 4)
#
#         return sent_section_scores


##

# import math
#
# import torch
# import torch.nn as nn
#
# from models.neural import MultiHeadedAttention, PositionwiseFeedForward
#
#
# class Classifier(nn.Module):
#     def __init__(self, hidden_size):
#         super(Classifier, self).__init__()
#         self.linear1 = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, mask_cls):
#         h = self.linear1(x).squeeze(-1)
#         sent_scores = self.sigmoid(h) * mask_cls.float()
#         return sent_scores
#
#
# class PositionalEncoding(nn.Module):
#
#     def __init__(self, dropout, dim, max_len=5000):
#         pe = torch.zeros(max_len, dim)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
#                               -(math.log(10000.0) / dim)))
#         pe[:, 0::2] = torch.sin(position.float() * div_term)
#         pe[:, 1::2] = torch.cos(position.float() * div_term)
#         pe = pe.unsqueeze(0)
#         super(PositionalEncoding, self).__init__()
#         self.register_buffer('pe', pe)
#         self.dropout = nn.Dropout(p=dropout)
#         self.dim = dim
#
#     def forward(self, emb, step=None):
#         emb = emb * math.sqrt(self.dim)
#         if (step):
#             emb = emb + self.pe[:, step][:, None, :]
#
#         else:
#             emb = emb + self.pe[:, :emb.size(1)]
#         emb = self.dropout(emb)
#         return emb
#
#     def get_emb(self, emb):
#         return self.pe[:, :emb.size(1)]
#
#
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, heads, d_ff, dropout):
#         super(TransformerEncoderLayer, self).__init__()
#
#         self.self_attn = MultiHeadedAttention(
#             heads, d_model, dropout=dropout)
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, iter, query, inputs, mask):
#         if (iter != 0):
#             input_norm = self.layer_norm(inputs)
#         else:
#             input_norm = inputs
#
#         mask = mask.unsqueeze(1)
#         context = self.self_attn(input_norm, input_norm, input_norm,
#                                  mask=mask)
#         out = self.dropout(context) + inputs
#         return self.feed_forward(out)
#
#
# class ExtTransformerEncoder(nn.Module):
#     def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
#         super(ExtTransformerEncoder, self).__init__()
#         self.d_model = d_model
#         self.num_inter_layers = num_inter_layers
#         self.pos_emb = PositionalEncoding(dropout, d_model)
#         self.transformer_inter = nn.ModuleList(
#             [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
#              for _ in range(num_inter_layers)])
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#         self.wo = nn.Linear(d_model, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, top_vecs, mask):
#         """ See :obj:`EncoderBase.forward()`"""
#
#         batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
#         pos_emb = self.pos_emb.pe[:, :n_sents]
#         x = top_vecs * mask[:, :, None].float()
#         x = x + pos_emb
#
#         for i in range(self.num_inter_layers):
#             x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
#
#         x = self.layer_norm(x)
#         sent_scores = self.sigmoid(self.wo(x))
#         sent_scores = sent_scores.squeeze(-1) * mask.float()
#
#         return sent_scores
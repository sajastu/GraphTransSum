import numpy
import numpy as np
import torch


class Node():
    def __init__(self,s_sent_idx, s_token_idx, e_sent_idx=None, e_token_idx=None, s_node_txt=None, e_node_txt=None, s_node_lemma=None):
        self.ss_idx = s_sent_idx
        self.st_idx = s_token_idx
        self.es_idx = e_sent_idx
        self.et_idx = e_token_idx
        self.s_txt = s_node_txt
        self.e_txt = e_node_txt
        self.st_lemma_txt = s_node_lemma
        self.tgt_subgraph = None
        self.src_subgraph = None

    def __str__(self):
        return f'({self.ss_idx}, {self.st_idx}, {self.s_txt}) ==> ({self.es_idx}, {self.et_idx}, {self.e_txt})'


class Graph():
    def __init__(self):
        self.type = ''
        self.commonality_edges = []
        self.sentence_edges = []
        self.node_family = []
        self.connectivity = []

        self.subgraph_intra_connectivity = []

        # should be empty in preprocessing step...
        self.nodes_from_common = []
        self.node_embeds = []

    def add_edge(self, node):
        if node.es_idx is not None:
            self.commonality_edges.append(node)
        else:
            self.sentence_edges.append(node)

    def add_edge_batch(self, nodes):
        for node in nodes:
            if node.es_idx is not None:
                self.commonality_edges.append(node)
            else:
                self.sentence_edges.append(node)

    def _get_tuple_repr(self, tuple):
        repr = ''
        for tup in tuple:
            repr += f'{tup}-'
        repr = repr[:-1]
        return repr


    def set_nodes_and_edges(self, bert_nodes_len):
        # Creating nodes in the graph
        for node in self.sentence_edges:
            dest_sent_info = node.ss_idx
            src_tokens_info = node.st_idx
            repr_start = self._get_tuple_repr(src_tokens_info)
            repr_end = self._get_tuple_repr(dest_sent_info)

            self.connectivity.append(
                (repr_start, repr_end)
            )

            if repr_end not in self.node_family:
                self.node_family.append(self._get_tuple_repr(dest_sent_info))

            if repr_start not in self.node_family:
                self.node_family.append(self._get_tuple_repr(src_tokens_info))



        self.node_family = [str(pre_id) for pre_id in range(int(self.node_family[0].split('-')[0]))] + \
                           self.node_family + [str(post_id) for post_id in range(int(self.node_family[-1].split('-')[-1]) + 1, bert_nodes_len)]

        for node in self.commonality_edges:
            src_tokens_info = node.st_idx
            dest_sent_info = node.et_idx
            repr_start = self._get_tuple_repr(src_tokens_info)
            repr_end = self._get_tuple_repr(dest_sent_info)

            self.connectivity.append(
                (repr_start, repr_end)
            )
            self.connectivity.append(
                (repr_end, repr_start)
            )




    def add_connections_with_tgt(self, tgt_subgraph):
        self.tgt_subgraph = tgt_subgraph
        for src_node in self.sentence_edges:
            src_start_token_idx = src_node.st_idx
            src_start_token_lemma = src_node.st_lemma_txt
            for tgt_node in self.tgt_subgraph.sentence_edges:
                tgt_start_token_idx = tgt_node.st_idx
                tgt_start_token_lemma = tgt_node.st_lemma_txt

                if src_start_token_lemma == tgt_start_token_lemma:
                    repr_start = self._get_tuple_repr(src_start_token_idx)
                    repr_end = self._get_tuple_repr(tgt_start_token_idx)
                    self.subgraph_intra_connectivity.append(
                        (repr_start, repr_end)
                    )
                    self.subgraph_intra_connectivity.append(
                        (repr_end, repr_start)
                    )


    def _prepare_graph(self, src_bert_embeds, clss, clss_mask, src_mask, src, id):
        """
            1- Check to see if connectivity matrix is consistent with the nodes
            2- Create adjacency matrix...

        """

        idx_mapping = {k: v for k, v in zip(self.node_family, [s for s in range(len(self.node_family))])}
        adj_mat = np.zeros((len(self.node_family), len(self.node_family)), dtype=numpy.long)
        for pair in self.connectivity:
            start = pair[0]
            end = pair[1]
            new_idx_start = idx_mapping[start]
            new_idx_end = idx_mapping[end]

            assert new_idx_start < len(self.node_family), print(
                f'adjacency matrix\'s index {new_idx_start} should be lower than graph nodes size {len(self.node_family)}')
            assert new_idx_end < len(self.node_family), print(
                f'adjacency matrix\'s index {new_idx_end} should be lower than graph nodes size {len(self.node_family)}')

            adj_mat[new_idx_start][new_idx_end] = 1

        adj_matrix = torch.from_numpy(adj_mat)

        graph_embs = torch.zeros(src_bert_embeds.size(0), len(self.node_family), src_bert_embeds.size(2)).cuda()
        clss_graph = torch.zeros(clss.size(0), clss.size(1)).cuda()
        sentence_num = 0
        for k, v in idx_mapping.items():
            mean_indices = k
            embedded_into = v
            token_embed = torch.index_select(src_bert_embeds, 1, torch.tensor(eval(f'[{mean_indices.replace("-", ",")}]')).cuda()).mean(dim=1).unsqueeze(0)

            if '-' not in k:
                if int(k) in clss.squeeze(0):
                    idx_new = idx_mapping[k]
                    clss_graph = torch.cat((clss_graph[:, :sentence_num], torch.LongTensor([[idx_new]]).cuda(), clss_graph[:, sentence_num + 1:]), dim=1)
                    sentence_num += 1

            graph_embs = torch.cat((graph_embs[:, :embedded_into, :], token_embed, graph_embs[:, embedded_into+1: , :]), dim=1)

        return graph_embs, clss_graph, adj_matrix.cuda()

    def __str__(self):
        print("Commonality edges: ")
        for ce in self.commonality_edges:
            print(ce)

        print("Sentence edges: ")
        for cs in self.sentence_edges:
            print(cs)

    def __iter__(self):
        return (n for n in self.commonality_edges + self.sentence_edges)

    def __len__(self):
        return len(self.commonality_edges) + len(self.sentence_edges)


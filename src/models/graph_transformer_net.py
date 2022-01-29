import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        
    def forward(self, batch_graph, hidden, edge, h_lap_pos_enc=None, h_wl_pos_enc=None):
        import pdb;
        pdb.set_trace()

        # input embedding
        hidden = self.embedding_h(hidden)
        hidden = self.in_feat_dropout(hidden)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            hidden = hidden + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            hidden = hidden + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            edge = torch.ones(edge.size(0), 1).to(self.device)
        edge = self.embedding_e(edge)
        
        # convnets
        for conv in self.layers:
            hidden, edge = conv(batch_graph, hidden, edge)
        batch_graph.ndata['h'] = hidden
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(batch_graph, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(batch_graph, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(batch_graph, 'h')
        else:
            hg = dgl.mean_nodes(batch_graph, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss

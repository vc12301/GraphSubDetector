import torch
import torch.nn as nn

from layers.layers import *

class Model(nn.Module):
    def __init__(self, args, device, **kwargs):
        super(Model, self).__init__()
        self.input_len = args.subseq_len
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.num_hop = args.num_hop

        self.device = device

        self.criterion = nn.MSELoss()

        self.proj = TCNEncoder(input_len=self.input_len, input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        self.semantic_GNN = GNN(self.hidden_dim, self.num_layer, proj=self.proj)
        self.temporal_GNN = GNN(self.hidden_dim, self.num_layer, proj=self.proj)
        self.context = MultiHopContext(self.num_hop, self.device)
        self.decoder = Decoder(self.hidden_dim, self.input_len * self.input_dim)

    def forward(self, semantic_data, temporal_data):
        self.edge_attr = semantic_data.edge_attr
        self.edge_index = semantic_data.edge_index
        self.x = semantic_data.x
        out1 = self.semantic_GNN(semantic_data)
        out2 = self.temporal_GNN(temporal_data)
        out = torch.cat([out1, out2], -1)
        context1 = self.context(out, self.edge_index)
        context2 = self.context(out, temporal_data.edge_index)
        x_hat = self.decoder(out)
        decode_loss = self.criterion(self.x, x_hat)
        return out - context1, out - context2, decode_loss, out

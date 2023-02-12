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
        self.num_level = args.num_level
        self.merge_ratio = args.merge_ratio
        self.k = args.k

        self.device = device
        self.MCAD_blocks = nn.ModuleList([MCADBlock(self.device, self.input_len * (2**l), self.input_dim, self.hidden_dim, self.num_layer, self.num_hop) for l in range(self.num_level)])
        self.merge_blocks = nn.ModuleList([Merge(self.device, self.merge_ratio, self.hidden_dim, self.input_len * (2**l), self.k) for l in range(self.num_level)])

    def forward(self, semantic_data, temporal_data, sliding_wdw):
        self.x = semantic_data.x
        self.y = semantic_data.y
        decode_loss = torch.tensor(0.).to(self.device)
        logit_semantic = torch.zeros(self.x.shape[0], self.hidden_dim * 2).to(self.device)
        logit_temporal = torch.zeros(self.x.shape[0], self.hidden_dim * 2).to(self.device)
        x_mask = torch.ones(semantic_data.x.shape, dtype=torch.long).to(self.device)
        for net, merge in zip(self.MCAD_blocks, self.merge_blocks):
            out, logit_1, logit_2, x_hat = net(semantic_data, temporal_data, x_mask)
            decode_loss += self.masked_mse(semantic_data.x, x_hat, x_mask)
            logit_semantic += logit_1
            logit_temporal += logit_2
            semantic_data, temporal_data, x_mask = merge(out, semantic_data.x, semantic_data.y, x_mask, temporal_data, sliding_wdw)

        return logit_semantic, logit_temporal, decode_loss, None

    def masked_mse(self, x, x_hat, x_mask):
        loss = nn.MSELoss()
        return loss(x[x_mask==1], x_hat[x_mask==1])
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_adj
from torch_geometric.nn.conv.utils.helpers import expand_left
from torch_geometric.data import Data
from torch_scatter import gather_csr, scatter, segment_csr
from copy import deepcopy

class GNNLayer(MessagePassing):
    def __init__(self, k, period, step, input_dim, edge_input_dim=1, hidden_dim=256, flow="source_to_target"):
        super(GNNLayer, self).__init__(flow=flow, aggr='add')
        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            )

    def forward(self, x, edge_index, edge_attr):
        self.network = self.network.to(dtype = torch.float32)
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr, network=self.network)
        return out

    def message(self, x_i, x_j, edge_attr):
        return self.edge_proj(edge_attr) + x_j

    def update(self, inputs, x, network):
        tmp = torch.cat([x, inputs], dim=-1)
        return network(tmp)
        
class SignedGNNLayer(MessagePassing):
    def __init__(self, k, period, step, input_dim, edge_input_dim=1, hidden_dim=256, flow="source_to_target"):
        super(SignedGNNLayer, self).__init__(flow=flow, aggr='add')
        self.k = k
        self.period = period
        self.step = step
        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim

        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.ReLU()
            )

        self.density_transform = nn.Sequential(
            nn.Linear(self.k, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.ReLU()
            )

        self.self_lin = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pos_lin = nn.Linear(input_dim, hidden_dim, bias=False)
        self.neg_lin = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.self_lin.reset_parameters()
        self.pos_lin.reset_parameters()
        self.neg_lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        # self.network = self.network.to(dtype = torch.float32)
        norm_pos_edge_weight, norm_neg_edge_weight = self.compute_adj(x, edge_index, edge_attr)
        out = self.propagate(
            edge_index = edge_index, 
            x=x, 
            edge_attr=edge_attr, 
            norm_pos_edge_weight = norm_pos_edge_weight, 
            norm_neg_edge_weight = norm_neg_edge_weight,
            self_lin = self.self_lin,
            pos_lin = self.pos_lin,
            neg_lin = self.neg_lin
        )
        out += self.bias
        return out

    def message(self, x_i, x_j, norm_pos_edge_weight, norm_neg_edge_weight, self_lin, pos_lin, neg_lin):
        message_1 = self_lin(x_i)
        message_2 = pos_lin(norm_pos_edge_weight * x_j)
        message_3 = neg_lin(norm_neg_edge_weight * x_j)
        return message_1 + message_2 - message_3

    # def update(self, inputs, x, network):
    #     tmp = torch.cat([x, inputs], dim=-1)
    #     return network(tmp)
    
    def compute_adj(self, x, edge_index, edge_attr):
        source, target = edge_index
        # temporal distance
        time_dist = torch.remainder(torch.abs(source-target) * self.step, self.period)
        # distance of node representations
        latent_dist = torch.norm(x[source]-x[target], p=2, dim=-1)
        # transformation of mulitscale (normalized) Euclidean distance
        data_dist = self.edge_proj(edge_attr).squeeze(-1)
        # edge_weight = torch.exp(-torch.square(time_dist)-torch.square(latent_dist)-torch.square(data_dist)).view(-1, self.k)
        edge_weight = torch.exp(-torch.square(latent_dist)-torch.square(data_dist)).view(-1, self.k)
        degree_inv = torch.sum(edge_weight, dim=-1, keepdim=True).pow(-1)

        edge_weight_bias = self.density_transform(edge_weight) + edge_weight.mean()
        signed_edge_weight = edge_weight - edge_weight_bias
        
        pos_edge_weight = torch.clamp(signed_edge_weight, min=0.0)
        neg_edge_weight = torch.abs(torch.clamp(signed_edge_weight, max=0.0))
        signed_degree_inv = torch.sum(torch.abs(signed_edge_weight), dim=-1, keepdim=True).pow(-1)

        norm_pos_edge_weight = signed_degree_inv * pos_edge_weight 
        norm_neg_edge_weight = signed_degree_inv * neg_edge_weight 

        return norm_pos_edge_weight.view(-1,1), norm_neg_edge_weight.view(-1,1)


class GNN(torch.nn.Module):
    def __init__(self, k, period, step, edge_input_dim=1, hidden_dim=256, num_layer=2, flow="source_to_target"):
        super(GNN, self).__init__()
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.k = k 
        self.period = period 
        self.step = step
        self.layers = nn.ModuleList([SignedGNNLayer(self.k, self.period, self.step, self.hidden_dim, self.edge_input_dim, self.hidden_dim, flow) for _ in range(num_layer)])
        self.dropout = nn.Dropout(p=0.)
    def forward(self, data, input):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        out = input
        for layer in self.layers:
            out += self.dropout(layer(out, self.edge_index, self.edge_attr)).squeeze(1)
        return out


class Decoder(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=200):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
    def forward(self, x):
        outs = self.net(x)
        return outs


class MultiHopContext(nn.Module):
    def __init__(self, num_hop, device):
        super(MultiHopContext, self).__init__()
        self.num_hop = num_hop
        self.device = device

    def forward(self, x, edge_index):
        adj = to_dense_adj(edge_index).squeeze(0).permute(1,0)
        high_order_adj = adj.clone()
        ind = high_order_adj.clone()
        for i in range(self.num_hop-1):
            high_order_adj = torch.matmul(high_order_adj, adj)
            ind += high_order_adj
        mask = torch.zeros(ind.shape).to(self.device)
        mask[ind!=0] = 1.
        # idx = np.diag_indices(mask.shape[0])
        # mask[idx[0], idx[1]] = torch.zeros(mask.shape[0]).to(mask.device)
        mask = mask / mask.sum(dim=-1, keepdim=True)
        context = torch.matmul(mask, x)
        return context


class Logit(MessagePassing):
    def __init__(self, flow="source_to_target"):
        super(Logit, self).__init__(flow=flow, aggr='mean')

    def forward(self, x, edge_index):
        logit = self.propagate(edge_index = edge_index, x=x)
        return logit

    def message(self, x_i, x_j):
        return (x_j-x_i).norm(p=2, dim=-1, keepdim=True).square()


class LengthPenalty(MessagePassing):
    def __init__(self, flow="source_to_target"):
        super(LengthPenalty, self).__init__(flow=flow, aggr='sum')

    def forward(self, x, edge_index):
        penalty = self.propagate(edge_index = edge_index, x=x)
        return penalty

    def message(self, x_i, x_j):
        return (x_j-x_i).square().sum(dim=-1, keepdim=True)



class TCNEncoder(nn.Module):

    def __init__(self, input_len, input_dim=1, hidden_dim=256, kernel_size=3):
        super(TCNEncoder, self).__init__()
        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # self.conv = nn.Sequential(
        #     nn.Conv1d(input_dim, hidden_dim, kernel_size, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding='same')
        #     )

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size),
            nn.Tanh()
            )

        self.dropout = nn.Dropout(p=0.)

    def forward(self, x):
        # x: (num_nodes, input_len, input_dim)
        if len(x.shape) < 3:
            x = x.view(x.shape[0], self.input_len, self.input_dim).permute(0,2,1)
        else:
            x = x.permute(0,2,1)
        out = self.conv(x).permute(0,2,1)
        return self.dropout(out)


class TruncStatsPool1d(nn.Module):

    def __init__(self, current_size, max_size, min_size, default_size, step_len, hidden_dim=256):
        super(TruncStatsPool1d, self).__init__()
        self.current_size = current_size
        self.step_len = step_len
        self.max_size = max_size
        self.min_size = min_size
        self.default_size = default_size
        self.hidden_dim = hidden_dim

        self.block_pool = nn.AvgPool1d(step_len)

    def forward(self, x, node_index=None):
        # x: (num_nodes, input_len, hidden_dim)
        x_square = x ** 2
        x = self.block_pool(x.permute(0,2,1)).permute(0,2,1)
        x_square = self.block_pool(x_square.permute(0,2,1)).permute(0,2,1)

        if node_index:
            pool_size = self.current_size.gather(node_index) + self.default_size
        else:
            pool_size = self.current_size + self.default_size

        pool_size = pool_size.clamp(self.min_size,self.max_size)
        pool_size_trunc, pool_size_frac = pool_size.trunc(), pool_size.frac()
        index = pool_size_trunc.long()

        mask = torch.zeros(x.shape[0], x.shape[1] + 1, dtype=x.dtype, device=x.device)
        mask[(torch.arange(x.shape[0]), index)] = 1.
        mask = mask.cumsum(dim=1)[:, :-1]  
        mask = 1. - mask
        mask[(torch.arange(x.shape[0]), index)] = pool_size_frac

        mean = torch.sum(x * mask.unsqueeze(-1), dim=1) / pool_size.view(-1,1)
        var = torch.sum(x_square * mask.unsqueeze(-1), dim=1) / pool_size.view(-1,1) - mean ** 2
        
        length = self.current_size.view(-1,1).repeat(1, self.hidden_dim)
        return  torch.cat([mean, var], dim=-1)
        # return torch.cat([mean, mean, mean], dim=-1)

class MultiScaleStatsPool1d(nn.Module):

    def __init__(self, max_order, min_order, step_len, hidden_dim=256):
        super(MultiScaleStatsPool1d, self).__init__()
        self.max_order = max_order
        self.min_order = min_order
        self.step_len = step_len
        self.hidden_dim = hidden_dim

    def forward(self, x, node_index=None):
        # x: (num_nodes, input_len, hidden_dim)
        x_square = x ** 2
        outs = []
        for order in range(self.max_order - self.min_order + 1):
            length = 2 ** order * self.step_len
            mean = x[:,:length,:].mean(dim=1)
            var = x_square[:,:length,:].mean(dim=1) - mean ** 2
            outs.append(torch.cat([mean, var], dim=-1))

        # return: (max_order+1, num_nodes, hidden_dim)
        if self.max_order - self.min_order > 0:
            return torch.stack(outs, dim=0)
        else:
            return outs[0].unsqueeze(0)
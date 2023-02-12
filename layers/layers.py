import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_adj
from torch_geometric.nn.conv.utils.helpers import expand_left
from torch_geometric.data import Data
from torch_scatter import gather_csr, scatter, segment_csr

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from copy import deepcopy

class GNNLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim=256):
        super(GNNLayer, self).__init__(flow="target_to_source", aggr='add')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(2 * self.input_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            )
        self.edge_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
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
    def __init__(self, k, input_dim, hidden_dim=256):
        super(SignedGNNLayer, self).__init__(flow="target_to_source", aggr='add')
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(2 * self.input_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            )
        self.edge_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.density_transform = nn.Sequential(
            nn.Linear(3 * self.k, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_attr):
        self.network = self.network.to(dtype = torch.float32)
        norm_pos_edge_weight, norm_neg_edge_weight = self.compute_adj(x, edge_index, edge_attr)
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr, network=self.network)
        return out

    def message(self, x_i, x_j, edge_attr):
        return self.edge_proj(edge_attr) + x_j


    def update(self, inputs, x, network):
        tmp = torch.cat([x, inputs], dim=-1)
        return network(tmp)

    def compute_adj(self, x, edge_index, edge_attr):
        dist = torch.norm(x[edge_index[0]]-x[edge_index[1]], p=2, dim=-1)
        edge_weight = torch.exp(-np.square(dist)).view(-1, self.k)

        # degree_inv = torch.sum(edge_weight, dim=-1, keepdim=True).pow(-1)

        density_profile = torch.concat([edge_weight, edge_attr], dim=-1)

        edge_weight_bias = self.density_transform(edge_weight)
        signed_edge_weight = edge_weight - edge_weight_bias
        
        pos_edge_weight = torch.clamp(signed_edge_weight, min=0.0)
        neg_edge_weight = torch.abs(torch.clamp(signed_edge_weight, max=0.0))
        signed_degree_inv = torch.sum(torch.abs(signed_edge_weight), dim=-1, keepdim=True).pow(-1)

        norm_pos_edge_weight = signed_degree_inv * pos_edge_weight 
        norm_neg_edge_weight = signed_degree_inv * neg_edge_weight 

        return norm_pos_edge_weight, norm_neg_edge_weight

        


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j



class GNN(torch.nn.Module):
    def __init__(self, hidden_dim=256, num_layer=2, proj=None):
        super(GNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.proj = proj
        self.layers = nn.ModuleList([GNNLayer(self.hidden_dim, self.hidden_dim) for _ in range(num_layer)])

    def forward(self, data, x_mask):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        if self.proj:
            out = self.proj(self.x, x_mask)
        else:
            out = self.x
        for layer in self.layers:
            out += layer(out, self.edge_index, self.edge_attr).squeeze(1)
        return out


class TCNEncoder(nn.Module):

    def __init__(self, input_len, input_dim=1, hidden_dim=256, kernel_size=5):
        super(TCNEncoder, self).__init__()
        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size),
            )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, x_mask):
        x = x.view(-1, self.input_len, self.input_dim).permute(0,2,1)
        out = self.conv(x).permute(0,2,1)
        out = self._masked_pooling(out, x_mask)
        # out = self.conv(x)
        # out = self.pool(out).view(-1, self.hidden_dim)
        return out

    def _masked_pooling(self, x, x_mask):
        x_mask = x_mask[:,4:-4]
        denom = torch.sum(x_mask, -1, keepdim=True)
        out = torch.sum(x * x_mask.unsqueeze(-1), dim=1) / denom
        # out = torch.mean(x, dim=1)
        return out

class TransformerEncoder(nn.Module):

    def __init__(self, device, input_len, input_dim=1, hidden_dim=256):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._init_cls_token(self.hidden_dim)
        self._init_positional_embedding(self.hidden_dim)

        self.token_embed = nn.Conv1d(input_dim, hidden_dim, 3, padding='same')

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=hidden_dim),
            num_layers=1
            )

    def forward(self, x, x_mask):
        x = x.view(-1, self.input_len, self.input_dim).permute(0,2,1)
        x = self.token_embed(x).permute(0,2,1)
        x = self._add_positional_embedding(x)
        x, x_mask = self._add_cls_token(x, x_mask)
        x = self.transformer_encoder(x, src_key_padding_mask=(1-x_mask).bool())
        out = x[:,0,...]
        return out

    def _init_positional_embedding(self, d_model, max_len=5000):
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def _add_positional_embedding(self, x):
        return x + self.pe[:, :x.size(1)]

    def _init_cls_token(self, d_model):
        self.register_parameter('cls', nn.Parameter(torch.rand(1,1,d_model)))

    def _add_cls_token(self, x, x_mask):
        cls_token = self.cls.tile(x.shape[0],1,1)
        cls_token_mask = torch.zeros((x.shape[0],1), dtype=torch.long).to(self.device)
        x = torch.cat((cls_token,x), dim=1)
        x_mask = torch.cat((cls_token_mask, x_mask), dim=1)
        return x, x_mask


class Decoder(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=200):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
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
        adj = to_dense_adj(edge_index).squeeze(0)
        high_order_adj = adj.clone()
        ind = high_order_adj.clone()
        for i in range(self.num_hop-1):
            high_order_adj = torch.matmul(high_order_adj, adj)
            ind += high_order_adj
        mask = torch.zeros(ind.shape).to(self.device)
        mask[ind!=0] = 1.
        mask = mask / mask.sum(dim=-1, keepdim=True)
        context = torch.matmul(mask, x)
        return context


class Merge(nn.Module):
    def __init__(self, device, merge_thred = 0.5, hidden_dim= 256, subseq_len=200, k = 5):
        super(Merge, self).__init__()
        self.device = device
        self.merge_thred = merge_thred
        self.hidden_dim = hidden_dim
        self.subseq_len = subseq_len
        self.k = k
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ) 

    def forward(self, z, x, y, x_mask, temporal_graph, sliding_wdw):
        # z is an ordered sequence of subsequence representations, and x, y is corresponding subsequences and labels.
        # x: (b, l), x_mask: (b, l), y: (b,), z(b, h) 
        merge_num = int(z.shape[0] * self.merge_thred)

        z_proj = self.proj(z)
        z_proj_roll = z_proj.roll(-1, 0)
        z_sim = torch.einsum('bh,bh->b', z_proj, z_proj_roll)
        z_sim[-1] = -1e8

        indices = z_sim.topk(merge_num, sorted=False).indices
        indices_comp = z_sim.topk(z.shape[0]-merge_num, largest=False, sorted=False).indices

        x_merge, x_merge_mask = self._merge_x(x, x_mask, indices, indices_comp, sliding_wdw)
        y_merge = self._merge_y(y, indices, indices_comp)
        z_merge = self._merge_z(z, indices, indices_comp)

        adj_matrix =  torch.einsum('ah,bh->ab', F.normalize(z_merge, dim=-1), F.normalize(z_merge, dim=-1)).topk(self.k + 1, sorted=True)
        dist, idx = adj_matrix.values, adj_matrix.indices
        dist = dist[...,1:]
        idx = idx[...,1:]
        semantic_graph = self._build_graph_torch(x_merge, y_merge, dist, idx)
        temporal_graph = self._update_temporal_graph_torch(temporal_graph, x_merge, y_merge)

        return semantic_graph, temporal_graph, x_merge_mask

    def _merge_x(self, x, x_mask, indices, indices_comp, sliding_wdw):
        x_roll = x.roll(-1, 0)
        x_mask_roll = x_mask.roll(-1, 0).clone()
        x_merge = torch.cat((x, x_roll), dim=1)
        # x_mask_clip = torch.zeros(x_mask.shape, dtype=torch.long).to(self.device)
        # x_mask_clip[:,:sliding_wdw] = 1
        x_mask_clip = x_mask.clone()
        x_mask_clip[:,sliding_wdw:] = 0
        x_mask_merge = torch.cat((x_mask, x_mask_roll), dim=1)
        x_mask_merge[indices_comp, -self.subseq_len:] = 0
        sorted_indices = torch.sort(x_mask_merge, dim=-1, descending=True, stable=True).indices
        x_merge = x_merge.gather(-1, sorted_indices)
        x_mask_merge = x_mask_merge.gather(-1, sorted_indices)
        return x_merge, x_mask_merge
        
    def _merge_y(self, y, indices, indices_comp):
        y_roll = y.roll(-1, 0)
        y_merge = torch.min(torch.stack((y, y_roll), dim=-1), dim=-1).values
        y_merge[indices_comp] = y[indices_comp]

        return y_merge

    def _merge_z(self, z, indices, indices_comp):
        z_roll = z.roll(-1, 0)
        z_merge = (z + z_roll) / 2
        z_merge[indices_comp] = z[indices_comp]
        return z_merge


    def _build_graph_torch(self, x, y, dist, idx):
         # array like [0,0,0,0,0,1,1,1,1,1,...,n,n,n,n,n] for k = 5 (i.e. edges sources)
        idx_source = torch.arange(x.shape[0], dtype=torch.long).repeat_interleave(dist.shape[-1], dim=0).to(self.device)
        # edge targets, i.e. the nearest k neighbours of point 0, 1,..., n
        idx_target = idx.flatten()

        # stack source and target indices
        idx = torch.stack((idx_source, idx_target), dim=0)
        # edge weights
        attr = dist.flatten()
        attr = attr.unsqueeze(1)
        data = Data(x=x, edge_index=idx, edge_attr=attr, y=y)

        return data

    def _update_temporal_graph_torch(self, temporal_graph, x, y):
        new_temporal_graph = Data(x=x, edge_index=temporal_graph.edge_index, edge_attr=temporal_graph.edge_attr, y=y)
        return new_temporal_graph

class MCADBlock(nn.Module):
    def __init__(self, device, input_len, input_dim=1, hidden_dim=256, num_layer=2, num_hop=1):
        super(MCADBlock, self).__init__()
        self.device = device
        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_hop = num_hop

        self.proj = TCNEncoder(input_len, input_dim=input_dim, hidden_dim=hidden_dim)
        # self.proj = TransformerEncoder(device, input_len, input_dim=input_dim, hidden_dim=hidden_dim)
        self.semantic_GNN = GNN(self.hidden_dim, self.num_layer, proj=self.proj)
        self.temporal_GNN = GNN(self.hidden_dim, self.num_layer, proj=self.proj)
        self.context = MultiHopContext(self.num_hop, self.device)
        self.decoder = Decoder(self.hidden_dim, self.input_len * self.input_dim)

    def forward(self, semantic_data, temporal_data, x_mask):
        self.edge_attr = semantic_data.edge_attr
        self.edge_index = semantic_data.edge_index
        self.x = semantic_data.x
        out1 = self.semantic_GNN(semantic_data, x_mask)
        out2 = self.temporal_GNN(temporal_data, x_mask)
        out = torch.cat([out1, out2], -1)
        context1 = self.context(out, self.edge_index)
        context2 = self.context(out, temporal_data.edge_index)
        x_hat = self.decoder(out)
        return out, out - context1, out - context2, x_hat
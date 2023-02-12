import torch
import torch.nn as nn
import torch.functional as F

from layers.new_layers import *
from torch_geometric.utils import add_self_loops, to_dense_adj, to_undirected



class Model(nn.Module):
    def __init__(self, args, period, step, device, eps=1., **kwargs):
        super(Model, self).__init__()
        self.args = args
        self.period = period
        self.step = step
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.num_hop = args.num_hop
        self.max_order = args.max_order
        self.min_order = args.min_order
        self.default_order = args.default_order
        self.step_len = kwargs['step_len']
        self.num_nodes = kwargs['num_nodes']
        self.input_len = 2 ** self.max_order * self.step_len
        
        self.edge_input_dim = 2 * (self.max_order - self.min_order + 1)

        emb = torch.zeros(self.num_nodes, (self.max_order - self.min_order) + 1)
        emb[:,self.default_order - self.min_order] += eps
        self.register_parameter('current_size', nn.Parameter(emb))

        self.device = device

        self.tcn = TCNEncoder(
            input_len=self.input_len,
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim
        )

        self.pool = MultiScaleStatsPool1d(
            max_order=self.max_order, 
            min_order=self.min_order, 
            step_len=self.step_len, 
            hidden_dim=self.hidden_dim
        )

        self.proj = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
            )

        self.GNN = GNN(args.k, period, step, self.edge_input_dim, self.hidden_dim, self.num_layer, flow="source_to_target")
        # self.temporal_GNN = GNN(1, self.hidden_dim, self.num_layer, flow="target_to_source")
        # self.context = MultiHopContext(self.num_hop, self.device)
        self.logit = Logit()
        self.length_reg = LengthPenalty()
        self.decoder = Decoder(self.hidden_dim, self.input_len * self.input_dim)

    def forward(self, data, node_index=None):
        x = data.x
        r = self.tcn(x)
        z = self.pool(r, node_index)
        distr = F.softmax(self.current_size, dim=-1)
        # z = z[3]
        z = torch.einsum('no,ond->nd', distr, z)
        z = self.proj(z)
        out = self.GNN(data, z)
        # context = self.context(out, data.edge_index)
        logit = self.logit(out, data.edge_index).squeeze(-1)
        x_hat = self.decoder(out)
        # length_penalty = self.length_regularization(data, distr)
        length_penalty = self.length_reg(distr, data.edge_index).sum()
        decode_loss = self.decode_regularization(x, x_hat)
        return logit, length_penalty, decode_loss, out

    def update_length(self):
        for name, param in self.named_parameters():
            if name == 'current_size':
                param.require_grads = True
            else:
                param.require_grads = False

    def update_model_params(self):
        for name, param in self.named_parameters():
            if name == 'current_size':
                param.require_grads = False
            else:
                param.require_grads = True

    def joint_update(self):
        for param in self.parameters():
            param.require_grads = True

    def length_regularization(self, data, distr):
        semantic_edge_index, semantic_edge_attr = to_undirected(data.edge_index, edge_attr=data.edge_attr, reduce='max')
        # temporal_edge_index, temporal_edge_attr = to_undirected(temporal_data.edge_index, edge_attr=temporal_data.edge_attr, reduce='max')
        # semantic_adj = to_dense_adj(semantic_edge_index, edge_attr=semantic_edge_attr).squeeze(0)
        # temporal_adj = to_dense_adj(temporal_edge_index, edge_attr=temporal_edge_attr).squeeze(0)
        adj = to_dense_adj(semantic_edge_index).squeeze(0).unsqueeze(-1)
        # temporal_adj = to_dense_adj(temporal_edge_index).squeeze(0).unsqueeze(-1)
        # adj = torch.cat((semantic_adj, temporal_adj), dim=-1)
        degree = adj.sum(dim=1)
        degree = torch.diag_embed(degree.permute(1,0)).permute(1,2,0)
        laplacian = degree - adj
        return torch.einsum('mo,mnk,no->ko', distr, laplacian, distr).sum()
        # return torch.zeros(1).to(adj.device)

    def decode_regularization(self, x, x_hat):
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        mask[:, :2 ** self.default_order * self.step_len] = 1.
        mse = nn.MSELoss()
        return mse(x[mask==1.], x_hat[mask==1.])
        # return mse(x, x_hat)



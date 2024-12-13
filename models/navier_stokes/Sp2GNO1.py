import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, dense_to_sparse
import torch_geometric.nn as gnn
import inspect
import os
from timeit import default_timer
from torch_geometric.nn import radius_graph
from torch_geometric.nn import knn_graph
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from ARMAConv import ARMAConv
import math
""""this is the fucking best model till now"""

class GraphFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, width, N, s, device):
        super(GraphFourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.device = device
        self.num_nodes = N
        self.no_low_freq = 32
        self.w = nn.Linear(self.width, self.width)

        self.scale = (1 / (in_channels * out_channels))
        self.p = 0.0
        # self.mha_spectral = nn.MultiheadAttention(self.width, 1, 0.0)

        # self.GAT = gnn.GATConv(self.width, self.width, heads=1)
        # self.armaconv = gnn.ARMAConv(self.width, self.width, num_stacks=1, num_layers=1, shared_weights=False)
        
        # self.s = s


        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.no_low_freq, dtype=torch.float))
        
        # self.gcn = GCNConv(in_channels, out_channels)


    def compl_mul2d(self, input, weights):
        out = torch.einsum("bmi,iom->bmo", input, weights)
        return out
    
    def graph_fourier_transform(self, x, U):
        
        batch_size = int(x.shape[0]/self.num_nodes)
        
        x = x.view(batch_size, self.num_nodes, self.in_channels)

        x_wt = torch.stack([torch.mm(U[b].t(), x[b]) for b in range(batch_size)])
        return x_wt
    
    def inverse_graph_fourier_transform(self, x_wt, U):

        batch_size = x_wt.shape[0]
        x_wt = torch.stack([torch.mm(U[b], x_wt[b]) for b in range(batch_size)])
        x_wt = x_wt.view(batch_size*self.num_nodes, self.in_channels)
        
        return x_wt

    def forward(self, x, U, edge_index):


        
        x_ft = self.graph_fourier_transform(x,U)

        # out_ft, _ = self.mha_spectral(x_ft, x_ft, x_ft)
        out_ft = torch.zeros_like(x_ft)

        out_ft[: , :self.no_low_freq, :] = self.compl_mul2d(x_ft[: , :self.no_low_freq , :], self.weights)

        x1 = self.inverse_graph_fourier_transform(out_ft, U)

        
        # x1 = F.gelu(x1)

        # x_out = x1 + x

        x_out = x1 + self.w(x)

        x_out = F.gelu(x_out)

        return x_out


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

    

class GraphFNO(nn.Module):
    def __init__(self, num_fourier_layers,  width,  N, s, device, dataset):
        super(GraphFNO, self).__init__()

        self.num_fourier_layers = num_fourier_layers
        self.width = width
        self.num_nodes = N
        self.device = device

        # self.p = nn.Linear(2, 32)
        self.p = nn.Linear(12, self.width)
        self.s = s
        self.dataset = dataset
        self.graph_fourier_layers = nn.ModuleList()
        self.no_low_freq = 32
        for _ in range(num_fourier_layers):
            self.graph_fourier_layers.append(GraphFourierLayer(self.width, self.width, self.width,
                                                               self.num_nodes, self.s, self.device))

        self.spatial_convs = nn.ModuleList()
        for i in range(num_fourier_layers):
            # self.spatial_convs.append(gnn.ARMAConv(self.width, self.width, num_stacks=1, num_layers=1, shared_weights=False))
            self.spatial_convs.append(FrigateConv(self.width, self.width))
        self.projection = nn.Linear(2*self.width, self.width)

        self.eig_encoder = SineEncoding(self.width)
        self.spectral_transformer = Specformer(self.width, 1)
        self.eig_decoder = nn.Linear(self.width, 1)

        self.q = MLP(self.width, 1, 128)

    def forward(self, x):

        x = x
        # # if self.dataset.normalized:
        # #     pos = self.dataset.x_normalizer.decode(x)     
        # else:
        #     pos = x

        U = self.dataset.U.unsqueeze(0)
        U = U[ :,:, :self.no_low_freq]  
        edge_index = self.dataset.edge_index
        edge_weight = self.dataset.edge_weight
        batch_size = x.size(0)
        num_nodes = x.size(1)
        eig = self.dataset.lambdas[:self.no_low_freq]
        lif_embed = self.dataset.lif_embed.float()
        # edge_index = torch.cat(
        #     [edge_index[i] for i in range(batch_size)],
        #     dim=1
        # )
        # edge_weight = torch.cat(
        #     [edge_weight[i] for i in range(batch_size)],
        #     dim=0
        # )

        # feature engineering

        grid = self.get_grid([x.shape[0], self.dataset.sx, self.dataset.sy], x.device).flatten(1,2)

        x_pos_encode = self.positional_encoding(grid)

        
        x = torch.cat([x, grid], dim=-1)


        x = x.view(-1, x.size(-1))
        x = self.p(x)
        
        x_spatials = []
        x_fourier =  []

        eig = self.eig_encoder(eig)
        eig = self.spectral_transformer(eig)
        eig_values = self.eig_decoder(eig)

        adj = torch.matmul(U.squeeze(0) ,  eig_values * U.squeeze(0).t())
        edge_index, edge_weight = self.create_edge_index_and_weight(adj, 0.00045)


        for i, graph_fourier_layer in enumerate(self.graph_fourier_layers):

            x_fourier = graph_fourier_layer(x, U, edge_index)
            x_spatial = self.spatial_convs[i](x, edge_index, edge_weight, lif_embed)
            # x_spatials.append(x_spatial)
            # if i > 0:
                
            #     x_spatial = x_spatial + x_spatials[-1]
            # #    x_spatial = torch.cat([x_spatial] + x_spatials, dim=1)
            # x_spatials.append(x_spatial)
            x = torch.cat([x_fourier, x_spatial], dim=1)
            x = self.projection(x)

        x = self.q(x)
        return x
    


    def positional_encoding(self, x):

        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        self.width = 32
        self.center = torch.tensor([0.0001,0.0001], device=self.device).reshape(1,1,2)
        self.B = torch.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device=self.device)).reshape(1,1,1,self.width//4)
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)

        # xd = self.fc0(xd)
        # xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,2*self.width + d)
        return xd
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def create_edge_index_and_weight(self, attn_tensor, threshold):
        # Step 1: Thresholding
        adj_matrix = (attn_tensor > threshold).float()

        # Step 2: Creating Edge Indices
        row_indices, col_indices = adj_matrix.nonzero(as_tuple=True)

        # Combine row and column indices into edge_index tensor
        edge_index = torch.stack([row_indices, col_indices], dim=0)

        # Step 3: Edge Weight Matrix
        edge_weight = attn_tensor[row_indices, col_indices]

        return edge_index, edge_weight


class FrigateConv(MessagePassing):
    def __init__(this, in_channels, out_channels):
        super(FrigateConv, this).__init__(aggr='add')
        this.lin = nn.Linear(in_channels, out_channels)
        this.lin_r = nn.Linear(in_channels, out_channels)
        this.lin_rout = nn.Linear(out_channels, out_channels)
        this.lin_ew = nn.Linear(1, 16)
        this.gate = nn.Sequential(
                nn.Linear(16 * 3, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
                nn.Sigmoid(),
                )
        #for p in this.lin_r.parameters():
        #    nn.init.constant_(p.data, 0.)
        #    p.requires_grad = False
    def forward(this, x, edge_index, edge_weight, lipschitz_embeddings):
        if isinstance(x, torch.Tensor):
            x_r = x
            x = this.lin(x)
            x = (x, x)
        else:
            x_r = this.lin_r(x[1])
            x_rest = this.lin(x[0])
            x = (x_rest, x_r)
        out = this.propagate(edge_index, x=x, edge_weight=edge_weight, lipschitz_embeddings=lipschitz_embeddings)
        #out += this.lin_rout(x_r)
        out = F.normalize(out, p=2., dim=-1)
        return out
    
    def message(this, x_j, edge_index_i, edge_index_j, edge_weight, lipschitz_embeddings):

        edge_weight_j = edge_weight.view(-1, 1)
        edge_weight_j = this.lin_ew(edge_weight_j)
        gating_input = torch.cat((edge_weight_j, lipschitz_embeddings[edge_index_i],
            lipschitz_embeddings[edge_index_j]), dim=1)

        gating = this.gate(gating_input)
        output = x_j * gating
        return output
    
class Specformer(nn.Module):

    def __init__(self, hidden_dim=32, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(Specformer, self).__init__()

        # self.norm = norm
        # self.nfeat = nfeat
        # self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        
        # self.feat_encoder = nn.Sequential(
        #     nn.Linear(nfeat, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, nclass),
        # )

        # # for arxiv & penn
        # self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        # self.classify = nn.Linear(hidden_dim, nclass)

        # self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = MLP(hidden_dim, hidden_dim, hidden_dim)

        # self.feat_dp1 = nn.Dropout(feat_dropout)
        # self.feat_dp2 = nn.Dropout(feat_dropout)

        

    # def forward(self, e, u, x):
    def forward(self, e):
        N = e.size(0)
        # ut = u.permute(1, 0)

        # if self.norm == 'none':
        #     h = self.feat_dp1(x)
        #     h = self.feat_encoder(h)
        #     h = self.feat_dp2(h)
        # else:
        #     h = self.feat_dp1(x)
        #     h = self.linear_encoder(h)

        # eig = self.eig_encoder(e)   # [N, d]
        # breakpoint()
        eig = e
        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        # new_e = self.decoder(eig)   # [N, m]

        # for conv in self.layers:
        #     basic_feats = [h]
        #     utx = ut @ h
        #     for i in range(self.nheads):
        #         basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
        #     basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]
        #     h = conv(basic_feats)

        # if self.norm == 'none':
        #     return h
        # else:
        #     h = self.feat_dp2(h)
        #     h = self.classify(h)
        return eig
    
class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=32):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)
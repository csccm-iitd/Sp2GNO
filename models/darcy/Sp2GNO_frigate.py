import torch
import torch.nn as nn
import torch.nn.functional as F
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
import math
""""this is the fucking best model till now"""

class GraphFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, width, N, num_freq, device):
        super(GraphFourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.device = device
        self.num_nodes = N
        self.no_low_freq = num_freq
        self.w = nn.Linear(self.width, self.width)

        self.scale = (1 / (in_channels * out_channels))
        self.p = 0.0
        self.mlp = MLP(self.width, self.width, self.width)

        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.no_low_freq, dtype=torch.float))
        self.norm1 = nn.LayerNorm(self.width)
        self.norm2 = nn.LayerNorm(self.width)
        self.mlp_dropout = MLPDropout(self.width, self.width, self.width)
        self.mlp_dropout1 = MLPDropout1(self.width, self.width, self.width)

    
        self.ft_dropout = nn.Dropout(0.1)

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

        U = U[:, :, :self.no_low_freq]

        
        x_ft = self.graph_fourier_transform(x,U)

        out_ft = torch.zeros_like(x_ft)

        out_ft[: , :self.no_low_freq, :] = self.compl_mul2d(x_ft[: , :self.no_low_freq , :], self.weights)

        x1 = self.inverse_graph_fourier_transform(out_ft, U)

        # x1 = self.ft_dropout(x1)
        # x1 = self.norm1(x1)

        # x1 = self.mlp_dropout(x1)

        # # x1 = F.relu(x1)
        # # x_out = x1 + self.w(x)

        x_out = x1 + self.w(x)

        x_out = F.gelu(x_out)






        # x1 = self.ft_dropout(x1)
        # x1 = self.norm1(x1)

        # x1 = self.mlp_dropout(x1)

        # x1 = F.relu(x1)
        # x_out = x1 + x


        

        x_out = self.norm2(x_out)

        return x_out


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        return x
    
class MLPDropout(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLPDropout, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = self.dropout2(x)
        return x
class MLPDropout1(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLPDropout1, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        # x = self.mlp2(x)
        # x = self.dropout2(x)
        return x
    

class GraphFNO(nn.Module):
    def __init__(self, num_fourier_layers,  width,  N, num_freq, device, dataset):
        super(GraphFNO, self).__init__()

        self.num_fourier_layers = num_fourier_layers
        self.width = width
        self.num_nodes = N
        self.device = device
        self.dataset = dataset
        

        if self.dataset.dataset_name == 'pipe' or self.dataset.dataset_name == 'airfoil':
            self.p = nn.Linear(4,self.width)
        if self.dataset.dataset_name == 'darcy':
            self.p = nn.Linear(3,self.width)
        if self.dataset.dataset_name == 'navier_stokes':
            self.p = nn.Linear(12,self.width)
        if self.dataset.dataset_name == 'elasticity':
            self.p = nn.Linear(112,self.width)
            


        self.graph_fourier_layers = nn.ModuleList()
        self.no_low_freq = num_freq
        for _ in range(num_fourier_layers):
            self.graph_fourier_layers.append(GraphFourierLayer(self.width, self.width, self.width,
                                                               self.num_nodes, self.no_low_freq, self.device))

        self.spatial_convs = nn.ModuleList()
        for i in range(num_fourier_layers):
            # self.spatial_convs.append(gnn.ARMAConv(self.width, self.width, num_stacks=1, num_layers=1, shared_weights=False))
            self.spatial_convs.append(FrigateConv(self.width, self.width))
        self.projection = nn.Linear(2*self.width, self.width)

        self.eig_decoder = nn.Linear(self.width, 1)

        self.q = MLP(self.width, 1, 128)

    def forward(self, batch):
        if self.dataset.dataset_name != 'navier_stokes':
            x,  U = batch.x, batch.U
            if self.dataset.normalized:
                pos = self.dataset.x_normalizer.decode(x)
            elif  self.dataset.dataset_name == 'darcy':
                pos = batch.pos 
            else:
                pos = x
            U = U[:, :, :self.no_low_freq]  
            edge_index = batch.edge_index
            edge_weight = batch.edge_weight
            edge_dist = batch.edge_dist
            batch_size = pos.size(0)
            num_nodes = pos.size(1)
            eig = batch.lambdas.squeeze(0)[:self.no_low_freq]
            lif_embed = batch.lifshitz_embedding.squeeze(0).float()
            edge_index = torch.cat(
                [edge_index[i] for i in range(batch_size)],
                dim=1
            )
            edge_weight = torch.cat(
                [edge_dist[i] for i in range(batch_size)],
                dim=0
            )
        

            if self.dataset.dataset_name != 'elasticity':
                grid = self.get_grid([x.shape[0], self.dataset.sx, self.dataset.sy], x.device).flatten(1,2)
                if self.dataset.dataset_name == 'darcy':
                    if self.dataset.noise_std ==None:
                        x_pos_encode = self.positional_encoding(grid)
                    else:
                        pos = batch.pos
                        x_pos_encode = self.positional_encoding(pos)
                        
                else:
                    x_pos_encode = self.positional_encoding(x)
            else:
                x_pos_encode = self.positional_encoding(x)

            if self.dataset.dataset_name != 'elasticity':
                if self.dataset.dataset_name == 'darcy':
                    x = torch.cat([x, grid], dim=-1)
                    # x = torch.cat([x,x_pos_encode, grid], dim=-1)
                else:
                    x = torch.cat([x, grid], dim=-1)
                    # x = torch.cat([x,x_pos_encode, grid], dim=-1)
                
                # x = self.feat_encoder(x)

            else:
                rr = batch.rr.repeat(1,x.shape[1],1)
                x = torch.cat([x, x_pos_encode,rr ], dim=-1)
                
            self.edge_index = edge_index
            self.edge_weight = edge_weight
       


        if self.dataset.dataset_name == 'navier_stokes':
            x  = batch
            U = self.dataset.U.unsqueeze(0)[:, :, :self.no_low_freq]
            edge_index = self.dataset.edge_index
            edge_weight = self.dataset.edge_weight
            grid = self.get_grid([x.shape[0], self.dataset.sx, self.dataset.sy], x.device).flatten(1,2)
            x = torch.cat([x, grid ], dim=-1)
            lif_embed = self.dataset.lif_embed.float()
        
        x = x.view(-1, x.size(-1))

        x = self.p(x)

        for i, graph_fourier_layer in enumerate(self.graph_fourier_layers):

            x_fourier = graph_fourier_layer(x, U, edge_index)
            # x_spatial = self.spatial_convs[i](x, edge_index)
            x_spatial = self.spatial_convs[i](x, edge_index, edge_weight, lif_embed)
            x = torch.cat([x_fourier, x_spatial], dim=1)
            x = self.projection(x)
            
            # if self.dataset.dataset_name == 'elasticity':
            #     x = x_fourier
            #     x_spatial = self.spatial_convs[0](x, edge_index, edge_weight, lif_embed)
            #     x = torch.cat([x_fourier, x_spatial], dim=1)
            #     x = self.projection(x)
            # else:
            #     x = x_fourier
            #     # x_spatial = self.spatial_convs[i](x, edge_index, edge_weight, lif_embed)
            #     # # x_spatial = self.spatial_convs[i](x, edge_index, edge_weight)
            #     # x = torch.cat([x_fourier, x_spatial], dim=1)
            #     # x = self.projection(x)

        x = self.q(x)
        return x
    


    def positional_encoding(self, x):

        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        width_ = 32
        self.center = torch.tensor([0.0001,0.0001], device=self.device).reshape(1,1,2)
        self.B = torch.pi*torch.pow(2, torch.arange(0, width_//4, dtype=torch.float, device=self.device)).reshape(1,1,1,width_//4)
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*width_//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*width_//4)

        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,2*width_ + d)
        return xd
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
 
class FrigateConv(MessagePassing):
    def __init__(this, in_channels, out_channels):
        super(FrigateConv, this).__init__(aggr='add')
        this.lin = nn.Linear(in_channels, out_channels)
        this.lin_r = nn.Linear(in_channels, out_channels)
        this.lin_rout = nn.Linear(out_channels, out_channels)
        this.lin_ew = nn.Linear(1, 16)
        this.gate = nn.Sequential(
                nn.Linear(16 +2*16, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
                nn.Sigmoid(),
                )

    def forward(this, x, edge_index, edge_weight, lipschitz_embeddings):
        x = this.lin(x)
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
        # output = x_j * edge_weight_j
        return output


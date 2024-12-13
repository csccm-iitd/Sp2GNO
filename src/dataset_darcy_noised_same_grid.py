import os
import torch
import numpy as np

from torch_geometric.nn import radius_graph
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader, Subset
from collections import OrderedDict
from .utilities import *
from .utilities3 import *
from .plotting import plot_graph
from .lipschitz import lifshitz_embedding

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path_x_coords, path_y_coords, path_sigma, path_train, path_test, ntrain, ntest, radius_train, k,
                  radius_test, s, model_type, dataset_name, path_xy=None , path_rr =None, is_sparse=False, old_chunk_size =500 ,new_chunk_Size=500,
                    normalized =True, cropped = True, g_type = 'knn', noise_std = None, dist_weighted_laplacian = False):
        
        self.dist_weighted_laplacian = dist_weighted_laplacian
        self.normalized = normalized
        self.dataset_name = dataset_name
        self.g_type = g_type
        self.ntrain = ntrain
        self.ntest = ntest
        self.s = s
        self.model_type = model_type
        self.is_sparse = is_sparse
        self.radius_train = radius_train
        self.radius_test = radius_test
        self.N_samples = ntrain + ntest
        self.nx = None
        self.ny = None
        self.cropped =None
        self.noise_std = noise_std
        self.old_chunk_size = old_chunk_size
        self.k = k
        
        r=5
        if self.noise_std == None:
            reader = MatReader(path_train)

            train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
            train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

            reader.load_file(path_test)
            test_a = reader.read_field('coeff')[:ntest,::r,::r].reshape(ntest,-1)
            test_u = reader.read_field('sol')[:ntest,::r,::r].reshape(ntest,-1)
            
        else:
            reader = MatReader(path_train)

            train_a = reader.read_field('coeff')[:ntrain,:]
            train_u = reader.read_field('sol')[:ntrain,:]


            reader.load_file(path_test)
            test_a = reader.read_field('coeff')[:ntrain,:]
            test_u = reader.read_field('sol')[:ntrain,:]
            
            
            pos = reader.read_field('pos').squeeze(0) # posis same for all the data points


            # normalize train and test data
            a_normalizer = GaussianNormalizer(train_a)
            train_a_normalized = a_normalizer.encode(train_a)
            test_a_normalized = a_normalizer.encode(test_a)


            u_normalizer = GaussianNormalizer(train_u)
            train_u_normalized = u_normalizer.encode(train_u)
            test_u_normalized = u_normalizer.encode(test_u)
            self.y_normalizer = u_normalizer
            
            self.coeffs_normalized = torch.cat([train_a_normalized.view(-1,85,85), test_a_normalized.view(-1,85,85)], dim=0)
            self.coeffs = torch.cat([train_a.view(-1,85,85), test_a.view(-1,85,85)], dim=0)
            self.u_normalized = torch.cat([train_u_normalized.view(-1,85,85), test_u_normalized.view(-1,85,85)], dim=0)
            self.u = torch.cat([train_u.view(-1,85,85), test_u.view(-1,85,85)], dim=0)
            
            if self.noise_std is None:
                self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_k_{self.k}'
            if noise_std is not None and self.dist_weighted_laplacian:
                self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}_dist_laplace'

            if noise_std is not None and self.dist_weighted_laplacian is False:
                self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}'
                
            if not os.path.exists(self.data_dir):
                

                if noise_std is None:
                    pos_ = self.get_grid(self.coeffs.shape, device= 'cuda')[0].flatten(0,1)
                else:
                    pos_ = pos.flatten(0,1)
                    

                edge_index = knn_graph(pos_,  k = self.k , loop=False)            
                self.edge_index = to_undirected(edge_index, num_nodes=pos_.shape[0])
                self.dist = (pos_[self.edge_index[0]] - pos_[self.edge_index[1]]).norm(dim=-1)
                edge_weight = 1.0 / (self.dist + 1e-6)
                threshold = 5000

                # Clip outliers in edge weights
                edge_weight_clipped = edge_weight.clip(min=None, max=threshold)

                # Normalize the clipped edge weights
                min_weight = edge_weight_clipped.min()
                max_weight = edge_weight_clipped.max()
                self.edge_weight = (edge_weight_clipped - min_weight) / (max_weight - min_weight)
                t1 = time.time()
                self.lips_embed = lifshitz_embedding(self.edge_index, self.dist, 16)
                print("lips_embedding.shape", self.lips_embed.shape)
                t2 = time.time()
                print(f"Time taken for lipschitz embedding: {t2-t1}")
                pos_ = pos_.cpu()
                dm = plot_graph(pos_, edge_index, 0)

                self.N = pos_.shape[0]
                if self.dist_weighted_laplacian:
                    self.lambdas, self.U = calculate_lambdas_U_truncated_edgeweight_sparse(edge_index, self.dist, self.N)
                else:
                    self.lambdas, self.U = calculate_lambdas_U_truncated_sparse(edge_index, num_nodes=self.N)
           
        if self.normalized:
            self.x = self.coeffs_normalized.flatten(1,2).unsqueeze(-1)
            self.y = self.u_normalized
            if self.noise_std == None:
                self.pos = self.get_grid(self.coeffs.shape, self.x.device).flatten(1,2)
            else:
  
                self.pos = pos.flatten(0,1)
            self.N =self.pos.shape[1]
            self.sx, self.sy = self.coeffs.shape[1], self.coeffs.shape[2]
            
        elif not self.normalized:
            self.x = self.coeffs.flatten(1,2).unsqueeze(-1)
            self.y = self.u
            
            if self.noise_std == None:
                self.pos = self.get_grid(self.coeffs.shape, self.x.device).flatten(1,2)
            else:
                self.pos = pos.flatten(0,1)




            self.N =self.pos.shape[0]
            self.sx, self.sy = self.coeffs.shape[1], self.coeffs.shape[2]


        if self.noise_std is None:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_k_{self.k}'
        if noise_std is not None and self.dist_weighted_laplacian:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}_dist_laplace'

        if noise_std is not None and self.dist_weighted_laplacian is False:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}'
        
            
        self.make_and_save_data()  # Process and save the graph data
        
    def get_grid(self, shape, device):

        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    

    
    def make_and_save_data(self):
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        all_files_exist = os.path.exists(os.path.join(self.data_dir, f"data.pt"))
        if all_files_exist:
            self.processed_data = torch.load(os.path.join(self.data_dir, f"data.pt"))
            print("Data files already exist. Skipping data generation.")
            return
        
        data_list = []
        for index in range(self.N_samples):
            data = self.process_and_save_graph_fno(index)
            data_list.append(data)
            data_path = os.path.join(self.data_dir, f"data.pt")
        torch.save(data_list, data_path)
        self.processed_data = torch.load(os.path.join(self.data_dir, f"data.pt"))
    
    def process_and_save_graph_fno(self, index):
    

        data_dict = {
            'x': self.x[index],
            'y': self.y[index],
            'lambdas': self.lambdas,
            'U': self.U,
            'radius_test': self.radius_test,
            'edge_index' : self.edge_index,
            'edge_weight' : self.edge_weight,
            'edge_dist'  : self.dist,
            'lifshitz_embedding': self.lips_embed,
            'pos': self.pos
        }
        return data_dict
        
    def __getitem__(self, index):
        return self.processed_data[index]


    
    def __len__(self):
        return self.N_samples



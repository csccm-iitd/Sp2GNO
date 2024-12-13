import os
import torch
import numpy as np

from torch_geometric.nn import radius_graph
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from collections import OrderedDict
from utilities import *
from utilities3 import *
from plotting import plot_graph
from lipschitz import lifshitz_embedding

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
        


        if self.dataset_name == 'airfoil':
            # self.nx = 50
            # self.ny = 20
            self.nx= 15
            self.ny =12
            self.pos_x = torch.tensor(np.load(path_x_coords), dtype=torch.float)
            self.pos_y = torch.tensor(np.load(path_y_coords), dtype=torch.float)
            self.cropped = cropped
            if self.cropped and self.noise_std is None:
                self.pos_unflatten = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples][:, self.nx:-self.nx, :-self.ny]
                self.pos = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples][:, self.nx:-self.nx, :-self.ny].flatten(1,2)
                self.y = torch.tensor(np.load(path_sigma), dtype=torch.float)[:, 4]
                self.y = self.y[:self.N_samples][:, self.nx:-self.nx, :-self.ny].flatten(1,2)
                i = 0
                plot_airfoil(self.pos[i].unsqueeze(0), self.y[i], f'./plot_{i}.png')
                print(f" dataset pos shape: {self.pos[i].shape}")  # Should match (sx, sy, 2) for 2D grids
                print(f"dataset airfoil pos_unflatten shape: {self.pos_unflatten[i].shape}")  # Should be (sx, sy, 2)
                print(f"dataset airfoil y shape: {self.y[i].shape}")  # Should match (sx * sy)
                print("Dataset pos:", self.pos[i][:10])
                print("Dataset pos_unflatten:", self.pos_unflatten[i][:10])  # Slice for readability
                print("Dataset y:", self.y[i][:10])
                breakpoint()
                self.sx = self.pos_unflatten.shape[1]
                self.sy = self.pos_unflatten.shape[2]
                self.N = self.sx * self.sy
                

            elif not self.cropped and self.noise_std is not None:
                self.pos_unflatten = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples]
                self.pos = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples].flatten(1,2)
                self.y = torch.tensor(np.load(path_sigma), dtype=torch.float)[:, 4]
                self.y = self.y[:self.N_samples].flatten(1,2)
                self.sx = self.pos_unflatten.shape[1]
                self.sy = self.pos_unflatten.shape[2]
                self.N = self.sx * self.sy

            else:
                self.pos_unflatten = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples]
                self.pos = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples].flatten(1,2)
                self.y = torch.tensor(np.load(path_sigma), dtype=torch.float)
                self.y = self.y[:self.N_samples].flatten(1,2)
                self.sx = self.pos_unflatten.shape[1]
                self.sy = self.pos_unflatten.shape[2]
                self.N = self.sx * self.sy
                
            
            

        if self.dataset_name == 'elasticity':
            self.pos = torch.tensor(np.load(path_xy), dtype =torch.float).permute(2,0,1)
            self.rr = torch.tensor(np.load(path_rr), dtype = torch.float).permute(1,0)
            self.y = torch.tensor(np.load(path_sigma), dtype = torch.float).permute(1,0).unsqueeze(-1)

            self.y = self.y[:self.N_samples]

            self.N =self.pos.shape[1]
            self.sx = 32
            self.sy = 32


        if self.dataset_name == 'pipe':
            self.pos_x = torch.tensor(np.load(path_x_coords), dtype=torch.float)
            self.pos_y = torch.tensor(np.load(path_y_coords), dtype=torch.float)
            if self.noise_std is None:
                self.y = torch.tensor(np.load(path_sigma), dtype=torch.float)[:, 0].flatten(1,2)
            else:
                self.y = torch.tensor(np.load(path_sigma), dtype=torch.float).flatten(1,2)
                
            self.pos = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples].flatten(1,2)
            self.y = self.y[:self.N_samples]
            self.N =self.pos.shape[1]
            self.sx = self.pos_x.shape[1]
            self.sy = self.pos_x.shape[2]
            
        if self.dataset_name == 'darcy':

            r=5
            if self.noise_std == None:
                reader = MatReader(path_train)

                train_a = reader.read_field('coeff')[:ntrain,::r,::r]
                train_u = reader.read_field('sol')[:ntrain,::r,::r]

                reader.load_file(path_test)
                test_a = reader.read_field('coeff')[:ntest,::r,::r]
                test_u = reader.read_field('sol')[:ntest,::r,::r]
            else:
                
                reader = MatReader(path_train)

                train_a = reader.read_field('coeff')[:ntrain,:]
                train_u = reader.read_field('sol')[:ntrain,:]
                train_pos = reader.read_field('pos')[:ntrain,:]

                reader.load_file(path_test)
                test_a = reader.read_field('coeff')[:ntrain,:]
                test_u = reader.read_field('sol')[:ntrain,:]
                test_pos = reader.read_field('pos')[:ntrain,:]
                pos = torch.cat([train_pos, test_pos], dim=0)

            

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
                
            if not os.path.exists(os.path.join(self.data_dir, 'chunk_0.pt')) and self.noise_std is None:
            # graph construction
                pos_ = self.get_grid(self.coeffs.shape, device= 'cuda')[0].flatten(0,1)

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





        if self.dataset_name == 'plasticity':
            pass



        if self.dataset_name == 'navier-stokes':
            pass
 


        self.chunk_size = new_chunk_Size 
        self.current_chunk_idx = -1
        self.num_chunks = (self.N_samples + self.chunk_size - 1) // self.chunk_size  # Calculate total number of chunks

        if self.dataset_name != "darcy" and self.normalized:
            self.x_normalizer = UnitGaussianNormalizer(self.pos)
            self.y_normalizer = UnitGaussianNormalizer(self.y)
            self.x = self.x_normalizer.encode(self.pos)
            self.y = self.y_normalizer.encode(self.y)
            self.sx, self.sy = self.coeffs.shape[1], self.coeffs.shape[2]

            
        elif self.dataset_name == "darcy" and self.normalized:
            self.x = self.coeffs_normalized.flatten(1,2).unsqueeze(-1)
            self.y = self.u_normalized
            if self.noise_std == None:
                self.pos = self.get_grid(self.coeffs.shape, self.x.device).flatten(1,2)
            else:
                self.pos = pos
            self.N =self.pos.shape[1]
            self.sx, self.sy = self.coeffs.shape[1], self.coeffs.shape[2]
        
        # this is the case reported in the comparison table    
        elif self.dataset_name == "darcy" and not self.normalized:
            # breakpoint()
            self.x = self.coeffs.flatten(1,2).unsqueeze(-1)
            self.y = self.u
            
            if self.noise_std == None:
                self.pos = self.get_grid(self.coeffs.shape, self.x.device).flatten(1,2)
            else:
                self.pos = pos.flatten(1,2)



            self.N =self.pos.shape[1]
            self.sx, self.sy = self.coeffs.shape[1], self.coeffs.shape[2]

        else:
            self.x = self.pos
            self.y = self.y

        if self.noise_std is None:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_k_{self.k}'
        if noise_std is not None and self.dist_weighted_laplacian:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}_dist_laplace'

        if noise_std is not None and self.dist_weighted_laplacian is False:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}'
        
            
        


        
        if self.data_dir_exists() and not self.chunk_size == self.old_chunk_size:

            self.rechunk_data(self.chunk_size)

        self.make_and_save_data()  # Process and save the graph data
        self.load_next_chunk()  # Load the first chunk into memory

    def data_dir_exists(self):
        return os.path.exists(self.data_dir)

    def check_chunk_size(self):
        # breakpoint()
        # Check if the current chunk size matches the desired chunk size
        chunk_path = os.path.join(self.data_dir, f"chunk_0.pt")
        if os.path.exists(chunk_path):
            chunk_data = torch.load(chunk_path)
            self.old_chunk_size = len(chunk_data)
            self.num_existing_chunks = (self.N_samples + self.old_chunk_size - 1) // self.old_chunk_size
            return len(chunk_data) == self.chunk_size
        return True


    def rechunk_data(self, new_chunk_size):

        if self.noise_std is None:
            new_data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{new_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_k_{self.k}'
        elif self.noise_std is None and self.dist_weighted_laplacian:
            new_data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{new_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}_dist_laplace'
        else:
            new_data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_{self.k}'

        if not os.path.exists(new_data_dir):
            os.makedirs(new_data_dir)
        chunk_idx=0
        for chunk_idx_new in range(self.num_chunks):
            chunk_path_new = os.path.join(new_data_dir, f"chunk_{chunk_idx_new}.pt")
            
            while chunk_idx <= self.num_existing_chunks:
                
                chunk_path = os.path.join(self.data_dir, f"chunk_{chunk_idx}.pt")
                old_chunk_data = torch.load(chunk_path)

                while len(old_chunk_data) < new_chunk_size:
                    chunk_idx +=1
                    next_chunk_path = os.path.join(self.data_dir, f"chunk_{chunk_idx}.pt")
                    if os.path.exists(next_chunk_path):
                        chunk_to_concatenate = torch.load(next_chunk_path)
                        needed_size = new_chunk_size - len(old_chunk_data)
                        old_chunk_data = old_chunk_data + chunk_to_concatenate[:needed_size]
                    else:
                        break
                break

            torch.save(old_chunk_data , chunk_path_new)

        # Update class variables
        self.data_dir = new_data_dir
        self.chunk_size = new_chunk_size
        
    def get_grid(self, shape, device):

        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def load_next_chunk(self):
        # Increment current chunk index
        self.current_chunk_idx += 1
        # Calculate start and end indices of the current chunk
        start_idx = self.current_chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.N_samples)
        # Load data for the current chunk
        chunk_path = os.path.join(self.data_dir, f"chunk_{self.current_chunk_idx}.pt")
        print(f"Loading chunk {self.current_chunk_idx} from {chunk_path}")
        self.loaded_chunk = torch.load(chunk_path, map_location='cpu')
    
    def make_and_save_data(self):
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        all_files_exist = all([os.path.exists(os.path.join(self.data_dir, f"chunk_{i}.pt")) for i in range(self.num_chunks)])
        if all_files_exist:
            print("Data files already exist. Skipping data generation.")
            return
        
        for chunk_idx in range(self.num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.N_samples)
            
            chunk_data = []
            for index in range(start_idx, end_idx):
                if self.model_type == 'GraphFNO':
                    data_dict = self.process_and_save_graph_fno(index)
                elif self.model_type == 'GraphWNO':
                    data_dict = self.process_and_save_graph_wno(index)
                else:
                    raise ValueError("Invalid model_type. Supported types: 'GraphFNO', 'GraphWNO'")
                chunk_data.append(data_dict)
            
            chunk_path = os.path.join(self.data_dir, f"chunk_{chunk_idx}.pt")
            torch.save(chunk_data, chunk_path)
    
    def process_and_save_graph_fno(self, index):
        
        # breakpoint()
        if self.dataset_name != 'darcy'  or ( self.dataset_name != 'darcy' and self.noise_std is not None ):
            pos = self.pos[index].cuda() # pos is always unnormalized and flattened in case of all dataset
        
        elif self.dataset_name == 'darcy' and self.noise_std is None:
            pos = self.pos
        # edge_index = radius_graph(pos,  self.radius_test, loop=False, max_num_neighbors=35)
        # k = 35  , for cropped # k = 45 for non-cropped

        if self.dataset_name == 'airfoil':
            if self.g_type == 'knn' and self.cropped == False and self.noise_std is None:
                # edge_index = knn_graph(pos,  k = 45, loop=False)
                edge_index = knn_graph(pos,  k = self.k, loop=False)

            elif self.g_type == 'knn' and self.cropped == True and self.noise_std is None:
                # edge_index = knn_graph(pos,  k = 35, loop=False)
                edge_index = knn_graph(pos,  k = self.k, loop=False)
            else:
                edge_index = knn_graph(pos,  k = self.k, loop=False)
                # edge_index = knn_graph(pos,  k = 40, loop=False) # 35 <= k <=55 # 50 may work well

        if self.dataset_name == 'elasticity':
            if self.g_type =='radius':
                edge_index = radius_graph(pos.cpu(), self.radius_test, loop=False)
            elif self.g_type == 'knn':
                edge_index = knn_graph(pos.cpu(),  k = self.k, loop=False)

            else:
                pass

        if self.dataset_name == 'pipe':

            if self.g_type =='radius':
                edge_index = radius_graph(pos, self.radius_test, loop=False)
            elif self.g_type == 'knn':
                edge_index = knn_graph(pos,  k = self.k, loop=False)
            else:
                pass
        
        if self.dataset_name == 'darcy' and self.noise_std is not None:

            if self.g_type =='radius':
                edge_index = radius_graph(pos, self.radius_test, loop=False)
            elif self.g_type == 'knn':
                edge_index = knn_graph(pos,  k = self.k , loop=False)
            else:
                pass

        
        if self.dataset_name != 'darcy': 
            edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
            
            dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
            edge_weight = 1.0 / (dist + 1e-6)
            threshold = 5000

            # Clip outliers in edge weights
            edge_weight_clipped = edge_weight.clip(min=None, max=threshold)

            # Normalize the clipped edge weights
            min_weight = edge_weight_clipped.min()
            max_weight = edge_weight_clipped.max()
            edge_weight = (edge_weight_clipped - min_weight) / (max_weight - min_weight)
            t1 = time.time()
            if self.dataset_name == 'airfoil': 
                lips_embed = lifshitz_embedding(edge_index, dist, 32)
            elif self.dataset_name == 'elasticity': 
                lips_embed = lifshitz_embedding(edge_index, dist, 6)

            print("lips_embedding.shape", lips_embed.shape)
            t2 = time.time()
            print(f"Time taken for lipschitz embedding: {t2-t1}")
            pos = pos.cpu()
            if index % 100 ==0:
                dm = plot_graph(pos, edge_index, index)
            if self.dist_weighted_laplacian:
                lambdas, U = calculate_lambdas_U_truncated_edgeweight_sparse(edge_index, dist, self.N)
            else:
                lambdas, U = calculate_lambdas_U_truncated_sparse(edge_index, num_nodes=self.N)
                print('U.shape',U.shape)
        
        
        
        
        
        if self.dataset_name == 'darcy' and self.noise_std is not None: 
            edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
            
            dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
            edge_weight = 1.0 / (dist + 1e-6)
            threshold = 5000

            # Clip outliers in edge weights
            edge_weight_clipped = edge_weight.clip(min=None, max=threshold)

            # Normalize the clipped edge weights
            min_weight = edge_weight_clipped.min()
            max_weight = edge_weight_clipped.max()
            edge_weight = (edge_weight_clipped - min_weight) / (max_weight - min_weight)
            t1 = time.time()
            if self.dataset_name == 'airfoil': 
                lips_embed = lifshitz_embedding(edge_index, dist, 32)
            elif self.dataset_name == 'elasticity': 
                lips_embed = lifshitz_embedding(edge_index, dist, 6)
            elif self.dataset_name == 'darcy': 
                lips_embed = lifshitz_embedding(edge_index, dist, 16)

            print("lips_embedding.shape", lips_embed.shape)
            t2 = time.time()
            print(f"Time taken for lipschitz embedding: {t2-t1}")
            pos = pos.cpu()

            if index % 100 ==0:
                dm = plot_graph(pos, edge_index, index)
            if self.dist_weighted_laplacian:
                lambdas, U = calculate_lambdas_U_truncated_edgeweight_sparse(edge_index, dist, self.N)
            else:
                lambdas, U = calculate_lambdas_U_truncated_sparse(edge_index, num_nodes=self.N)
                print('U.shape',U.shape)
        
        
        
        
                
        if self.dataset_name == 'elasticity':
            data_dict = {
                'x': self.x[index],
                'y': self.y[index],
                'lambdas': lambdas,
                'U': U,
                'radius_test': self.radius_test,
                'edge_index' : edge_index,
                'edge_weight' : edge_weight,
                'edge_dist'  : dist,
                'lifshitz_embedding': lips_embed,
                'rr': self.rr[index],
                'pos': self.pos[index]
            }
        elif self.dataset_name == 'darcy' and self.noise_std is None:
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
            
        elif self.dataset_name == 'darcy' and self.noise_std is not None:
            data_dict = {
                'x': self.x[index],
                'y': self.y[index],
                'lambdas': lambdas,
                'U': U,
                'radius_test': self.radius_test,
                'edge_index' : edge_index,
                'edge_weight' : edge_weight,
                'edge_dist'  : dist,
                'lifshitz_embedding': lips_embed,
                'pos': pos
            }
            
            
        else:
            data_dict = {
                'x': self.x[index],
                'y': self.y[index],
                'lambdas': lambdas,
                'U': U,
                'radius_test': self.radius_test,
                'edge_index' : edge_index,
                'edge_weight' : edge_weight,
                'edge_dist'  : dist,
                'lifshitz_embedding': lips_embed,
                'pos': self.pos[index]
            }
            
        return data_dict

    def process_and_save_graph_wno(self, index):
        pos = self.pos_unflatten[index].flatten(0, 1)
        edge_index = radius_graph(pos, self.radius_test, loop=False)
        edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
        dm = plot_graph(pos, edge_index, index)
        lambdas, U = calculate_lambdas_U_truncated_sparse(edge_index, self.N)
        psi = [calculate_psi_psi_inv_normalized_exact(lambdas,U, s, self.is_sparse) for s in self.s]
        data_dict = {
            'x': self.pos[index],
            'y': self.y[index],
            'psi_list': torch.stack(list(list(zip(*psi))[0]), dim=0)[0],
            'psi_inv_list': torch.stack(list(list(zip(*psi))[1]), dim=0)[0],
            'radius_train': self.radius_train,
        }

        return data_dict
    
    def __getitem__(self, index):
        if index == 0 and self.num_chunks>1:
            # reset the index after one epoch
            self.current_chunk_idx = -1
            self.load_next_chunk()
        # Check if index is within the loaded chunks # The second if statement will run if the
        if index < (self.current_chunk_idx +1) * self.chunk_size:
            # Index is within the current chunk
            chunk_index = index - (self.current_chunk_idx) * self.chunk_size
            # print(f"index: {index}, Chunk index: {chunk_index}, Chunk size: {len(self.loaded_chunk)}")
            return self.loaded_chunk[chunk_index]
        else:
            # Index is outside the current chunk, load the next chunk
            # print('chunk id before loading next chunk',self.current_chunk_idx )
            self.load_next_chunk()
            # print('chunk id after loading next chunk',self.current_chunk_idx )
            # # Return the data corresponding to the requested index from the newly loaded chunk
            return self.loaded_chunk[index % self.chunk_size]


    
    def __len__(self):
        return self.N_samples


def plot_airfoil(x, y, plot_folder):
    # Parameters for reshaping
    sx, sy = 191, 39
    
    # Reshape input
    x_cont = x.view(sx, sy, 2)
    y_cont = y.view(sx, sy)
    
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(plot_folder), exist_ok=True)
    
    # Create plots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # Scatter plot
    scatter = ax[0].scatter(
        x[..., 0].cpu().numpy(),
        x[..., 1].cpu().numpy(),
        c=y.cpu().numpy(),
        s=2,
        cmap='viridis'
    )
    plt.colorbar(scatter, ax=ax[0], orientation='vertical')
    ax[0].set_title('Scatter Plot')

    
    # Adjust shapes for pcolormesh
    x_grid, y_grid = x_cont[..., 0].cpu().numpy(), x_cont[..., 1].cpu().numpy()
    value_grid = y_cont.cpu().numpy()
    
    # Pcolormesh plot
    colormesh = ax[1].pcolormesh(
        x_grid,
        y_grid,
        value_grid,
        cmap='viridis',
        shading='auto'  # Adjust shading as needed
    )
    plt.colorbar(colormesh, ax=ax[1], orientation='vertical')
    ax[1].set_title('Pcolormesh')
    
    # Finalize and save
    plt.tight_layout()
    plt.savefig(plot_folder)
    plt.close(fig)
    
if __name__ == '__main__':
    
    PATH_XY =None
    PATH_rr =None
    INPUT_X =None
    INPUT_Y =None
    OUTPUT_Sigma =None
    radius_train = None
    radius_test = None
    PATH_TRAIN = None
    PATH_TEST = None
    noise_std = None



    PATH = "../../../data/naca"

    INPUT_X = PATH + '/NACA_Cylinder_X.npy'
    INPUT_Y = PATH + '/NACA_Cylinder_Y.npy'
    OUTPUT_Sigma = PATH + '/NACA_Cylinder_Q.npy'
    radius_train = 0.055
    radius_test = 0.055

    ntrain = 1000
    ntest = 200
    batch_size = 1
    # radius_train = 0.06
    # radius_test = 0.06
    dataset_name = 'airfoil'
    normalized = False
    cropped = True
    s =[4.0]
    g_type = 'knn'

    k = 30
    # Create dataset and dataloaders
    dataset = Dataset(INPUT_X, INPUT_Y, OUTPUT_Sigma, PATH_TRAIN, PATH_TEST, ntrain, ntest, radius_train, k, 
                    radius_test,  s, 'GraphFNO', path_xy= PATH_XY, path_rr = PATH_rr,
                        new_chunk_Size=1200, old_chunk_size = 1200,
                        normalized = normalized, cropped =cropped, g_type = 'knn', 
                        dataset_name = dataset_name, noise_std = noise_std , dist_weighted_laplacian = False)
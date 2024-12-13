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

    def __init__(self, path_x_coords, path_y_coords, path_sigma, path_train, path_test, ntrain, ntest, radius_train, k_train, k_test,
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
        self.k = k_train
        self.k_train = k_train
        self.k_test = k_test

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

            self.pos = torch.tensor(np.load(path_xy), dtype =torch.float)
            self.pos_train = self.pos[::2,:,:self.ntrain].permute(2,0,1)
            self.pos_test = self.pos[:,:,self.ntrain:(self.ntrain+ self.ntest)].permute(2,0,1)

            self.rr = torch.tensor(np.load(path_rr), dtype = torch.float).permute(1,0)
            self.y = torch.tensor(np.load(path_sigma), dtype = torch.float).permute(1,0).unsqueeze(-1)

            self.y = self.y[:self.N_samples]

            self.y_train = self.y[:self.ntrain][:,::2,:]
            self.y_test = self.y[self.ntrain:(self.ntrain+self.ntest)]

            self.N =self.pos.shape[1]
            self.sx = 32
            self.sy = 32


        if self.dataset_name == 'pipe':
            self.pos_x = torch.tensor(np.load(path_x_coords), dtype=torch.float)
            self.pos_y = torch.tensor(np.load(path_y_coords), dtype=torch.float)
            self.y = torch.tensor(np.load(path_sigma), dtype=torch.float)[:, 0].flatten(1,2)
            self.pos = torch.stack([self.pos_x, self.pos_y], dim=-1)[:self.N_samples].flatten(1,2)
            self.y = self.y[:self.N_samples]
            self.N =self.pos.shape[1]
            self.sx = self.pos_x.shape[1]
            self.sy = self.pos_x.shape[2]
            
        if self.dataset_name == 'darcy':
            r_test = 3
            r_train = 5
            reader = MatReader(path_train)
            self.train_a = reader.read_field('coeff')[:ntrain,::r_train,::r_train]
            self.train_u = reader.read_field('sol')[:ntrain,::r_train,::r_train]


            # loading test data
            reader.load_file(path_test)
            self.test_a = reader.read_field('coeff')[:ntest,::r_test,::r_test]
            self.test_u = reader.read_field('sol')[:ntest,::r_test,::r_test]
            self.sx = self.test_a.shape[1]
            self.sy = self.test_a.shape[2]
            self.Ntrain = self.train_a.shape[1]*self.train_a.shape[2]
            self.Ntest = self.sx*self.sy



            if self.noise_std is None:
                self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_k_train_{self.k_train}_k_test_{self.k_test}'
            elif self.noise_std is None and self.dist_weighted_laplacian:
                self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_train_{self.k_train}_k_test_{self.k_test}_dist_laplace'
            else:
                self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_train_{self.k_train}_k_test_{self.k_test}'
            
            chunk_path = os.path.join(self.data_dir, f"chunk_0.pt")

            if not os.path.exists(chunk_path):

                pos_train = self.get_grid(self.train_a.shape, device= 'cuda')[0].flatten(0,1)

                pos_train = pos_train.cpu()

                edge_index_train = knn_graph(pos_train,  k = self.k_train , loop=False)            
                self.edge_index_train = to_undirected(edge_index_train, num_nodes=pos_train.shape[0])
                self.dist_train = (pos_train[self.edge_index_train[0]] - pos_train[self.edge_index_train[1]]).norm(dim=-1)
                edge_weight = 1.0 / (self.dist_train + 1e-6)
                threshold = 5000

                # Clip outliers in edge weights
                edge_weight_clipped = edge_weight.clip(min=None, max=threshold)

                # Normalize the clipped edge weights
                min_weight = edge_weight_clipped.min()
                max_weight = edge_weight_clipped.max()
                self.edge_weight = (edge_weight_clipped - min_weight) / (max_weight - min_weight)
                t1 = time.time()
                self.lips_embed_train = lifshitz_embedding(self.edge_index_train, self.dist_train, 16)
                print("lips_embedding.shape", self.lips_embed_train.shape)
                t2 = time.time()
                print(f"Time taken for lipschitz embedding: {t2-t1}")
                pos_train = pos_train.cpu()
                dm = plot_graph(pos_train, edge_index_train, 0)
                self.Ntrain = pos_train.shape[0]
                if self.dist_weighted_laplacian:
                    self.lambdas_train, self.U_train = calculate_lambdas_U_truncated_edgeweight_sparse(self.edge_index_train, self.dist_train, self.N)
                else:
                    self.lambdas_train, self.U_train = calculate_lambdas_U_truncated_sparse(self.edge_index_train, num_nodes=self.Ntrain)


                # test data

                pos_test = self.get_grid(self.test_a.shape, device= 'cuda')[0].flatten(0,1)

                edge_index_test = knn_graph(pos_test.cpu(),  k = self.k_test , loop=False)            
                self.edge_index_test = to_undirected(edge_index_test, num_nodes=pos_test.shape[0])
                self.dist_test = (pos_test[self.edge_index_test[0]] - pos_test[self.edge_index_test[1]]).norm(dim=-1)
                edge_weight = 1.0 / (self.dist_test + 1e-6)
                threshold = 5000

                # Clip outliers in edge weightstrain
                edge_weight_clipped = edge_weight.clip(min=None, max=threshold)

                # Normalize the clipped edge weights
                min_weight = edge_weight_clipped.min()
                max_weight = edge_weight_clipped.max()
                self.edge_weight = (edge_weight_clipped - min_weight) / (max_weight - min_weight)
                t1 = time.time()
                self.lips_embed_test = lifshitz_embedding(self.edge_index_test, self.dist_test, 16)
                print("lips_embedding.shape", self.lips_embed_test.shape)
                t2 = time.time()
                print(f"Time taken for lipschitz embedding: {t2-t1}")
                pos_test = pos_test.cpu()
                dm = plot_graph(pos_test, edge_index_test, 0)
                self.Ntest = pos_test.shape[0]
                if self.dist_weighted_laplacian:
                    self.lambdas_test, self.U_test = calculate_lambdas_U_truncated_edgeweight_sparse(self.edge_index_test, self.dist_test, self.Ntest)
                else:
                    self.lambdas_test, self.U_test = calculate_lambdas_U_truncated_sparse(self.edge_index_test, num_nodes=self.Ntest)





        if self.dataset_name == 'plasticity':
            pass



        if self.dataset_name == 'navier-stokes':
            pass
 


        self.chunk_size = new_chunk_Size 
        self.current_chunk_idx = -1
        self.num_chunks = (self.N_samples + self.chunk_size - 1) // self.chunk_size  # Calculate total number of chunks

        if self.dataset_name != 'darcy':

            self.x = self.pos
            self.y = self.y


        if self.noise_std is None:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_k_train_{self.k_train}_k_test_{self.k_test}'
        elif self.noise_std is None and self.dist_weighted_laplacian:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_train_{self.k_train}_k_test_{self.k_test}_dist_laplace'
        else:
            self.data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_train_{self.k_train}_k_test_{self.k_test}'
        


        
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
            new_data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{new_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_k_train_{self.k_train}_k_test_{self.k_test}'
        elif self.noise_std is None and self.dist_weighted_laplacian:
            new_data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{new_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_train_{self.k_train}_k_test_{self.k_test}_dist_laplace'
        else:
            new_data_dir = f'../processed_graphs_{self.dataset_name}_{self.model_type}_s_{"_".join(map(str, s))}_r_{self.radius_test}_nx_{self.nx}_ny_{self.ny}_cropped_{self.cropped}_chunksize_{self.old_chunk_size}_normalized_{self.normalized}_g_type_{self.g_type}_noise_std_{self.noise_std}_k_train_{self.k_train}_k_test_{self.k_test}'

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
        self.loaded_chunk = torch.load(chunk_path,  map_location='cpu')
    
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
        if self.dataset_name !='darcy':
            if index<self.ntrain:

                pos = self.pos_train[index].cuda()
                self.x = self.pos_train
                self.y = self.y_train
                self.N = pos.shape[0]
                self.k = self.k_train

            else:
                index = index-self.ntrain
                pos = self.pos_test[index].cuda()
                self.x = self.pos_test
                self.y = self.y_test
                self.N = pos.shape[0]
                self.k = self.k_test

         # pos is always unnormalized and flattened in case of all dataset
        # edge_index = radius_graph(pos,  self.radius_test, loop=False, max_num_neighbors=35)
        # k = 35  , for cropped # k = 45 for non-cropped
        if self.dataset_name != 'darcy':
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
                    edge_index = radius_graph(pos, self.radius_test, loop=False)
                elif self.g_type == 'knn':

                    edge_index = knn_graph(pos,  k = self.k, loop=False)

                else:
                    pass

            if self.dataset_name == 'pipe':

                if self.g_type =='radius':
                    edge_index = radius_graph(pos, self.radius_test, loop=False)
                elif self.g_type == 'knn':
                    edge_index = knn_graph(pos,  k = self.k, loop=False)
                else:
                    pass
            
            if self.dataset_name == 'darcy':

                if self.g_type =='radius':
                    edge_index = radius_graph(pos, self.radius_test, loop=False)
                elif self.g_type == 'knn':
                    edge_index = knn_graph(pos,  k = self.k , loop=False)
                else:
                    pass

            
            
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
                'rr': self.rr[index]
            }
        elif self.dataset_name == 'darcy':
            data_dict = {
                'x': self.train_a[index] if index < self.ntrain else self.test_a[index - self.ntrain],
                'y': self.train_u[index] if index < self.ntrain else self.test_u[ index- self.ntrain],
                'lambdas': self.lambdas_train if index < self.ntrain else self.lambdas_test,
                'U': self.U_train if index < self.ntrain else self.U_test,
                'radius_test': self.radius_test,
                'edge_index' : self.edge_index_train if index < self.ntrain else self.edge_index_test,
                'edge_weight' : self.edge_weight,
                'edge_dist'  : self.dist_train if index < self.ntrain else self.dist_test,
                'lifshitz_embedding': self.lips_embed_train if index < self.ntrain else self.lips_embed_train
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



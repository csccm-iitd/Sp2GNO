import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer
import os
import shutil

from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected, dense_to_sparse, add_remaining_self_loops
from torch.utils.data import Subset, DataLoader
from datetime import datetime
import inspect
import importlib.util
import sys
import logging
from torch_geometric.nn import radius_graph
from torch_geometric.nn import knn_graph
from src.lipschitz import lifshitz_embedding
from src.utilities3 import *
from src.utilities import *
from src.dataset import Dataset
from models.navier_stokes.ns_spatial import GraphFNO
from src.save_loss_excel import export_excel

dataset_name = 'navier_stokes'

def setup_logging(log_folder, log_level=logging.INFO):
    # Configure logging
    logging.basicConfig(
        level=log_level, 

        format='%(message)s',
        filename=os.path.join(log_folder, "log.txt"), 
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO) 
    logging.getLogger('').addHandler(console)


# Define a function to create a folder with a timestamp
def create_log_folder():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = f"./log_folder/training_logs_{timestamp}"
    os.makedirs(log_folder)
    path = f'./{log_folder}/GraphFNO_{dataset_name}_Results/'
    path_model =  path+'model_irregular/'
    path_image_test61 = path +'image_irregular/'
    os.makedirs(path_model)
    os.makedirs(path_image_test61)

    return log_folder , path_model, path_image_test61

# Define a function to get the module name from where GraphFNO is imported
def get_module_name():
    import_line = next(line for line in open(__file__) if "import GraphFNO" in line)
    module_name = import_line.split(" ")[1]  # Get the module name from the import statement
    return module_name

# Define a function to save the current script and the module script to the log folder
def save_scripts(log_folder):
    shutil.copy(__file__, os.path.join(log_folder, "train.py"))
    module_name = get_module_name()
    module_spec = importlib.util.find_spec(module_name)
    module_path = module_spec.origin
    shutil.copy(module_path, os.path.join(log_folder, f"{module_name}.py"))


def save_results(log_folder, results, training_info):
    with open(os.path.join(log_folder, "results.txt"), "w") as results_file:
        results_file.write(results)
        results_file.write("\n\nTraining Loop Information:\n")
        results_file.write(training_info)
        results_file.write("\n")

def Darcy_imshow(Data, vertexCoords, axes, cmap='viridis', vmin=None, vmax=None, title=None):
    x = vertexCoords[:, 0]
    y = vertexCoords[:, 1]
    z = Data.flatten()  # Flatten the 2D array to a 1D array
    im = axes.tripcolor(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')


    return im

def model_predict(model_ , xx):
    xx = xx[None, :,:].to(device)
    for t in range(0, T_out, step):
        im = model_(xx).unsqueeze(0)

        if t == 0:
            pred = im
        else:
            pred = torch.cat((pred, im), -1)

        # xx = torch.cat(( pred , xx[..., (t+step):]), dim=-1)
        xx = torch.cat((xx[..., step:], im), dim=-1)   # best results
        # xx = torch.cat(( xx[..., (t+step):] , pred ), dim=-1)
    return pred


log_folder , path_model, path_image_test61 = create_log_folder()
save_scripts(log_folder)
setup_logging(log_folder)



  
#==================================data preprocessing starts===================================

torch.manual_seed(0)
np.random.seed(0)

PATH = '..'
TRAIN_PATH = PATH + "/data/navier_stokes/NavierStokes_V1e-5_N1200_T20.mat"
TEST_PATH =  PATH + "/data/navier_stokes/NavierStokes_V1e-5_N1200_T20.mat"

 # data related hyper parameters
r = 1
s = int(((64 - 1)/r) + 1)
n = s**2
m = 100
k = 1

radius_train = 0.1
radius_test = 0.1

print('resolution', s)


ntrain = 1000
ntest = 200

batch_size = 1
batch_size2 = 1
# batch_size2 = 2

# training related hyperparameters

epochs = 1500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5


#==================================data preprocessing starts===================================

t1 = default_timer()
S = 64
T_in = 10
T_out = 10
step = 1

B = 1

"""
def get_grid( shape):

    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).squeeze(0).flatten(0,1)

def get_graph(pos, radius=0.08, g_type = 'knn', dist_bool =True):
    if g_type =='radius':
        edge_index = radius_graph(pos, radius, loop=False)
    elif g_type == 'knn':
        edge_index = knn_graph(pos,  k = 35, loop=False)
    edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
    dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    if dist_bool:
        edge_weight = dist
    else:
        edge_weight = 1.0 / (dist + 1e-6)

    return edge_index , edge_weight

pos = get_grid([1,s,s])
edge_index , edge_weight = get_graph(pos)
    
N = pos.shape[0]

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain, ::r, ::r, :T_in].flatten(1,2)
train_u = reader.read_field('u')[:ntrain, ::r, ::r, T_in:T_in + T_out].flatten(1,2)

test_a = reader.read_field('u')[-ntest:, ::r, ::r, :T_in].flatten(1,2)
test_u = reader.read_field('u')[-ntest:, ::r, ::r, T_in:T_in + T_out].flatten(1,2)

print(train_u.shape)
print(test_u.shape)

train_a = train_a.reshape(ntrain, s*s, T_in)
test_a = test_a.reshape(ntest, s*s, T_in)

"""
# print("train_a.shape", train_a.shape)
# print("test_a.shape", test_a.shape)
# print("train_u.shape", train_u.shape)
# print("test_u.shape", test_u.shape)
# print("=======================reshape======================")
# assert (S == train_u.shape[-2])
# assert (T_out == train_u.shape[0])


# print("train_a.shape", train_a.shape)
# print("test_a.shape", test_a.shape)
# print("train_u.shape", train_u.shape)
# print("test_u.shape", test_u.shape)

class Dataset:
    def __init__(self, train_path, s, T_in , T_out, device, gtype= 'knn',  dist_weighted_laplacian = False):
        self.sx = s
        self.sy = s
        self.x_normalizer = None 
        self.y_normalizer = None
        self.g_type = gtype
        self.device = device
        
        self.read_data(train_path)
        self.pos = self.get_grid([1,s,s])
        self.N = self.pos.shape[0]
        self.dist_weighted_laplacian  = dist_weighted_laplacian
        self.edge_index, self.edge_weight = self.get_graph(self.pos , radius=0.08, g_type = self.g_type, dist_bool =True)
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.lif_embed = torch.tensor(lifshitz_embedding(self.edge_index, self.edge_weight, 8)).to(self.device)
        if self.dist_weighted_laplacian:
            self.lambdas, self.U = calculate_lambdas_U_truncated_edgeweight_sparse(self.edge_index, self.edge_weight, num_nodes=self.N)
        else:
            self.lambdas, self.U = calculate_lambdas_U_truncated_sparse(self.edge_index, self.N)
        self.lambdas = self.lambdas.to(self.device)
        self.U = self.U.to(device)

    def read_data(self, TRAIN_PATH):
        reader = MatReader(TRAIN_PATH)
        self.train_a = reader.read_field('u')[:ntrain, ::r, ::r, :T_in].flatten(1,2)
        self.train_u = reader.read_field('u')[:ntrain, ::r, ::r, T_in:T_in + T_out].flatten(1,2)

        self.test_a = reader.read_field('u')[-ntest:, ::r, ::r, :T_in].flatten(1,2)
        self.test_u = reader.read_field('u')[-ntest:, ::r, ::r, T_in:T_in + T_out].flatten(1,2)

        print(self.train_u.shape)
        print(self.test_u.shape)

        self.train_a = self.train_a.reshape(ntrain, self.sx*self.sy, T_in)
        self.test_a = self.test_a.reshape(ntest, self.sx*self.sy, T_in)

    def get_grid( self, shape):

        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).squeeze(0).flatten(0,1)

    def get_graph(self, pos, radius=0.08, g_type = 'knn', dist_bool =True):
        if g_type =='radius':
            edge_index = radius_graph(pos, radius, loop=False)
        elif g_type == 'knn':
            edge_index = knn_graph(pos,  k = 3030, loop=False)
        edge_index = to_undirected(edge_index, num_nodes=pos.shape[0])
        dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

        if dist_bool:
            edge_weight = dist
        else:
            edge_weight = 1.0 / (dist + 1e-6)

        return edge_index , edge_weight

        

device = torch.device('cuda:0')
dataset = Dataset(TRAIN_PATH, s, T_in, T_out , device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dataset.train_a, dataset.train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dataset.test_a, dataset.test_u), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################


# model hyper paramters

width = 32
num_wavelet_layers = 6

model = GraphFNO(num_wavelet_layers , width, dataset.N, s, device, dataset).to(device)

# print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

logging.info("==================================Training Started============================================")


myloss = LpLoss(size_average=False)
ttrain = []
ttest = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    batch_loss = 0.0
    for i, (xx, yy) in enumerate(train_loader):
        loss = 0.0
        xx = xx.to(device)
        yy = yy.to(device)

        # print("xx.shape", xx.shape)
        # print("yy.shape", yy.shape)

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            
            im = model(xx).unsqueeze(0)

            # print('im.shape' , im.shape)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            batch_loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            # xx = torch.cat(( xx[..., (t+step):] , pred  ), dim=-1)
            # xx = torch.cat(( pred , xx[..., (t+step):]), dim=-1)
            # print(f"xx.shape at {t}", xx.shape)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()
        
        if (i + 1) % B == 0:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0.0
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
    ttrain.append(train_l2_full/ntrain)

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                im = model(xx).unsqueeze(0)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)    # best result
                # xx = torch.cat(( xx[..., (t+step):] , pred  ), dim=-1)
                # xx = torch.cat(( pred , xx[..., (t+step):]), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
    
    # if  and ep % 500 == 0:
    if ep >= 1000 and ep % 500 == 0:
        torch.save(model.state_dict(), os.path.join(path_model, f'model_epoch_{ep}.pth'))
        logging.info(f'Model saved at epoch {ep}')


    if ep%100==0:
        
        xx ,  yy  = test_loader.dataset[0]
        xx = xx.detach().cpu()
        truth = yy.detach().cpu()
        approx = model_predict(model , xx).reshape(dataset.N, T_out).detach().cpu()
        print("truth.shape", truth.shape)
        print("approx.shape", approx.shape)

        mesh_coordinates = dataset.pos
        _min = torch.min(truth)
        _max = torch.max(truth)

        num_t_plot = len(range(0, T_out,2))

        fig, ax = plt.subplots(4, num_t_plot, figsize=(5 * num_t_plot, 20))
        row_titles = ["Ground Truth", "Prediction", "Error"]
        column_titles = [f"Time: {t+1}" for t in range(0, T_out, 2)]
        
        im_init_cond = Darcy_imshow(xx[...,0], dataset.pos, ax[0,2],
                            cmap='viridis',vmin=_min, vmax=_max ,title=f' ground truth at time : {t+1}')
        ax[0,2].set_ylabel("Initial Condition", fontsize=30, fontname='serif')
        
        for i,  t in enumerate(range(0,T_out,2)):

            
            im_train_truth = Darcy_imshow(truth[...,t], dataset.pos, ax[1,i],
                            cmap='viridis',vmin=_min, vmax=_max ,title=f' ground truth at time : {t+1}')

            im_train_approx = Darcy_imshow(approx[...,t], dataset.pos, ax[2,i], 
                            cmap='viridis',vmin=_min, vmax=_max , title= f'prediction at : {t+1}')
            im_train_error = Darcy_imshow((approx[...,t] - truth[...,t]) ** 2, mesh_coordinates, ax[3,i], cmap='viridis', title = "error")
            
            for axes in ax[:, i]:
                axes.set_xticks([])  # Remove x-axis ticks
                axes.set_yticks([])  # Remove y-axis ticks
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                axes.spines['bottom'].set_visible(False)
                axes.spines['left'].set_visible(False)


        for i, ax_row in enumerate(ax[1:, 0]):
            ax_row.set_ylabel(row_titles[i], fontsize=30, fontname='serif')

        for i, ax_col in enumerate(ax[1, :]):
            ax_col.set_title(column_titles[i], fontsize=30, fontname='serif')
            



        # Adjust layout to avoid overlapping, and position the colorbar externally
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the tight layout to leave space on the right

        # Colorbar with increased fontsize for ticks, placed externally
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height] for the colorbar
        cbar = fig.colorbar(im_train_error, cax=cbar_ax, label='Value')
        cbar.ax.tick_params(labelsize=20)  # Increase font size of colorbar ticks

        plt.savefig(path_image_test61 + f"navier_stokes_test_loss_{test_l2_full/ntest}.png")
        plt.close()

    ttest.append(test_l2_full/ntest)

    t2 = default_timer()
    scheduler.step()
    logging.info(f"epoch : {ep}     time  :  {t2 - t1}     train l2 full:  {train_l2_full / ntrain}     test l2 full:  {test_l2_full / ntest}")



save_results(log_folder, f"Final Test mse for GraphFNO after {epochs} epochs: {test_l2_full / ntest}")
export_excel(os.path.join(log_folder, 'log.txt'), os.path.join(log_folder, f'{dataset_name}_spatial_loss.xlsx'))

logging.info("==================Training finished !==================")

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer
import os
import shutil
from utilities import *
from dataset import Dataset
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected, dense_to_sparse, add_remaining_self_loops
from GraphFNO_Frigate_specformer import GraphFNO
from torch.utils.data import Subset, DataLoader
from datetime import datetime
import inspect
import importlib.util
import sys
import logging


dataset_name = 'darcy'

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


log_folder , path_model, path_image_test61 = create_log_folder()
save_scripts(log_folder)
setup_logging(log_folder)



  
#==================================data preprocessing starts===================================


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

if dataset_name == 'airfoil':

    PATH = "/home/subhankar/data/naca"
    if noise_std is None:
        INPUT_X = PATH + '/NACA_Cylinder_X.npy'
        INPUT_Y = PATH + '/NACA_Cylinder_Y.npy'
        OUTPUT_Sigma = PATH + '/NACA_Cylinder_Q.npy'
        radius_train = 0.055
        radius_test = 0.055
    else:

        INPUT_X = PATH + f'/NACA_Cylinder_X_noised_{noise_std}.npy'
        INPUT_Y = PATH + f'/NACA_Cylinder_Y_noised_{noise_std}.npy'
        OUTPUT_Sigma = PATH + f'/NACA_Cylinder_Q_noised_{noise_std}.npy'
        radius_train = 0.055
        radius_test = 0.055


elif dataset_name == 'elasticity':
    PATH = '../data/elasticity/Meshes'
    OUTPUT_Sigma = PATH+'/Random_UnitCell_sigma_10.npy'
    PATH_XY = PATH+ '/Random_UnitCell_XY_10.npy'
    PATH_rr = PATH+ '/Random_UnitCell_rr_10.npy'
    radius_train = 0.08
    radius_test = 0.08

elif dataset_name == 'pipe':
    PATH = "../data/pipe"
    INPUT_X = PATH +'/Pipe_X.npy'
    INPUT_Y = PATH + '/Pipe_Y.npy'
    OUTPUT_Sigma = PATH + '/Pipe_Q.npy'
    radius_train = 0.08
    radius_test = 0.08
    
elif dataset_name == 'darcy':
    PATH = "../data/darcy/Darcy_421/"
    PATH_TRAIN = PATH +'piececonst_r421_N1024_smooth1.mat'
    PATH_TEST = PATH + 'piececonst_r421_N1024_smooth2.mat'
    radius_train = 0.08
    radius_test = 0.08



ntrain = 1000
ntest = 200
batch_size = 1
# radius_train = 0.06
# radius_test = 0.06

normalized = False
cropped = True
s =[4.0]
g_type = 'knn'
k_train = 21
k_test =  21
# Create dataset and dataloaders
dataset = Dataset(INPUT_X, INPUT_Y, OUTPUT_Sigma, PATH_TRAIN, PATH_TEST, ntrain, ntest, radius_train, k_train, k_test,
                  radius_test,  s, 'GraphFNO', path_xy= PATH_XY, path_rr = PATH_rr,
                    new_chunk_Size=1200, old_chunk_size = 1200,
                    normalized = normalized, cropped =cropped, g_type = 'knn', 
                    dataset_name = dataset_name, noise_std = noise_std , dist_weighted_laplacian = False)

train_indices = [i for i in range(ntrain)]
test_indices = [i for i in range(ntrain, ntrain+ntest)]
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#==========================================================================================================================



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


logging.info(f"device : {device}")  
logging.info(f'batch_size {batch_size}')
learning_rate = 0.001
epochs = 1500
step_size = 100
gamma = 0.65
width = 32
#width = 56
num_wavelet_layers = 6
N = dataset.Ntrain



logging.info("================training started ===================")

logging.info(f"dataset_name : {dataset_name}")
logging.info(f"number of GraphFourier Layer {num_wavelet_layers}" )

model = GraphFNO(num_wavelet_layers, width, N, s, device, dataset).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
# myloss = torch.nn.MSELoss(reduction='sum')

ttrain = np.zeros((epochs, ))
# from torch.autograd.profiler import profile
# with profile(use_cuda=True) as prof:
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0.0
    # breakpoint()
    for i, batch  in enumerate(train_loader):
        batch = Data(**batch)
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)

        # breakpoint()
        # mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1, 1))
        # loss = torch.norm(out.view(-1) - batch.y.view(-1), 1)
        # loss.backward()

        if normalized:
            
            out = dataset.y_normalizer.decode(out.view(batch_size, -1))
            batch.y = dataset.y_normalizer.decode(batch.y.view(batch_size, -1))

        # print(i)
        l2 = myloss(out.view(batch_size, -1), batch.y.view(batch_size, -1))

        l2.backward()
        optimizer.step()
        train_mse += l2.item()

    scheduler.step()
    t2 = default_timer()

    
    ttrain[ep] = train_mse / len(train_loader)
    test_l2_61 = 0.0
    if ep % 1 == 0:
        with torch.no_grad():
            for batch in test_loader:
                batch = Data(**batch)
                batch = batch.to(device)
                out = model(batch)

                if normalized:
                    # breakpoint()
                    out = dataset.y_normalizer.decode(out.view(batch_size, -1))
                    batch.y = dataset.y_normalizer.decode(batch.y.view(batch_size , -1))

                test_l2_61 += myloss(out.view(batch_size, -1),
                                    batch.y.view(batch_size, -1)).item()

        logging.info(f'{ep} time: {t2-t1} train_mse: {train_mse / len(train_loader)} test_mse: {test_l2_61 / len(test_loader)}')
    
    if ep%20 ==0:
        logging.info(f"edge_index : {model.edge_index.shape}")
        logging.info(f"edge_weight min max : {model.edge_weight.min()} {model.edge_weight.max()}")
        
    if ep%20 ==0:

        if dataset_name == 'airfoil' or dataset_name == 'pipe':
            ind = -1
            r1 = 1
            r2 = 1
            s1 = dataset.sx
            s2 = dataset.sy
            if normalized:
                batch.x = dataset.x_normalizer.decode(batch.x)
            x_test = batch.x[ind].view(s1, s2, 2)
            y_test = batch.y[ind].view(s1, s2)
            X = x_test[:, :, 0].squeeze().detach().cpu().numpy()
            Y = x_test[:, :, 1].squeeze().detach().cpu().numpy()
            truth = y_test.squeeze().detach().cpu().numpy()
            pred = out.squeeze().view(batch_size, s1, s2)[ind].detach().cpu().numpy()
            nx = 40 // r1
            ny = 20 // r2

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

            # Determine the color scale range for truth and prediction plots
            vmin = min(truth.min(), pred.min())
            vmax = max(truth.max(), pred.max())

            # Plotting for the entire region
            pic0 = ax[0].pcolormesh(X, Y, truth, cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
            pic1 = ax[1].pcolormesh(X, Y, pred, cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
            pic2 = ax[2].pcolormesh(X, Y, (pred - truth)**2, cmap='viridis', shading='gouraud')

            # Adding colorbars
            cbar0 = fig.colorbar(pic0, ax=ax[0])
            cbar1 = fig.colorbar(pic1, ax=ax[1])
            cbar2 = fig.colorbar(pic2, ax=ax[2])

            # Adding subtitles
            ax[0].set_title('Ground Truth (Small Region)')
            ax[1].set_title('Predicted Values (Small Region)')
            ax[2].set_title('Prediction Error (Small Region)')

            testloss = test_l2_61 / len(test_loader)
            plt.savefig(path_image_test61 + f'test_at_epoch_{ep}_loss_{testloss}.png')
            plt.close()


        if dataset_name == 'elasticity':
            if normalized:
                batch.x = dataset.x_normalizer.decode(batch.x)
            
            mesh = batch.x[-1]
            sigma = batch.y[-1]
            XY = mesh.detach().cpu().numpy()
            truth = sigma.squeeze().detach().cpu().numpy()
            pred = out.view(batch_size, -1, 1)[-1].squeeze().detach().cpu().numpy()

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

            # Scatter plot for Ground Truth
            sc0 = ax[0].scatter(XY[:, 0], XY[:, 1], c=truth, cmap='RdBu_r', edgecolor='w', lw=0.1)
            cbar0 = fig.colorbar(sc0, ax=ax[0])
            cbar0.set_label('Ground Truth')
            ax[0].set_title('Ground Truth')

            # Scatter plot for Predictions
            sc1 = ax[1].scatter(XY[:, 0], XY[:, 1], c=pred, cmap='RdBu_r', edgecolor='w', lw=0.1, vmin=truth.min(), vmax=truth.max())
            cbar1 = fig.colorbar(sc1, ax=ax[1])
            cbar1.set_label('Predicted Values')
            ax[1].set_title('Predicted Values')

            # Scatter plot for Error
            sc2 = ax[2].scatter(XY[:, 0], XY[:, 1], c=(truth - pred)**2, cmap='RdBu_r', edgecolor='w', lw=0.1)
            cbar2 = fig.colorbar(sc2, ax=ax[2])
            cbar2.set_label('Prediction Error')
            ax[2].set_title('Prediction Error')

            testloss = test_l2_61 / len(test_loader)
            plt.savefig(path_image_test61 + f'test_at_epoch_{ep}_loss_{testloss}.png')
            plt.close()
            
        if dataset_name == "darcy":
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
            
            # Generate grid coordinates
            X, Y = torch.meshgrid(torch.arange(dataset.sx), torch.arange(dataset.sy))
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            
            # Get the prediction and ground truth data
            pred = out.view(batch_size, dataset.sx, dataset.sy)[-1].detach().cpu().numpy()
            truth = batch.y[-1].view(dataset.sx, dataset.sy).squeeze().detach().cpu().numpy()
            error = (truth - pred) ** 2

            # Determine color limits based on the truth data
            vmin, vmax = truth.min(), truth.max()

            # Plot prediction
            pic0 = ax[0].pcolormesh(X, Y, pred, cmap='RdBu_r', shading='gouraud', vmin=vmin, vmax=vmax)
            cbar0 = fig.colorbar(pic0, ax=ax[0])
            cbar0.set_label('Predicted Values')
            ax[0].set_title('Predicted Values')

            # Plot ground truth
            pic1 = ax[1].pcolormesh(X, Y, truth, cmap='RdBu_r', shading='gouraud', vmin=vmin, vmax=vmax)
            cbar1 = fig.colorbar(pic1, ax=ax[1])
            cbar1.set_label('Ground Truth')
            ax[1].set_title('Ground Truth')

            # Plot error
            pic2 = ax[2].pcolormesh(X, Y, error, cmap='RdBu_r', shading='gouraud')
            cbar2 = fig.colorbar(pic2, ax=ax[2])
            cbar2.set_label('Prediction Error')
            ax[2].set_title('Prediction Error')

            # Set overall title and save the figure
            plt.suptitle(f'Darcy Flow Predictions at Epoch {ep}')
            testloss = test_l2_61 / len(test_loader)
            plt.savefig(path_image_test61 + f'test_at_epoch_{ep}_loss_{testloss:.4f}.png')
            plt.close()


save_results(log_folder, f"Final Test mse for GraphFNO after {epochs} epochs: {test_l2_61 / ntest}")

logging.info("==================Training finished !==================")
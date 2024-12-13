import torch
import numpy as np
import matplotlib as mpl
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer
import os
import shutil
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected, dense_to_sparse, add_remaining_self_loops
from models.elasticity.Sp2GNO_frigate import GraphFNO
from src.utilities import *
from src.dataset import Dataset
from torch.utils.data import Subset, DataLoader
from datetime import datetime
import inspect
import importlib.util
import sys
import logging
import random
import argparse
from src.save_loss_excel import export_excel

dataset_name = 'elasticity'

def adjust_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    logging.info(f"Loaded checkpoint at epoch {start_epoch}. Learning rate manually set to {new_lr}.")

def load_checkpoint(model, optimizer, scheduler1, scheduler2, scheduler3, new_lr = None ):
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        

        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1  
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Loaded optimizer state. Current learning rate is {current_lr}")
            num_adjust_lr = start_epoch // 100
            print(f"Loaded optimizer state. learning previously at epoch {start_epoch} was {0.001*(0.65)**num_adjust_lr}")

        else:
            print("Optimizer state not found, initializing optimizer from scratch.")

        if not new_lr is None:
            adjust_lr(optimizer, new_lr)

        if 'scheduler_state_dict' in checkpoint:
            if start_epoch <200:
                scheduler1.load_state_dict(checkpoint['scheduler_state_dict'])
            elif 200<= start_epoch < 350:
                scheduler2.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler3.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state.")
        else:
            print("Scheduler state not found, initializing scheduler from scratch.")

        
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")
        start_epoch = 0
    return start_epoch

def setup_logging(log_folder, params, log_level=logging.INFO):
    # Configure logging
    logging.basicConfig(
        level=log_level, 

        format='%(message)s',
        filename=os.path.join(log_folder, f"log_{params}.txt"), 
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO) 
    logging.getLogger('').addHandler(console)


# Define a function to create a folder with a timestamp
def create_log_folder(exp_name, params):
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_folder = f"./log_folder/training_logs_"+exp_name
    os.makedirs(log_folder, exist_ok=True)
    path = f'./{log_folder}/GraphFNO_{dataset_name}_Results/'
    path_model =  path+f'model_irregular_{params}/'
    path_image_test61 = path +f'image_irregular_{params}/'
    os.makedirs(path_model,exist_ok=True)
    os.makedirs(path_image_test61,exist_ok=True)

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


def save_results(log_folder, results,):
    with open(os.path.join(log_folder, "results.txt"), "w") as results_file:
        results_file.write(results)
        results_file.write("\n\nTraining Loop Information:\n")
        results_file.write("\n")






# Add argument parsing
parser = argparse.ArgumentParser(description='Ablation Study Script')

# Define parameters with default values
parser.add_argument('--exp_name', type=str, default='elasticity_varing_m', help='Name of the experiment (str)')
parser.add_argument('--num_wavelet_layers', type=int, default=4, help='Number of wavelet layers (default: 4)')
parser.add_argument('--width', type=int, default=32, help='Width parameter (default: 32)')
parser.add_argument('--k', type=int, default=20, help='N parameter (default: 100)')
parser.add_argument('--num_of_frequency_to_learn', type=int, default=48, help='Number of frequencies to learn (default: 48)')

# Parse arguments
args = parser.parse_args()




# Assign to variables for easier use in your script
num_wavelet_layers = args.num_wavelet_layers
width = args.width
k = args.k
num_of_frequency_to_learn = args.num_of_frequency_to_learn
exp_name = args.exp_name
params = f'layers_{num_wavelet_layers}_width_{width}_k_{k}_m_{num_of_frequency_to_learn}'

log_folder , path_model, path_image_test61 = create_log_folder(exp_name, params)
save_scripts(log_folder)
setup_logging(log_folder, params )

  
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

PATH = '../data/elasticity/elasticity/Meshes'
OUTPUT_Sigma = PATH+'/Random_UnitCell_sigma_10.npy'
PATH_XY = PATH+ '/Random_UnitCell_XY_10.npy'
PATH_rr = PATH+ '/Random_UnitCell_rr_10.npy'
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



# Create dataset and dataloaders
dataset = Dataset(INPUT_X, INPUT_Y, OUTPUT_Sigma, PATH_TRAIN, PATH_TEST, ntrain, ntest, radius_train, k, 
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
device = 'cpu'
logging.info(f"device : {device}")  
logging.info(f'batch_size {batch_size}')
# learning_rate = 0.001
learning_rate = 0.01
epochs = 800


#====================== optimizable parameters======================
# num_wavelet_layers = 4
# width = 32
# num_of_frequency_to_learn = 48
# step_size = 100
# gamma = 0.65
# B = 1


step_size = 25
gamma = 0.5
B = 5
checkpoint_dir = './checkpts/elasticity/_'
checkpoint_file = os.path.join(checkpoint_dir, 'elas_model.pth')
#===================================================================

N = dataset.N



logging.info("================training started ===================")

logging.info(f"dataset_name : {dataset_name}")
logging.info(f"number of GraphFourier Layer {num_wavelet_layers}" )

model = GraphFNO(num_wavelet_layers, width, N, num_of_frequency_to_learn, device, dataset).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.75)

scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.65)

start_epoch = load_checkpoint(model, optimizer, scheduler1, scheduler2, scheduler3)




myloss = LpLoss(size_average=False)
# myloss = torch.nn.MSELoss(reduction='sum')

ttrain = np.zeros((epochs, ))
# from torch.autograd.profiler import profile
# with profile(use_cuda=True) as prof:
for ep in range(start_epoch,epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0.0
    batch_loss = 0.0
    # breakpoint()
    for i, batch in enumerate(train_loader):
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


        l2 = myloss(out.view(batch_size, -1), batch.y.view(batch_size, -1))
        # l2.backward()
        # optimizer.step()
        batch_loss += l2
        if (i+1)%B==0:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss = 0.0

        train_mse += l2.item()

    if ep < 200:
        scheduler1.step()
    elif 200 <= ep < 350:
        scheduler2.step()
    else:
        scheduler3.step()
        
    t2 = default_timer()

    
    ttrain[ep] = train_mse / len(train_loader)
    test_l2_61 = 0.0
    plot_data = []
    pred_times = []
    if ep % 1 == 0:
        with torch.no_grad():
            for batch in test_loader:
                data={}
                batch = Data(**batch)
                batch = batch.to(device)
                t1 = time.perf_counter()
                out = model(batch)
                t2 = time.perf_counter()
                pred_times.append(t2-t1)
                data['x'] = batch.x
                data['y'] = batch.y
                data['out'] = out
                plot_data.append(data)

                if normalized:
                    # breakpoint()
                    out = dataset.y_normalizer.decode(out.view(batch_size, -1))
                    batch.y = dataset.y_normalizer.decode(batch.y.view(batch_size , -1))

                test_l2_61 += myloss(out.view(batch_size, -1),
                                    batch.y.view(batch_size, -1)).item()
        
        logging.info(f'avg prediction time in {device} is {np.mean(pred_times)}')

        # logging.info(f'{ep} time: {t2-t1} train_mse: {train_mse / len(train_loader)} test_mse: {test_l2_61 / len(test_loader)}')
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'{ep} time: {t2-t1} train_mse: {train_mse / len(train_loader)} test_mse: {test_l2_61 / len(test_loader)} lr: {current_lr}')
    
    # if ep%20 ==0:
    #     logging.info(f"edge_index : {model.edge_index.shape}")
    #     logging.info(f"edge_weight min max : {model.edge_weight.min()} {model.edge_weight.max()}")
    
    # if ep >= 1500 and ep % 500 == 0:
    if ep >= 0 and ep % 20 == 0:
        if ep <200:
            scedlr = scheduler1
        elif 200<=ep <350:
            scedlr = scheduler2
        else:
            scedlr = scheduler3
        save_dict = {'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':  scedlr.state_dict()}
        torch.save(save_dict, os.path.join(path_model, f'elas_model.pth'))
        logging.info(f'Model saved at epoch {ep}')
        
    if ep%20 ==0:

        if dataset_name == 'airfoil' or dataset_name == 'pipe':
            indices = random.sample(range(ntest), 4)

            r1 = 1
            r2 = 1
            s1 = dataset.sx
            s2 = dataset.sy

            fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(25, 15))

            for col, ind in enumerate(indices):
                # Decode normalized data if necessary
                if normalized:
                    batch.x = dataset.x_normalizer.decode(batch.x)

                # Reshape test sample data
                x_test = plot_data[ind]['x'].view(s1, s2, 2)
                y_test = plot_data[ind]['y'].view(s1, s2)
                X = x_test[:, :, 0].squeeze().detach().cpu().numpy()
                Y = x_test[:, :, 1].squeeze().detach().cpu().numpy()
                truth = y_test.squeeze().detach().cpu().numpy()
                pred = plot_data[ind]['out'].view(s1, s2).detach().cpu().numpy()
                
                # Determine color scale range based on the min and max values of truth and prediction
                vmin = min(truth.min(), pred.min())
                vmax = max(truth.max(), pred.max())
                
                # Plot ground truth
                pic0 = ax[0, col].pcolormesh(X, Y, truth, cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
                cbar0 = fig.colorbar(pic0, ax=ax[0, col])
                # Plot predicted values
                pic1 = ax[1, col].pcolormesh(X, Y, pred, cmap='viridis', shading='gouraud', vmin=vmin, vmax=vmax)
                cbar1 = fig.colorbar(pic1, ax=ax[1, col])
                # Plot prediction error
                pic2 = ax[2, col].pcolormesh(X, Y, (pred - truth)**2, cmap='viridis', shading='gouraud')
                cbar2 = fig.colorbar(pic2, ax=ax[2, col])

            # Set column-wise titles
            for col, ind in enumerate(indices):
                ax[0, col].set_title(f'Test Example: {col}', fontsize=20, fontweight='bold')

            # Set row-wise titles
            row_titles = ['Truth', 'Prediction', 'Error']
            for row, title in enumerate(row_titles):
                ax[row, 0].set_ylabel(title, fontsize=20, rotation=90, fontweight='bold')

            # Adjust layout and save the figure
            plt.tight_layout()
            testloss = test_l2_61 / len(test_loader)
            plt.savefig(path_image_test61 + f'test_at_epoch_{ep}_loss_{testloss:.4f}.png')
            plt.close()


        if dataset_name == 'elasticity':
            
            indices = random.sample(range(ntest), 4)

            fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(30, 25))

            for col, ind in enumerate(indices):
                # Decode normalized data if necessary
                if normalized:
                    batch.x = dataset.x_normalizer.decode(batch.x)

                mesh = plot_data[ind]['x'].squeeze(0)
                sigma = plot_data[ind]['y']
                out = plot_data[ind]['out']
                XY = mesh.detach().cpu().numpy()
                
                truth = sigma.squeeze().detach().cpu().numpy()
                pred = out.view(batch_size, -1, 1).squeeze().detach().cpu().numpy()
                
                sc0 = ax[0, col].scatter(XY[:, 0], XY[:, 1], c='grey', edgecolor='w', lw=0.1)
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                empty_cbar = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
                empty_cbar.set_array([])
                fig.colorbar(empty_cbar, ax=ax[0, col]).remove()
                
                

                # breakpoint()
                # Scatter plot for Ground Truth
                sc1 = ax[1, col].scatter(XY[:, 0], XY[:, 1], c=truth, cmap='RdBu_r', edgecolor='w', lw=0.1)
                cbar1 = fig.colorbar(sc1, ax=ax[1, col])
                cbar1.ax.tick_params(labelsize=20)  #
                # cbar0.set_label('Ground Truth', fontsize=14, fontweight='bold')
                # ax[0, col].set_title(f'Index: {ind}', fontsize=14, fontweight='bold')

                # Scatter plot for Predictions
                sc2 = ax[2, col].scatter(XY[:, 0], XY[:, 1], c=pred, cmap='RdBu_r', edgecolor='w', lw=0.1, vmin=truth.min(), vmax=truth.max())
                cbar2 = fig.colorbar(sc2, ax=ax[2, col])
                cbar2.ax.tick_params(labelsize=20)  #
                # cbar1.set_label('Predicted Values', fontsize=14, fontweight='bold')

                # Scatter plot for Error
                sc3 = ax[3, col].scatter(XY[:, 0], XY[:, 1], c= abs(truth - pred), cmap='RdBu_r', edgecolor='w', lw=0.1)
                # sc3 = ax[3, col].scatter(XY[:, 0], XY[:, 1], c=(truth - pred)**2, cmap='RdBu_r', edgecolor='w', lw=0.1)
                cbar3 = fig.colorbar(sc3, ax=ax[3, col])
                cbar3.ax.tick_params(labelsize=20)  #
                # cbar2.set_label('Prediction Error', fontsize=14, fontweight='bold')
                ax[0, col].set_xticks([])
                ax[0, col].set_yticks([])
                ax[1, col].set_xticks([])
                ax[1, col].set_yticks([])
                ax[2, col].set_xticks([])
                ax[2, col].set_yticks([])
                ax[3, col].set_xticks([])
                ax[3, col].set_yticks([])


            # Set column-wise titles
            for col, ind in enumerate(indices):
                pass
                # ax[0, col].set_title(f'Test Example: {col}', fontsize=20, fontweight='bold')

            # Set row-wise titles
            row_titles = ['Input','Truth', 'Prediction', 'Error']
            for row, title in enumerate(row_titles):
                ax[row, 0].set_ylabel(title, fontdict={'fontsize': 30, 'fontweight': 'bold', 'fontname': 'serif'}, rotation=90)

            # Adjust layout and save the figure
            plt.tight_layout()
            testloss = test_l2_61 / len(test_loader)
            plt.savefig(path_image_test61 + f'test_at_epoch_{ep}_loss_{testloss:.4f}.png')
            plt.close()
            
        if dataset_name == "darcy":
            indices = random.sample(range(ntest), 4)

            fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(25, 15))
            X, Y = torch.meshgrid(torch.arange(dataset.sx), torch.arange(dataset.sy))
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()

            for col, ind in enumerate(indices):
                
                # Get the prediction and ground truth data
                out = plot_data[ind]['out']
                pred = out.view(batch_size, dataset.sx, dataset.sy).squeeze(0).detach().cpu().numpy()
                truth = plot_data[ind]['y'].view(dataset.sx, dataset.sy).squeeze().detach().cpu().numpy()
                error = (truth - pred) ** 2

                # Determine color limits based on the truth data
                vmin, vmax = truth.min(), truth.max()

                # Plot prediction
                pic0 = ax[0, col].pcolormesh(X, Y, truth, cmap='RdBu_r', shading='gouraud', vmin=vmin, vmax=vmax)
                cbar0 = fig.colorbar(pic0, ax=ax[0, col])
                # cbar0.set_label('Predicted Values', fontsize=14, fontweight='bold')
                # ax[0, col].set_title(f'Index: {ind}', fontsize=14, fontweight='bold')

                # Plot ground truth
                pic1 = ax[1, col].pcolormesh(X, Y, pred, cmap='RdBu_r', shading='gouraud', vmin=vmin, vmax=vmax)
                cbar1 = fig.colorbar(pic1, ax=ax[1, col])
                # cbar1.set_label('Ground Truth', fontsize=14, fontweight='bold')
                # ax[1, col].set_title('Ground Truth', fontsize=14, fontweight='bold')

                # Plot error
                pic2 = ax[2, col].pcolormesh(X, Y, error, cmap='RdBu_r', shading='gouraud')
                cbar2 = fig.colorbar(pic2, ax=ax[2, col])
                # cbar2.set_label('Prediction Error', fontsize=14, fontweight='bold')
                # ax[2, col].set_title('Prediction Error', fontsize=14, fontweight='bold')

            # Set column-wise titles
            for col, ind in enumerate(indices):
                ax[0, col].set_title(f'Test Example: {col}', fontsize=20, fontweight='bold')

            # Set row-wise titles
            row_titles = ['Ground Truth', 'Prediction' , 'Error']
            for row, title in enumerate(row_titles):
                ax[row, 0].set_ylabel(title, fontsize=20, rotation=90, fontweight='bold')

            # Set overall title
            plt.suptitle(f'Darcy Flow Predictions at Epoch {ep}', fontsize=16, fontweight='bold')

            # Adjust layout and save the figure
            plt.tight_layout()
            testloss = test_l2_61 / len(test_loader)
            plt.savefig(path_image_test61 + f'test_at_epoch_{ep}_loss_{testloss:.4f}.png')
            plt.close()


save_results(log_folder, f"Final Test mse for GraphFNO after {epochs} epochs: {test_l2_61 / ntest}")
export_excel(os.path.join(log_folder, 'log.txt'), os.path.join(log_folder, f'{dataset_name}_sp2gno_loss.xlsx'))

logging.info("==================Training finished !==================")
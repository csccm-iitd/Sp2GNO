import numpy as np
import torch
from noise import *
from utilities3 import MatReader
from scipy.io import savemat

ntrain = 1000
ntest = 200
S = 64
T_in = 10
T_out = 10
step = 1
r= 1
std = 0.006

def get_grid( shape):

    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1)



pos = get_grid([1,S,S]).squeeze(0).flatten(0,1).numpy()
same_transformed_pos = add_gaussian_noise(pos,  std)
PATH = "/home/subhankar/data/navier_stokes/"
TRAIN_PATH = PATH + "NavierStokes_V1e-5_N1200_T20.mat"
TEST_PATH =  PATH + "NavierStokes_V1e-5_N1200_T20.mat"

reader = MatReader(TRAIN_PATH)
total_u = reader.read_field('u')[:ntrain+ntest].flatten(1,2).numpy()
transformed_total_u = transform_and_interpolate_ns(pos, total_u, same_transformed_pos)

data_to_save = {
    'u': torch.tensor(transformed_total_u).permute(0,2,1).view(-1,S, S, 20), 
    'pos' : torch.tensor(same_transformed_pos).view(S, S, 2)
}

savemat(PATH + f'NavierStokes_V1e-5_N1200_T20_noised_{std}.mat', data_to_save)

print("data saved !")

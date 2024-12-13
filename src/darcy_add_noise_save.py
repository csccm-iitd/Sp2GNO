import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from utilities3 import *
from noise import transform_and_interpolate_darcy, plot, add_gaussian_noise

PATH = "/home/subhankar/data/darcy/Darcy_421/"
PATH_TRAIN = PATH +'piececonst_r421_N1024_smooth1.mat'
PATH_TEST = PATH + 'piececonst_r421_N1024_smooth2.mat'


def get_grid(shape):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1)

std = 0.008
ntrain = 1200
ntest = 300
same_grid = True
r = 5
reader = MatReader(PATH_TRAIN)
train_a = reader.read_field('coeff')[:ntrain,::r,::r]
train_u = reader.read_field('sol')[:ntrain,::r,::r]

# loading test data
reader.load_file(PATH_TEST)
test_a = reader.read_field('coeff')[:ntest,::r,::r]
test_u = reader.read_field('sol')[:ntest,::r,::r]
pos = get_grid(train_a.shape)[0].flatten(0,1).numpy()


if not same_grid:
    pos_train = get_grid(train_a.shape).flatten(1,2).numpy()
else:
    pos_train = pos
y_train = train_u.flatten(1,2)
x_train = train_a.flatten(1,2)

if not same_grid:
    pos_test = get_grid(test_a.shape).flatten(1,2).numpy()
else:
    pos_test = pos
y_test = test_u.flatten(1,2)
x_test = test_a.flatten(1,2)

sx = 85
sy = 85
if same_grid:
    same_transformed_pos = add_gaussian_noise(pos,  std)
else:
    same_transformed_pos = None
    
transformed_pos_train, transformed_y_train, transformed_a_train = transform_and_interpolate_darcy(
    pos_train, y_train, x_train,same_transformed_pos, same_grid , std)

transformed_pos_test, transformed_y_test , transformed_a_test = transform_and_interpolate_darcy(
    pos_test, y_test, x_test, same_transformed_pos, same_grid, std)

data_to_save_train = {
    'pos': torch.tensor(transformed_pos_train).view(-1,sx,sy,2),
    'coeff': torch.tensor(transformed_a_train).view(-1, sx, sy),
    'sol': torch.tensor(transformed_y_train).view(-1, sx, sy)
}
data_to_save_test = {
    'pos': torch.tensor(transformed_pos_test).view(-1, sx, sy,2),
    'coeff': torch.tensor(transformed_a_test).view(-1, sx, sy),
    'sol': torch.tensor(transformed_y_test).view(-1, sx, sy)
}
savemat(PATH + f'piececonst_r421_N1024_smooth1_noised_{std}.mat', data_to_save_train)
savemat(PATH + f'piececonst_r421_N1024_smooth2_noised_{std}.mat', data_to_save_test)


print("data saved !")


from noise import transform_and_interpolate_airfoil, plot
import torch
import torch
import numpy as np
import matplotlib.pyplot as plt


PATH = "/home/subhankar/data/naca"
INPUT_X = PATH + '/NACA_Cylinder_X.npy'
INPUT_Y = PATH + '/NACA_Cylinder_Y.npy'
OUTPUT_Sigma = PATH + '/NACA_Cylinder_Q.npy'
cropped = True
nx= 15
ny =12
N_samples = 1400
std = 0.006


pos_x = torch.tensor(np.load(INPUT_X), dtype=torch.float)
pos_y = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
y = torch.tensor(np.load(OUTPUT_Sigma), dtype=torch.float)[:, 4]
print("pos_x.shape",pos_x.shape )
print("pos_y.shape",pos_y.shape )
print("y.shape",y.shape )
if cropped:
    pos_unflatten = torch.stack([pos_x, pos_y], dim=-1)[:N_samples][:, nx:-nx, :-ny]
    pos = torch.stack([pos_x, pos_y], dim=-1)[:N_samples][:, nx:-nx, :-ny].flatten(1,2)
    y = y[:N_samples][:, nx:-nx, :-ny].flatten(1,2)
    sx = pos_unflatten.shape[1]
    sy = pos_unflatten.shape[2]
else:
    pos_unflatten = torch.stack([pos_x, pos_y], dim=-1)[:N_samples]
    pos = torch.stack([pos_x, pos_y], dim=-1)[:N_samples].flatten(1,2)
    y = y[:N_samples].flatten(1,2)
    sx = pos_unflatten.shape[1]
    sy = pos_unflatten.shape[2]

transformed_pos, transformed_y = transform_and_interpolate_airfoil(pos, y, noise_std = std)
transformed_xx = transformed_pos[:, :, 0].view(-1, sx, sy)
transformed_yy = transformed_pos[:, :, 1].view(-1, sx, sy)
transformed_y = transformed_y.view(-1, sx, sy)
print("transformed xx shape", transformed_xx.shape)
print("transformed yy shape", transformed_yy.shape)
print("transformed y shape", transformed_y.shape)
np.save(PATH + f'/NACA_Cylinder_X_noised_{std}.npy', transformed_xx)
np.save(PATH + f'/NACA_Cylinder_Y_noised_{std}.npy', transformed_yy)
np.save(PATH + f'/NACA_Cylinder_Q_noised_{std}.npy', transformed_y)
print("data saved !")
from noise import transform_and_interpolate_pipe, plot
import torch
import torch
import numpy as np
import matplotlib.pyplot as plt



PATH = "/home/subhankar/data/pipe"
INPUT_X = PATH +'/Pipe_X.npy'
INPUT_Y = PATH + '/Pipe_Y.npy'
OUTPUT_Sigma = PATH + '/Pipe_Q.npy'
cropped = True
nx= 15
ny =12
N_samples = 1400
std = 0.01

pos_x = torch.tensor(np.load(INPUT_X), dtype=torch.float)
pos_y = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
y = torch.tensor(np.load(OUTPUT_Sigma), dtype=torch.float)[:, 0]
print("pos_x.shape",pos_x.shape )
print("pos_y.shape",pos_y.shape )
print("y.shape",y.shape )

pos_unflatten = torch.stack([pos_x, pos_y], dim=-1)[:N_samples]
pos = torch.stack([pos_x, pos_y], dim=-1)[:N_samples].flatten(1,2).numpy()
y = y[:N_samples].flatten(1,2).numpy()
sx = pos_unflatten.shape[1]
sy = pos_unflatten.shape[2]

transformed_pos, transformed_y = transform_and_interpolate_pipe(pos, y, noise_std = std)
# breakpoint()
transformed_xx = torch.tensor(transformed_pos[:, :, 0]).view(-1, sx, sy)
transformed_yy = torch.tensor(transformed_pos[:, :, 1]).view(-1, sx, sy)
transformed_y = torch.tensor(transformed_y).view(-1, sx, sy)
print("transformed xx shape", transformed_xx.shape)
print("transformed yy shape", transformed_yy.shape)
print("transformed y shape", transformed_y.shape)
np.save(PATH + f'/Pipe_X_noised_{std}.npy', transformed_xx)
np.save(PATH + f'/Pipe_Y_noised_{std}.npy', transformed_yy)
np.save(PATH + f'/Pipe_Q_noised_{std}.npy', transformed_y)
print("data saved !")
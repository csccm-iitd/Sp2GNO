import numpy as np
import torch
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp2d
from tqdm import tqdm

def is_on_boundary(point_index, vertex_coordinates, neighborhood_indices):
    point = vertex_coordinates[point_index]
    if 0<len(neighborhood_indices[point_index]) < 7 or (-0.1 <= point[0] <= 1.0 and -0.15 <= point[1] <= 0.15) or (1.8 <= point[0] <= 2.2 and -0.375 <= point[1] <= 0.375):
        return True  # Boundary point
    else:
        return False

def add_noise_to_pos(pos, std=0.01):
    """
    Add zero mean unit Gaussian noise to all points except the boundary points.
    
    Parameters:
        pos (numpy.ndarray): Tensor of shape (N, 2) containing coordinates of points.
        std_dev (float): Standard deviation of the Gaussian noise.
    
    Returns:
        numpy.ndarray: Transformed pos tensor with noise added.
    """
    transformed_pos = pos.clone()
    # print("input pos shape", transformed_pos.shape)

    vertex_coordinates = transformed_pos
    # Set the neighborhood radius
    neighborhood_radius = 0.053  # Adjust this radius as needed

    # Calculate neighborhood indices for each point
    # distances = np.sqrt(np.sum((vertex_coordinates[:, np.newaxis] - vertex_coordinates) ** 2, axis=2))
    # print(vertex_coordinates.shape)
    distances = torch.cdist(vertex_coordinates, vertex_coordinates)
    neighborhood_indices = [np.where(distances[i] <= neighborhood_radius)[0] for i in range(len(vertex_coordinates))]

    # Find boundary vertices within the neighborhood
    boundary_vertices = [i for i in range(len(vertex_coordinates)) if is_on_boundary(i, vertex_coordinates, neighborhood_indices)]

    # Add noise to all points except the boundary points
    for i in range(len(vertex_coordinates)):
        if i not in boundary_vertices:
            noise = np.random.normal(loc=0, scale=std, size=2)
            transformed_pos[i] += noise
    
    return transformed_pos


def add_gaussian_noise(pos, noise_std):
    boundary_vertices = {}
    x_max = np.max(pos[:, 0])
    x_min = np.min(pos[:, 0])
    
    for i in range(pos.shape[0]):
        x, y = pos[i]
        if x == x_max or x == x_min:
            if x not in boundary_vertices:
                boundary_vertices[x] = [(x, y)]
            else:
                boundary_vertices[x].append((x, y))
        else:
            if x not in boundary_vertices:
                boundary_vertices[x] = [(x, np.min(pos[pos[:, 0] == x, 1])), (x, np.max(pos[pos[:, 0] == x, 1]))]

    noise = np.random.normal(loc=0, scale=noise_std, size=(pos.shape[0], 2))
    transformed_pos = pos + noise

    for x, points in boundary_vertices.items():
        for point in points:
            idx = np.where((pos[:, 0] == point[0]) & (pos[:, 1] == point[1]))[0]
            transformed_pos[idx] = pos[idx]

    # Ensure no coordinates exceed 1
    for i in range(transformed_pos.shape[0]):
        if transformed_pos[i, 0] > 1 or transformed_pos[i, 1] > 1:
            transformed_pos[i] = pos[i]

    return transformed_pos


# def add_gaussian_noise(pos, noise_std):

#     boundary_vertices = {}
#     x_max = np.max(pos[:, 0])
#     x_min = np.min(pos[:, 0])
#     for i in range(pos.shape[0]):
#         x = pos[i, 0]
#         y = pos[i, 1]
#         if x == x_max:
#             if x not in boundary_vertices:
#                 boundary_vertices[x] = [(x, y)]
#             else:
#                 boundary_vertices[x].append((x, y))
#         elif x == x_min:
#             if x not in boundary_vertices:
#                 boundary_vertices[x] = [(x, y)]
#             else:
#                 boundary_vertices[x].append((x, y))
#         else:
#             if x not in boundary_vertices:
#                 boundary_vertices[x] = [(x, np.min(pos[pos[:, 0] == x, 1])), (x, np.max(pos[pos[:, 0] == x, 1]))]
    
#     noise = np.random.normal(loc=0, scale=noise_std, size=(pos.shape[0], 2))
#     transformed_pos = pos + noise
    
#     for x, points in boundary_vertices.items():
#         for point in points:
#             idx = np.where((pos[:, 0] == point[0]) & (pos[:, 1] == point[1]))
#             transformed_pos[idx] = pos[idx]
    
#     return transformed_pos


def interpolate_y_griddata(pos, y, transformed_pos):
    transformed_y = griddata(pos, y, transformed_pos, method='linear')
    transformed_y = torch.Tensor(transformed_y)
    return transformed_y
def interpolate_y_linear(pos, y, transformed_pos):
    interpolator = LinearNDInterpolator(pos, y)
    transformed_y = interpolator(transformed_pos)
    transformed_y = torch.Tensor(transformed_y)
    return transformed_y
def interpolate_y_rbf(pos, y, transformed_pos):
    # Use radial basis function interpolation
    rbf_interp = Rbf(pos[:, 0], pos[:, 1], y.flatten(), method='linear')
    transformed_y = rbf_interp(transformed_pos[:, 0], transformed_pos[:, 1])
    transformed_y = torch.Tensor(transformed_y)
    return transformed_y

def interpolate_y_clough_tocher(pos, y, transformed_pos):
    interpolator = CloughTocher2DInterpolator(pos, y)
    transformed_y = interpolator(transformed_pos)
    transformed_y = torch.Tensor(transformed_y)
    return transformed_y



def bilinear_interpolation(pos, z, pos_new):
    interp_func = interp2d(pos[:, 0], pos[:, 1], z, kind='linear')
    return interp_func(pos_new[:, 0], pos_new[:, 1])


def akima_interpolation(pos, z, pos_new):
    interp_func = interp2d(pos[:, 0], pos[:, 1], z, kind='cubic')
    return interp_func(pos_new[:, 0], pos_new[:, 1])


def thin_plate_spline_interpolation(pos, z, pos_new):
    rbf = Rbf(pos[:, 0], pos[:, 1], z, function='thin_plate')
    return rbf(pos_new[:, 0], pos_new[:, 1])

#rbf # thinplate spline # bilinear is working

def transform_and_interpolate_airfoil(pos, y, noise_std=0.01):
    transformed_pos_list = []
    transformed_y_list = []
    
    for batch in range(pos.shape[0]):
        transformed_pos = add_noise_to_pos(pos[batch], noise_std)
        # Example usage of other interpolation methods:
        # transformed_y = interpolate_y_rbf(pos[batch], y[batch], transformed_pos)
        transformed_y = interpolate_y_griddata(pos[batch], y[batch], transformed_pos)
        # transformed_y = interpolate_y_linear(pos[batch], y[batch], transformed_pos)
        #transformed_y = thin_plate_spline_interpolation(pos[batch], y[batch], transformed_pos)
        # transformed_y = akima_interpolation(pos[batch], y[batch], transformed_pos)
        # transformed_y = interpolate_y_clough_tocher(pos[batch], y[batch], transformed_pos)
        
        # Check for NaN values in transformed_y
        if np.isnan(transformed_y).any():
            # Handle NaN values here, such as by discarding the corresponding data point
            print(f"Warning: NaN values found in batch {batch}. Discarding the corresponding data point.")
            plot1(pos[batch], y[batch], transformed_pos, transformed_y, batch)
            continue
        
        transformed_pos_list.append(transformed_pos)
        transformed_y_list.append(transformed_y)
    
    transformed_pos = torch.stack(transformed_pos_list, dim=0)
    transformed_y = torch.stack(transformed_y_list, dim=0)
    
    return transformed_pos, transformed_y


def transform_and_interpolate_pipe(pos, y, noise_std):
    transformed_pos_list = []
    transformed_y_list = []
    for batch in range(pos.shape[0]):
        transformed_pos = add_gaussian_noise(pos[batch], noise_std = 0.01)
        # transformed_y = interpolate_y_rbf(pos[batch], y[batch], transformed_pos)
        # transformed_y = interpolate_y_griddata(pos[batch], y[batch], transformed_pos)
        # transformed_y = interpolate_y_linear(pos[batch], y[batch], transformed_pos)
        transformed_y = interpolate_y_clough_tocher(pos[batch], y[batch], transformed_pos)
        transformed_pos_list.append(transformed_pos)
        transformed_y_list.append(transformed_y)
    transformed_pos = torch.stack(transformed_pos_list, dim = 0)
    transformed_y = torch.stack(transformed_y_list, dim = 0)
    return transformed_pos, transformed_y


def transform_and_interpolate_darcy(pos, y, coeffs, same_transformed_pos, same_grid, noise_std):
    transformed_pos_list = []
    transformed_y_list = []
    transformed_coeffs_list = []
    
    for batch in tqdm(range(coeffs.shape[0])):
        if not same_grid:
            transformed_pos = add_gaussian_noise(pos[batch], noise_std = noise_std)

        # transformed_y = interpolate_y_rbf(pos[batch], y[batch], transformed_pos)
        # transformed_y = interpolate_y_griddata(pos[batch], y[batch], transformed_pos)
        # transformed_y = interpolate_y_linear(pos[batch], y[batch], transformed_pos)
        if not same_grid:
            transformed_coeffs = interpolate_y_clough_tocher(pos[batch], coeffs[batch], transformed_pos)
        else:
            transformed_coeffs = interpolate_y_clough_tocher(pos, coeffs[batch], same_transformed_pos)
            
        if not same_grid:
            transformed_y = interpolate_y_clough_tocher(pos[batch], y[batch], transformed_pos)
        else:
            transformed_y = interpolate_y_clough_tocher(pos, y[batch], same_transformed_pos)
            
        nan_count_y = np.isnan(transformed_y).sum()
        nan_count_coeffs = np.isnan(transformed_coeffs).sum()
            
        if np.isnan(transformed_y).any() or np.isnan(transformed_coeffs).any() :
            print(f"Batch {batch}: {nan_count_y} NaN values in 'transformed_y', {nan_count_coeffs} NaN values in 'transformed_coeffs'.")
            
            # Replace NaN values with zero
            transformed_y = np.nan_to_num(transformed_y, nan=0.0)
            transformed_coeffs = np.nan_to_num(transformed_coeffs, nan=0.0)
            print(f"NaN values replaced with zero.", np.isnan(transformed_y).any(), np.isnan(transformed_coeffs).any())
            # Handle NaN values here, such as by discarding the corresponding data point
            print(f"Warning: NaN values found in batch {batch}. Discarding the corresponding data point.")
            # plot1(pos, y[batch], same_transformed_pos, transformed_y, batch)
            # plot1(pos, coeffs[batch], same_transformed_pos, transformed_coeffs, batch)


        if not same_grid:
            transformed_pos_list.append(transformed_pos)
        transformed_y_list.append(transformed_y)
        transformed_coeffs_list.append(transformed_coeffs)
        
    if not same_grid:   
        transformed_pos = np.stack(transformed_pos_list, axis = 0)
    transformed_y = np.stack(transformed_y_list, axis = 0)
    transformed_coeff = np.stack(transformed_coeffs_list, axis = 0)
    if same_grid:
        return same_transformed_pos, transformed_y , transformed_coeff
    else:
        return transformed_pos, transformed_y, transformed_coeff


def transform_and_interpolate_ns(pos, u, same_transformed_pos ):
    transformed_u_list = []

    for batch in tqdm(range(u.shape[0])):
        batch_u = []
        for frame in range(u.shape[-1]):
            transformed_u = interpolate_y_clough_tocher(pos, u[batch, :, frame], same_transformed_pos)
            batch_u.append(transformed_u)

        batch_array = np.stack(batch_u, axis = 0)    
        nan_count_u = np.isnan(batch_array).sum()
            
        if np.isnan(batch_array).any() :
            
            print(f"Batch {batch}: {nan_count_u} NaN values in 'transformed_u'.")
            # Replace NaN values with zero
            batch_array = np.nan_to_num(batch_array, nan=0.0)
            print(f"NaN values replaced with zero.", np.isnan(batch_array).sum())

        transformed_u_list.append(batch_array)


    transformed_u = np.stack(transformed_u_list, axis = 0)

    return  transformed_u



def plot(pos, y, transformed_pos, transformed_y, index , save_path= '.'):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with 1 row and 2 columns
    # Plot original points on the first axis
    axs[0].scatter(pos[index, :, 0], pos[index, :, 1], label='Original Points', s=1, c= y[index])
    axs[0].set_title('Original Points')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].grid(True)

    # Plot transformed points on the second axis
    # axs[1].scatter(transformed_pos[index, :, 0], transformed_pos[index, :, 1], label='Transformed Points', s=1, c = transformed_y[index])
    axs[1].scatter(transformed_pos[index, :, 0], transformed_pos[index, :, 1], label='Transformed Points', s=1)
    axs[1].set_title('Transformed Points')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].grid(True)
    plt.savefig(os.path.join(save_path, f'points_plot_{index}.png'))
    plt.close()


def plot1(pos, y, transformed_pos, transformed_y, index , save_path= '.'):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with 1 row and 2 columns
    # Plot original points on the first axis
    axs[0].scatter(pos[ :, 0], pos[ :, 1], label='Original Points', s=1, c= y)
    axs[0].set_title('Original Points')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].grid(True)

    # Plot transformed points on the second axis
    axs[1].scatter(transformed_pos[ :, 0], transformed_pos[ :, 1], label='Transformed Points', s=1, c = transformed_y)
    # axs[1].scatter(transformed_pos[:, 0], transformed_pos[ :, 1], label='Transformed Points', s=1)
    axs[1].set_title('Transformed Points')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].grid(True)
    plt.savefig(os.path.join(save_path, f'points_plot_{index}.png'))
    plt.close()
    



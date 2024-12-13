import numpy as np
import torch
from timeit import default_timer
from scipy.sparse import lil_matrix
import scipy
import time
from torch_geometric.utils import get_laplacian

def approximate_Psi(L, m, s):
    l_max = 2
    g = lambda x: np.exp(s*x)
    g_inv = lambda x: np.exp(-s*x)
    arange = (0.0, l_max)
    
    c = cheby_coeff(g, m, m+1, arange)
    c_inv = cheby_coeff(g_inv, m, m+1, arange)

    psi = cheby_op2(L, c, arange)
    psi_inv = cheby_op2(L, c_inv, arange)

    psi = torch.Tensor(psi)
    psi_inv = torch.Tensor(psi_inv)

    return psi, psi_inv

def cheby_coeff(g, m, N=None, arange=(-1, 1)):
    if N is None:
        N = m+1

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0
    n = np.pi * (np.r_[1:N+1] - 0.5) / N
    s = g(a1 * np.cos(n) + a2)
    c = np.zeros(m+1)
    for j in range(m+1):
        c[j] = np.sum(s * np.cos(j * n)) * 2 / N

    return c

def cheby_op2(L, c, arange):
    if not isinstance(c, list) and not isinstance(c, tuple):
        r = cheby_op2(L, [c], arange)
        return r[0]

    N_scales = len(c)
    M = np.array([coeff.size for coeff in c])
    max_M = M.max()

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0

    Twf_old = 0
    Twf_cur = (L-a2*torch.eye(L.shape[0])) / a1
    r = [0.5*c[j][0]*Twf_old + c[j][1]*Twf_cur for j in range(N_scales)]

    for k in range(1, max_M):
        Twf_new = (2/a1) * (L @ Twf_cur - a2*Twf_cur) - Twf_old
        for j in range(N_scales):
            if 1 + k <= M[j] - 1:
                r[j] = r[j] + c[j][k+1] * Twf_new

        Twf_old = Twf_cur
        Twf_cur = Twf_new

    return r

def calculate_psi_psi_inv_normalized_approximate_cheby(edge_index, num_nodes, s):
    s = s[0]
    t1 = time.time()
    L_norm = calculate_normalized_L(edge_index, num_nodes)
    t2 = time.time()
    print('time for L calculation', t2-t1 )
    t1= time.time()
    psi, psi_inv = approximate_Psi(L_norm, 40, s)
    t2 = time.time()
    print('time for psi_Calculation' , t2-t1)
    return psi, psi_inv

##########################################################################################
##
#########################################################################################



def full_eigen_decomposition(L):
    """
    L: Laplacian matrix
    """
    lambdas, U = torch.linalg.eigh(L)
    # print('lambdas', lambdas)
    num_zeros = torch.sum(lambdas < 1e-5).item()
    # print('number of zero eigenvalues', num_zeros)

    return lambdas, U

def truncated_eigen_decomposition(L, k1, k2):

    lambdas_low, U_low = torch.lobpcg(L, k = k1, largest = False, tol = 1E-3)

    # lambdas_high, U_high = torch.lobpcg(L, k = k2, largest = True, tol = 1E-3)
    # lambdas = np.concatenate((lambdas_low, lambdas_high))
    # U = np.concatenate((U_low, U_high), axis=1)
    return lambdas_low, U_low



def calculate_normalized_L(edge_index, num_nodes):
    edge_index_dense = torch.zeros((num_nodes, num_nodes))
    edge_index_dense[edge_index[0], edge_index[1]] = 1

    A = edge_index_dense
    # print('A', A)
    t_out = default_timer()
    # print('time for creating dense matrix', t_out-t_in)
    def is_symmetric(matrix):
        dense_matrix = matrix.to_dense()  # Convert the sparse matrix to dense
        return torch.equal(dense_matrix, dense_matrix.t())

    # Check if the adjacency matrix A is symmetric
    if is_symmetric(A):
        print("The adjacency matrix is symmetric.")
    else:
        print("The adjacency matrix is not symmetric.")

    D = torch.sum(A, dim=1)
    eps = 1e-6
    D_inv_sqrt_vec = 1.0 / torch.sqrt(D + eps)
    D_inv_sqrt = torch.diag(D_inv_sqrt_vec)
    # breakpoint()
    L_norm = torch.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    # L_norm = lil_matrix(L_norm)
    return L_norm



def calculate_psi_psi_inv_normalized_exact(lambdas, U, s, is_sparse):
    t1 = time.time()
    
    if not isinstance(U, torch.Tensor):
        U = torch.tensor(U)
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s)
    if not isinstance(lambdas, torch.Tensor):
        lambdas = torch.tensor(lambdas)
    
    diag_exp_s_lambdas = torch.exp(s * lambdas)
    diag_exp_neg_s_lambdas = torch.exp(-s * lambdas)
    
    psi = (U * diag_exp_s_lambdas).matmul(U.T)  # Element-wise multiplication and then matrix multiplication
    psi_inv = (U * diag_exp_neg_s_lambdas).matmul(U.T)  # Element-wise multiplication and then matrix multiplication
    
    if is_sparse:
        thr = 1e-5
        psi[torch.abs(psi) < thr] = 0
        psi_inv[torch.abs(psi_inv) < thr] = 0

        psi_inv = psi_inv.to_sparse()
        psi = psi.to_sparse()
    
    t2 = time.time()
    print('time for psi, psi_inv calculations', t2-t1)
    return psi, psi_inv



def calculate_lambdas_U_list_exact(edge_index, num_nodes):
    t1 = time.time()
    L_norm = calculate_normalized_L(edge_index, num_nodes)
    t2 = time.time()
    print('time for Laplacian construction from edge_index', (t2 -t1))
    t1 = time.time()
    lambdas, U = full_eigen_decomposition(L_norm)
    t1 = time.time()
    print('time for eigen value decomp', (t2 -t1))

    return lambdas , U



def calculate_lambdas_U_truncated_sparse(edge_index, num_nodes):
    t1 = time.time()
    L_norm_index , L_norm_values = get_laplacian(edge_index, normalization='sym')
    L_norm = torch.sparse_coo_tensor(indices = L_norm_index, values = L_norm_values, size=[num_nodes, num_nodes])
    t2 = time.time()
    print('time for Laplacian construction from edge_index', (t2 -t1))
    t1 = time.time()
    lambdas, U = truncated_eigen_decomposition(L_norm , k1 = 64 , k2 = 1)
    t2 = time.time()
    print('time for eigen value decomp', (t2 -t1))

    return lambdas, U
def calculate_lambdas_U_truncated_edgeweight_sparse(edge_index, edge_weight, num_nodes):
    t1 = time.time()
    L_norm_index , L_norm_values = get_laplacian(edge_index, edge_weight,  normalization='sym')
    L_norm = torch.sparse_coo_tensor(indices = L_norm_index, values = L_norm_values, size=[num_nodes, num_nodes])
    t2 = time.time()
    print('time for Laplacian construction from edge_index', (t2 -t1))
    t1 = time.time()
    lambdas, U = truncated_eigen_decomposition(L_norm , k1 = 64 , k2 = 1)
    t2 = time.time()
    print('time for eigen value decomp', (t2 -t1))

    return lambdas, U
def calculate_lambdas_U_truncated(edge_index, num_nodes):
    t1 = time.time()
    L_norm = calculate_normalized_L(edge_index, num_nodes)
    t2 = time.time()
    print('time for Laplacian construction from edge_index', (t2 -t1))
    t1 = time.time()
    lambdas, U = truncated_eigen_decomposition(L_norm , k1 = 24 , k2 = 64)
    t2 = time.time()
    print('time for eigen value decomp', (t2 -t1))
    return lambdas, U

def calculate_psi_psi_inv_list_exact(edge_index, num_nodes, s=[3.5, 4.0, 4.5]):
    L_norm = calculate_normalized_L(edge_index, num_nodes)
    lambdas, U = full_eigen_decomposition(L_norm)

    psi_list = []
    psi_inverse_list = []
    for s_value in s:
        psi, psi_inv = calculate_psi_psi_inv_normalized_exact(lambdas, U, s_value)
        psi_list.append(psi)
        psi_inverse_list.append(psi_inv)
    
    return psi_list, psi_inverse_list

def calculate_psi_psi_inv_list_truncated(edge_index, num_nodes, s=[3.5, 4.0, 4.5]):
    L_norm = calculate_normalized_L(edge_index, num_nodes)
    lambdas, U = truncated_eigen_decomposition(L_norm , k1 = 30 , k2 = 12)

    psi_list = []
    psi_inverse_list = []
    for s_value in s:
        psi, psi_inv = calculate_psi_psi_inv_normalized_exact(lambdas, U, s_value)
        psi_list.append(psi)
        psi_inverse_list.append(psi_inv)
    
    return psi_list, psi_inverse_list

def calculate_psi_psi_inv_list_cheby(edge_index, num_nodes, s=[ 4.0 ]):
    psi_list = []
    psi_inverse_list = []
    for s_value in s:
        psi, psi_inv = calculate_psi_psi_inv_normalized_approximate_cheby(edge_index, num_nodes, s_value)
        psi_list.append(psi)
        psi_inverse_list.append(psi_inv)
    
    return psi_list, psi_inverse_list



class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        # x -= self.mean
        # x /= (self.std + self.eps)
        return (x - self.mean) / (self.std + self.eps)
    
    def encode_(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        x -= self.mean
        x /= (self.std + self.eps)
        

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        # x *= std 
        # x += mean
        # breakpoint()
        return (x * std.to(x.device)) + mean.to(x.device)

    def decode_(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        x *= std 
        x += mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

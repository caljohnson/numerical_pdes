#MAT228A HW 4
#Compute Residual

#Computing the residual 
#for use in Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing 
#to find soluton to Poisson eqn.
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np
from math import exp, sin, pi

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg

def make_sparse_Laplacian(h):
	#set sparse matrix A, the discrete Laplacian
	N = int(1/h) -1
	offdiag = (1.0/(h**2))*np.ones(N**2)
	diag = np.ones(N**2)*(-4.0/(h**2))
	data = np.vstack((offdiag, offdiag, diag, offdiag, offdiag))
	A = sparse.dia_matrix((data, [-N,-1, 0,1,N]), shape = (N**2,N**2))
	return scipy.sparse.csr_matrix(A)

def apply_Laplacian(u, h):
	#remove Dirichlet BC's from u to work with Au
	#and flatten the matrix into a vector for Au multiplication
	u_inner = u[1:-1, 1:-1].flatten(order='F')
	A = make_sparse_Laplacian(h)

	#apply Laplacian to get matrix-vector product
	product = A.dot(u_inner)

	#shape matrix-vector product back into matrix with Dirichlet BC padding
	n = int(1/h)-1
	product_matrix = np.reshape(product, (n, n))
	padded_prod_matrix = np.pad(product_matrix, ((1,1),(1,1)), mode='constant')

	return padded_prod_matrix

def apply_Laplacian_nomatrix(u,h):
	n = int(1/h) -1
	Au = np.zeros((n+2,n+2))
	for i in range(1,n+1):
		for j in range(1,n+1):
			Au[i][j] = (1/(h**2))*(u[i+1][j]+u[i][j+1] -4*u[i][j] + u[i-1][j] + u[i][j-1])

	return Au

def compute_residual(u, f, h):
	#apply Laplacian to u
	Au = apply_Laplacian_nomatrix(u,h)

	#compute residual
	residual = f - Au

	return residual

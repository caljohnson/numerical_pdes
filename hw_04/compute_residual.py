#MAT228A HW 4
#Compute Residual

#Computing the residual 
#for use in Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing 
#to find soluton to Laplacian u = -e^(-(x-.25)^2 - (y-.6)^2)
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np
from math import exp, sin, pi

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
import tabulate
import argparse
from timeit import default_timer as timer 

def make_sparse_Laplacian(h):
	#set sparse matrix A, the discrete Laplacian
	N = int(1/h) -1
	offdiag = (1/(h**2))*np.ones(N)
	diag = np.ones(N)*(-4/(h**2))
	data = np.vstack((offdiag, offdiag, diag, offdiag, offdiag))
	A = sparse.dia_matrix((data, [-N,-1, 0,1,N]), shape = (N**2,N**2))
	return scipy.sparse.csr_matrix(A)

def RHS_function_sampled(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	f = [[-exp(-(x[i]-0.25)**2 - (y[j]-0.6)**2) for i in range(1,n+1)] for j in range(1,n+1)]
	
	#return flattened f, for dimensions to work with f-Au
	return np.asarray(f).flatten()

def compute_residual(u, h):
	#remove Dirichlet BC's from u to work with Au
	#and flatten the matrix into a vector for Au multiplication
	u_inner = u[1:-1, 1:-1].flatten()
	A = make_sparse_Laplacian(h)
	f = RHS_function_sampled(h)
	#obtain flattened residual
	flat_residual = f - A.dot(u_inner) 

	#reformat residual into a matrix
	n = int(1/h)-1
	residual = np.zeros((n+2, n+2))
	for i in range(1,n+1):
		for j in range(1,n+1):
			residual[i][j] = flat_residual[n*(j-1) + (i-1)]
	return residual		

# def test():
# 	h = 2**(-2)
# 	n = int(1/h)-1
# 	u = np.zeros((n+2, n+2))
# 	for i in range(1,n+1):
# 		for j in range(1,n+1):
# 			u[i][j] = 1
# 	residual = compute_residual(u, h)
# 	print residual		

# test()		

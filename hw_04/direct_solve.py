#MAT228A HW 4
#Direct Solve

#Directly solve for the error on the coarsest grid
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
import compute_residual
import full_weighting_restriction

def make_sparse_Laplacian(h):
	#set sparse matrix A, the discrete Laplacian
	N = int(1/h) -1
	offdiag = (1/(h**2))*np.ones(N)
	diag = np.ones(N)*(-4/(h**2))
	data = np.vstack((offdiag, offdiag, diag, offdiag, offdiag))
	A = sparse.dia_matrix((data, [-N,-1, 0,1,N]), shape = (N**2,N**2))
	return scipy.sparse.csc_matrix(A)

def direct_solve(residual, h):
	#make discrete Laplacian for solver
	A = make_sparse_Laplacian(h)

	#remove Dirichlet BC's from residual
	#and flatten the matrix into a vector for solver
	res = residual[1:-1, 1:-1].flatten()

	#obtain direct solve approximation
	approx_u = scipy.sparse.linalg.spsolve(A, res)

	#reformat solution into a matrix
	n = int(1/h)-1
	approx_u_soln = np.zeros((n+2, n+2))
	for i in range(1,n+1):
		for j in range(1,n+1):
			approx_u_soln[i][j] = approx_u[n*(j-1) + (i-1)]
	return approx_u_soln		

def RHS_function(x, y):
	return -exp(-(x-0.25)**2 - (y-0.6)**2) 

def trivial_direct_solve(residual, h):
	#for when h=1/2 and direct solve is completely trivial
	n= int(1/h)-1
	approx_u = np.zeros((n+2, n+2))
	approx_u[1][1] = -h**2 * residual[1][1] / 4

	return approx_u

# def test():
# 	h = 2**(-3)
# 	n = int(1/h)-1
# 	u = np.zeros((n+2, n+2))
# 	for i in range(1,n+1):
# 		for j in range(1,n+1):
# 			u[i][j] = 1
# 	residual = compute_residual.compute_residual(u,h)
# 	soln = direct_solve(full_weighting_restriction.full_weighting_restriction(residual, h), 2*h)
# 	print soln
		

# test()		

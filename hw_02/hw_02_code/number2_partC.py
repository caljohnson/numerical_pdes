#MAT228A Homework 2
#Question 2, Part C

#Investigate the effect of O(h^p) LTE at interior and boundary points
#on whether the solution to the
#standard 3-pt Lapacian discretization 
#of u_xx = exp(x), u(0)=0, u(1)=1
#changes the order of accuracy


from __future__ import division

import numpy as np
from math import exp

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
import tabulate



#Refinement study loop size
max_loop = 10

#O(h^p) LTE  - vector of orders p
p_orders = [-3, -2, -1, 0, 1, 2, 3] 
p_size = np.size(p_orders)

#initialize grid size vector, error vectors
h=1/10
grid_spacing_size = [h/(2**i) for i in range(max_loop)]
one_norm_errors_pa = np.zeros((max_loop, p_size))
one_norm_errors_pb = np.zeros((max_loop, p_size))


#Loop for refinement study
for i in range(max_loop):

	#size of grid spacing
	h = h/2
	#Set number of grid points
	N = 1/h - 1

	#set grid points starting from h to 1-h 
	#since we include Dirichlet conditions in RHS
	x = np.linspace(h, 1-h, num=N)

	#set sparse matrix A, the discrete Laplacian
	offdiag = (1/(h**2))*np.ones(N)
	diag = np.ones(N)*(-2/(h**2))
	data = np.vstack((offdiag, diag,offdiag))
	A = sparse.dia_matrix((data, [-1, 0,1]), shape = (N,N))
	A_CSC_form = scipy.sparse.csc_matrix(A)

	#find actual solution u(x) = e^x + (2-e)x -1
	u = np.exp(x) + (2-exp(1))*x - np.ones(N)
	
	#Loop for LTE order effects 
	for j in range(p_size):
		
		#Do two analyses:
		#set RHS of PDE, Au(x_i) = f(x_i) = b_pa
		b_pa = np.exp(x)
		#include BC u(1)=1 in RHS, (u(0)=0 trivially included)
		b_pa[-1] = b_pa[-1] - 1/(h**2)
		#include O(h^p) LTE at boundary point x_1=h as in part a
		b_pa[0] = b_pa[0] + (h**p_orders[j])

		#set RHS of PDE, Au(x_i) = f(x_i) = b_pb
		b_pb = np.exp(x)
		#include BC u(1)=1 in RHS, (u(0)=0 trivially included)
		b_pb[-1] = b_pb[-1] - 1/(h**2)
		#include O(h^p) LTE at fixed interior point at x=1/2 as in part b
		b_pb[N/2] = b_pb[N/2] + (h**p_orders[j])

		#solve Au = b for parts a and b
		approx_u_pa = scipy.sparse.linalg.spsolve(A_CSC_form, b_pa)
		approx_u_pb = scipy.sparse.linalg.spsolve(A_CSC_form, b_pb)

		#one norm errors
		one_norm_errors_pa[i,j] = h*np.sum(np.abs(u-approx_u_pa))
		one_norm_errors_pb[i,j] = h*np.sum(np.abs(u-approx_u_pb))



#give tables of ratios of successive errors with given LTE O(h^p)
for j in range(p_size):
	print(p_orders[j])
	one_norm_table_pa = [[grid_spacing_size[i], one_norm_errors_pa[i,j], one_norm_errors_pa[i+1,j]/one_norm_errors_pa[i,j]] for i in range(max_loop-1)]
	print tabulate.tabulate(one_norm_table_pa, headers = ["h", "Boundary Point 1-norm Error", "Succesive Error Ratio"], tablefmt="latex")
	one_norm_table_pb = [[grid_spacing_size[i], one_norm_errors_pb[i,j], one_norm_errors_pb[i+1,j]/one_norm_errors_pb[i,j]] for i in range(max_loop-1)]
	print tabulate.tabulate(one_norm_table_pb, headers = ["h", "Interior Point 1-norm Error", "Succesive Error Ratio"], tablefmt="latex")

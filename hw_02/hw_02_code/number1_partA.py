#MAT228A Homework 2
#Question 1, Part A

#Use standard 3-pt Lapacian discretization 
#to find soluton to u_xx = exp(x), u(0)=0, u(1)=1

#perform a refinement study using exact solution
#to show error convergence for 1-norm and max 1-norm

from __future__ import division

import numpy as np
from math import exp

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
import tabulate

#Refinement study loop size
max_loop = 10

#initialize grid size vector, error vectors
h=1/10
grid_spacing_size = [h/(2**i) for i in range(max_loop)]
one_norm_errors = np.zeros(max_loop)
max_norm_errors = np.zeros(max_loop)


#Loop for refinement study
for i in range(max_loop):

	#size of grid spacing
	h = h/2
	#Set number of grid points
	N = 1/h - 1

	#set grid points starting from h to 1-h 
	#since we include Dirichlet conditions in RHS
	x = np.linspace(h, 1-h, num=N)
	#set RHS of PDE, Au(x_i) = f(x_i) = b
	b = np.exp(x)
	#include BC u(1)=1 in RHS, (u(0)=0 trivially included)
	b[-1] = b[-1] - 1/(h**2)

	#set sparse matrix A, the discrete Laplacian
	offdiag = (1/(h**2))*np.ones(N)
	diag = np.ones(N)*(-2/(h**2))
	data = np.vstack((offdiag, diag,offdiag))
	A = sparse.dia_matrix((data, [-1, 0,1]), shape = (N,N))
	A_CSC_form = scipy.sparse.csc_matrix(A)

	#solve Au = b
	approx_u = scipy.sparse.linalg.spsolve(A_CSC_form, b)

	#find actual solution u(x) = e^x + (2-e)x -1
	u = np.exp(x) + (2-exp(1))*x - np.ones(N)

	#one norm error
	one_norm_errors[i] = h*np.sum(np.abs(u-approx_u))

	#max norm error
	max_norm_errors[i] = np.amax(np.abs(u-approx_u))

#give table of ratios of successive errors
one_norm_table = [[grid_spacing_size[i], one_norm_errors[i], one_norm_errors[i+1]/one_norm_errors[i]] for i in range(max_loop-1)]
print tabulate.tabulate(one_norm_table, headers = ["a", "b", "c"], tablefmt="latex")

max_norm_table = [[grid_spacing_size[i], max_norm_errors[i], max_norm_errors[i+1]/max_norm_errors[i]] for i in range(max_loop-1)]
print tabulate.tabulate(max_norm_table, headers = ["a", "b", "c"], tablefmt="latex")

#plot errors
plt.figure()
plt.loglog(grid_spacing_size, max_norm_errors, '-o', label="Max Norm Error")
plt.loglog(grid_spacing_size, one_norm_errors, '--s', label="One Norm Error")
plt.legend(loc=0)
plt.title(r"$u_{xx} = e^{x}$ approximate solution", fontsize=12)
plt.xlabel(r"grid spacing size $h$")
plt.ylabel("error")
plt.savefig("problem1a_refinement_study.png", dpi=300)
plt.show()
plt.close()






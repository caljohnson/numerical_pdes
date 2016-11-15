#MAT228A Homework 2
#Question 1, Part B

#Use standard 3-pt Lapacian discretization 
#to find soluton to u_xx = 2cos^2(pi x), u'(0)=0, u'(1)=1

#perform a refinement study using exact solution
#to show error convergence for 1-norm and max 1-norm

from __future__ import division

import numpy as np
from math import exp, pi

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.sparse import coo_matrix, bmat
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
	#set grid points
	x = np.linspace(0, 1, num=(N+2))
	#set RHS of PDE, Au(x_i) = f(x_i) = b
	b = 2*(np.cos(pi*x)**2)
	#include BCs u'(0)=0, u'(1)=1 in RHS
	b[0] = b[0]/2 + 0/h
	b[-1] = b[-1]/2-1/h

	#set sparse matrix A, the discrete Laplacian
	offdiag = (1/(h**2))*np.ones(N+2)
	diag = np.concatenate(([-1],-2*np.ones(N),[-1]))*(1/(h**2))
	data = np.vstack((offdiag, diag,offdiag))
	A = sparse.dia_matrix((data, [-1, 0,1]), shape = (N+2,N+2))


	#set augmented system matrix for Au + Lambda v = b, 1T u = 0
	A_temp = scipy.sparse.hstack([A, np.ones((N+2,1))])
	A_aug = scipy.sparse.vstack([A_temp, np.concatenate((np.ones(N+2), [0]))
	])
	A_aug_CSC_form = scipy.sparse.csc_matrix(A_aug)

	#solve Au + Lambda v = b, 1T u = 0
	approx_u = scipy.sparse.linalg.spsolve(A_aug_CSC_form, np.concatenate((b, [0])))
	approx_lambda = approx_u[N+2]
	print(approx_lambda)
	#get rid of lambda element
	approx_u = np.delete(approx_u, N+2) 
	
	#find actual solution u(x) = x^2/2 - 1/(4pi^2) cos(2pi x) + c
	u = (1/2)*x**2 - 1/(4*pi**2)*np.cos(2*pi*x) - np.sum((1/2)*x**2 - 1/(4*pi**2)*np.cos(2*pi*x))/(N+2)
	
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
plt.ylim([-0.002, 0.005])
plt.legend(loc=0)
plt.title(r"$u_{xx} = 2\cos^2(2\pi x)$ approximate solution", fontsize=12)
plt.xlabel(r"grid spacing size $h$")
plt.ylabel("error")
plt.savefig("problem1b_refinement_study.png", dpi=300)
plt.show()
plt.close()





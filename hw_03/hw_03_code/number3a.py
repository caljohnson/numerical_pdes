#MAT228A Homework 3
#Question 3 Part A

#Use standard 5-pt Lapacian discretization 
#to find soluton to Au = f, dirichlet bc's

#Perform a Direct Solve and time it!
#compare with SOR from question 1

from __future__ import division

import numpy as np
from np import exp

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
import tabulate
from timeit import default_timer as timer 

#mesh spacings
mesh_spacings = [2**(-5), 2**(-6), 2**(-7)]

def RHS(x,y):
	return -exp(-(x-0.25)**2 - (y-0.6)**2)

#Loop for refinement study
for h in mesh_spacings:

	#start timer
	start = timer()
	#Set number of grid points
	N = int(1/h - 1)

	#set grid points starting 
	x = [i*h for i in range(N+2)]
	y = [j*h for j in range(N+2)]
	#set RHS of PDE, Au(x_i,y_j) = f(x_i,y_j) = b
	f = [ [RHS(x[i],y[j]) for i in range(1,N+1)] for j in range(1,N+1)] 
	b = np.asarray(f)
	b= b.flatten()

	#set sparse matrix A, the discrete Laplacian
	offdiag = (1/(h**2))*np.ones(N)
	diag = np.ones(N)*(-4/(h**2))
	data = np.vstack((offdiag, offdiag, diag,offdiag, offdiag))
	A = sparse.dia_matrix((data, [-N,-1, 0,1,N]), shape = (N**2,N**2))
	A_CSC_form = scipy.sparse.csc_matrix(A)

	#solve Au = b
	approx_u = scipy.sparse.linalg.spsolve(A_CSC_form, b)
	end = timer()
	print(end-start)


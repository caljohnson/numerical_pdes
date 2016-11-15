#MAT228A Homework 3
#Question 3 Part B

#Use standard 7-pt Lapacian discretization 
#to find soluton to Au = f, dirichlet bc's

#Perform a Direct Solve and time it!
#compare with SOR from question 1

from __future__ import division

import numpy as np
from math import exp

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
import tabulate
from timeit import default_timer as timer 

#mesh spacings
mesh_spacings = [2**(-4), 2**(-5), 2**(-6)]



#Loop for refinement study
for h in mesh_spacings:

	#start timer
	start = timer()
	#Set number of grid points
	N = int(1/h - 1)

	#set grid points starting 
	x = [i*h for i in range(N+2)]
	y = [j*h for j in range(N+2)]
	z = [k*h for k in range(N+2)]

	#set RHS of PDE, Au(x_i,y_j) = f(x_i,y_j) = b
	f = [ [ [-exp(-(x[i]-0.25)**2 - (y[j]-0.6)**2 -(z[k])**2) for i in range(1,N+1)] for j in range(1,N+1)] for k in range(1,N+1)]
	b = np.asarray(f)
	b= b.flatten()

	#set sparse matrix A, the discrete Laplacian
	offdiag = (1/(h**2))*np.ones(N)
	diag = np.ones(N)*(-6/(h**2))
	data = np.vstack((offdiag, offdiag, offdiag, diag,offdiag, offdiag, offdiag))
	A = sparse.dia_matrix((data, [-N**2,-N,-1, 0,1,N, N**2]), shape = (N**3,N**3))
	A_CSC_form = scipy.sparse.csc_matrix(A)

	#solve Au = b
	approx_u = scipy.sparse.linalg.spsolve(A_CSC_form, b)
	end = timer()
	print(end-start)


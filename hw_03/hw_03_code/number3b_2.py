#MAT228A Homework 3
#Question 3 part B (2)

#Use Jacobi, GS-Lex, and SOR (on top of GS-lex)
#to find soluton to Laplacian u = -e^(-(x-.25)^2 - (y-.6)^2)
#with homogeneous Dirichlet BC's

#use mesh spacings h=2^-5, 2^-6, 2^-7
#track iteration counts for each method

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

#set empty array to hold iteration counts for different mesh spacings
counts = np.zeros(3)
#initialize mesh spacing number, to access proper counts row
mesh_number = 0
#set mesh spacings
mesh_spacings = [2**(-4), 2**(-5), 2**(-6)]

#loop through mesh spacings
for h in mesh_spacings:

	#start timer
	start = timer()
	#set relative error tolerance
	tol = h**2

	#set number of grid points in each row/column
	n = int(1/h - 1)

	
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]
	z = [k*h for k in range(n+2)]
	
	#set empty SOR solution matrices
	u_SOR = np.zeros((n+2, n+2, n+2))
	old_SOR = np.zeros((n+2, n+2,n+2))


	#set optimum w for SOR
	w = 2/(1+sin(pi*h))

	#sample f = -e^(-(x-.25)^2 - (y-.6)^2) at grid points
	f = [ [ [-exp(-(x[i]-0.25)**2 - (y[j]-0.6)**2 - z[k]**2) for i in range(n+2)] for j in range(n+2)] for k in range(n+2)]
	

	#initialize method finished flags and method iteration counts to 0
	flag_SOR = 1
	itcount_SOR = 0

	#begin iterative schemes
	while  flag_SOR:

		#update iteration counts
		if flag_SOR:
			itcount_SOR += 1

		for k in range(1,n+1):
			for j in range(1,n+1):
				for i in range(1,n+1):
					u_SOR[i][j][k] = (w/6)*(u_SOR[i-1][j][k] +u_SOR[i][j-1][k] + u_SOR[i][j][k-1]+u_SOR[i+1][j][k] + u_SOR[i][j+1][k]+u_SOR[i][j][k+1] - (h**2)*f[i][j][k]) + (1-w)*u_SOR[i][j][k]

		#check whether these methods have relative error within the tolerance THIS IS A BAD ERROR TO LOOK AT	
		if np.amax(np.abs(u_SOR-old_SOR)) <= tol*np.amax(np.abs(old_SOR)) and flag_SOR:
			flag_SOR = 0

		#remember (+0 is so that they dont BECOME THE SAME VARIABLE ?!?!?! WTF WHY)
		old_SOR = u_SOR + 0

	#store iteration counts
	print(itcount_SOR)
	counts[mesh_number] = itcount_SOR
	mesh_number+=1
	end = timer()
	print(end-start)

#make table of iteration counts
table = [[1/mesh_spacings[i], counts[i]] for i in range(3)]
print tabulate.tabulate(table, headers = ["Grid size NxN (N shown)", "SOR"], tablefmt="latex")


#MAT228A Homework 4
#GS-RB Smoother

#Use GS-RB to smooth 
#in larger multigrid V-cycle program
#to find soluton to Poisson eqn.
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np

def GS_RB_smoother(u, f, h, steps):

	#set number of grid points in each row/column
	n = int(1/h - 1)
	
	#separate red, black indices into two lists
	reds = []
	blacks = []
	for i in range(1,n+1):
		for j in range(1,n+1):
			if (i+j)%2==0:
				reds.append((i,j))
			else:
				blacks.append((i,j))

	#begin iterative scheme
	for k in range(steps):

		#loop red 
		for (i,j) in reds:
			# print "red", i,j
			u[i][j] = (1/4)*(u[i-1][j]+u[i][j-1]+u[i+1][j] + u[i][j+1] - (h**2)*f[i][j])
		
		#loop black 
		for (i,j) in blacks:
			# print "black", i,j
			u[i][j] = (1/4)*(u[i-1][j]+u[i][j-1]+u[i+1][j] + u[i][j+1] - (h**2)*f[i][j])

	return u
#MAT228A Homework 4
#GS-RB Smoother

#Use GS-RB to smooth 
#in larger multigrid V-cycle program
#to find soluton to Poisson eqn.
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np
from math import exp, sin, pi
from time import clock

def GS_RB_smoother(u, f, h, steps):

	#set number of grid points in each row/column
	n = int(1/h - 1)
	
	#set empty Jacobi, GS, and SOR solution matrices
	u_RB = u+0
	old_RB = u + 0

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
			u_RB[i][j] = (1/4)*(old_RB[i-1][j]+old_RB[i][j-1]+old_RB[i+1][j] + old_RB[i][j+1] - (h**2)*f[i][j])
		
		#update old_RB
		old_RB = u_RB + 0
		
		#loop black 
		for (i,j) in blacks:
			u_RB[i][j] = (1/4)*(old_RB[i-1][j]+old_RB[i][j-1]+old_RB[i+1][j] + old_RB[i][j+1] - (h**2)*f[i][j])

		#update old_RB (+0 is so that the data is copied, not the pointer)
		old_RB = u_RB+0

	return u_RB
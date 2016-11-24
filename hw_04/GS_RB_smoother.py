#MAT228A Homework 4
#GS-RB Smoother

#Use GS-RB to smooth 
#in larger multigrid V-cycle program
#to find soluton to Laplacian u = -e^(-(x-.25)^2 - (y-.6)^2)
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

	#begin iterative scheme
	for k in range(steps):

		#loop red (even entried points)
		for i in range(1,n+1):
			for j in range(1,n+1):
				if (i+j)%2==0:
					u_RB[i][j] = (1/4)*(old_RB[i-1][j]+old_RB[i][j-1]+old_RB[i+1][j] + old_RB[i][j+1] - (h**2)*f[i][j])
		#update old_RB
		old_RB = u_RB + 0
		#loop black (odd entried points)
		for i in range(1,n+1):
			for j in range(1,n+1):
				if (i+j)%2!=0:
					u_RB[i][j] = (1/4)*(old_RB[i-1][j]+old_RB[i][j-1]+old_RB[i+1][j] + old_RB[i][j+1] - (h**2)*f[i][j])

		#remember (+0 is so that the data is copied, not the pointer)
		old_RB = u_RB+0

	return u_RB
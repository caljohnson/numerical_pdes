#MAT228A Homework 4
#GS-RB Smoother

#Use GS-RB to smooth 
#in larger multigrid V-cycle program
#to find soluton to Laplacian u = -e^(-(x-.25)^2 - (y-.6)^2)
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np
from math import exp, sin, pi

def GS_RB_smoother(u, f, h, steps):

	#set number of grid points in each row/column
	n = int(1/h - 1)
	
	#set empty Jacobi, GS, and SOR solution matrices
	u_RB = u+0
	old_RB = u + 0
	
	#initialize method finished flag to 1 and method iteration count to 0
	flag = 1
	itcount = 0

	#begin iterative scheme
	for k in range(steps):

		#update iteration count
		itcount+=1

		for j in range(1,int((n+1)/2)):
			#loop red (even entried points)
			for i in range(1,int((n+1)/2)):
				u_RB[2*i][2*j] = (1/4)*(u_RB[i-1][j]+u_RB[i][j-1]+u_RB[i+1][j] + u_RB[i][j+1] - (h**2)*f[i][j])
			#loop black (odd entried points)
			for i in range(1,int((n+1)/2)):
				u_RB[2*i-1][2*j-1] = (1/4)*(u_RB[i-1][j]+u_RB[i][j-1]+u_RB[i+1][j] + u_RB[i][j+1] - (h**2)*f[i][j])

		#remember (+0 is so that the data is copied, not the pointer)
		old_RB = u_RB+0

	return u_RB
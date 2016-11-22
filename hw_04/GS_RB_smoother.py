#MAT228A Homework 4
#GS-RB Smoother

#Use GS-RB to smooth 
#in larger multigrid V-cycle program
#to find soluton to Laplacian u = -e^(-(x-.25)^2 - (y-.6)^2)
#with homogeneous Dirichlet BC's

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

def GS_RB_smoother(u, f, h, steps):

	#start timer
	# start = timer()

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

		# #check whether these methods have relative error within the tolerance
		# if np.amax(np.abs(u_RB-old_RB)) <=  tol*np.amax(np.abs(old_RB)):
		# 	flag = 0

		#remember (+0 is so that the data is copied, not the pointer)
		old_RB = u_RB+0

	# end = timer()
	#print(end-start)
	# print 'iterations = ', itcount

	return u_RB

# def test():
# 	h = 2**(-5)
# 	n = int(1/h - 1)
# 	u = np.zeros((n+2, n+2))
# 	u_RB = GS_RB_smoother(u, h, 2)
# 	print u_RB
# 	print(np.amax(np.abs(u_RB-u)))

# test()	

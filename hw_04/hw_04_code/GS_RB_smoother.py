#MAT228A Homework 4
#GS-RB Smoother

#Use GS-RB to smooth 
#in larger multigrid V-cycle program
#to find soluton to Poisson eqn.
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np
import copy
from math import exp, sin, pi
from time import clock

def GS_RB_smoother(u, f, h, steps):

	#set number of grid points in each row/column
	n = int(1/h - 1)
	
	#set empty Jacobi, GS, and SOR solution matrices
	# u_RB = copy.deepcopy(u)

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

def GS_lex_smoother(u, f, h, steps):

	n=int(1/h) - 1

	for i in range(1,n+1):
		for j in range(1,n+1):
			if (i + j) % 2 == 0:
				u[i][j] = (0.25)*(u[i-1][j]+ u[i][j-1] + u[i+1][j] + u[i][j+1] - (h**2)*f[i][j])

	for i in range(1,n+1):
		for j in range(1,n+1):
			if (i + j) % 2 == 1:
				u[i][j] = (0.25)*(u[i-1][j]+ u[i][j-1] + u[i+1][j] + u[i][j+1] - (h**2)*f[i][j])

	return u


def main():
	f = np.array([[1.4193, -0.2437, -0.6669, -0.8880, 1.7119, 0.1240, 1.3790 ],
		[0.2916,  0.2157,  0.1873,  0.1001, -0.1941, 1.4367, -1.0582],
		[0.1978, -1.1658,-0.0825,-0.5445,-2.1384, -1.9609,-0.4686],
		[1.5877, -1.1480, -1.9330, 0.3035, -0.8396,-0.1977,-0.2725],
		[ -0.8045, 0.1049, -0.4390, -0.6003, 1.3546, -1.2078,  1.0984],
		[0.6966, 0.7223, -1.7947, 0.4900,-1.0722,2.9080,-0.2779],
		[0.8351, 2.5855, 0.8404, 0.7394, 0.9610, 0.8252, 0.7015]])
	f = np.pad(f, ((1,1), (1,1)), mode='constant')
	u = 0*f
	u = GS_RB_smoother(u,f,1/8,1)
	print u
	raise

	raise

	while True:
		v=GS_RB_smoother(u,f, 1/4, 1)
		# print v; raise
		normu = np.amax(np.abs(v))
		print normu
		if normu < 0.001:
			break
		u = copy.deepcopy(v)	

if __name__ == "__main__":
	print "uh oh"
	main()

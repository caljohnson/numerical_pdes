#MAT228A HW 4
#Number 2


#Numerically estimate error convergence of
#multigrid solver
#with homogeneous Dirichlet BC's

#using known problem u_xx + u_yy = -2 pi^2 sin pi x sin pi y
#which has solution u(x,y) = sin pi x sin pi y 


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
from time import clock

from multigrid_V_cycle import V_cycle

def RHS_function_sampled(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	f = np.zeros((n+2,n+2))
	for i in range(1,n+1):
		for j in range(1,n+1):
			f[i][j] = -2*(pi**2)*sin(pi*x[i])*sin(pi*y[j])
	return f

def known_solution(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	u = np.zeros((n+2,n+2))
	for i in range(1,n+1):
		for j in range(1,n+1):
			u[i][j] = sin(pi*x[i])*sin(pi*y[j])
	return u

def main():
	h= 2**(-7)
	n = int(1/h - 1)
	u = np.zeros((n+2,n+2))
	f = RHS_function_sampled(h)
	u_known = known_solution(h)

	init_error = np.amax(np.abs(u_known-u))

	tol = 10**(-6)

	step_number=0
	#smoothing step counts to check over
	smooth_steps = [(0,0),(0,1), (1, 0), (1, 1), (0,2), (2,0), (1,2), (2, 1), (3,0), (0,3), (2, 2), (1, 3), (3,1), (4,0), (0,4)]
	conv_Factor = np.zeros((15,8))
	itcounts = np.zeros(15)

	for step in tqdm(smooth_steps):
		errors = [init_error]
		u = np.zeros((n+2,n+2))
		#do multigrid iteration
		itcount=0
		tic = clock()
		while True:
			u_old = u+0
			itcount += 1
			#use a V-cycle iteration with smoothing steps as given
			u = V_cycle(u_old, f, h, step[0], step[1])

			#calculate convergence factor
			error_k = np.amax(np.abs(u_known-u))
			errors.append(error_k)

			#stop when iterate differences within relative tol
			if np.amax(np.abs(u-u_old)) < tol*np.amax(np.abs(u_old)):
				break

		toc = clock()
		print toc-tic

		#compute convergence factors
		conv_factors = [(errors[k]/errors[0])**(1/k) for k in range(1,8)]
		#save data for table
		conv_Factor[step_number][0] = conv_factors[0]
		for i in range(1,8):
			conv_Factor[step_number][i] = np.average(conv_factors[:i])
		itcounts[step_number] = itcount

		step_number+=1

		print errors[itcount]

	#create table of output data
	smoothing_table = [[smooth_steps[i], conv_Factor[i][2],conv_Factor[i][3],conv_Factor[i][4],conv_Factor[i][5],conv_Factor[i][6], itcounts[i]] for i in range(15)]
	print tabulate.tabulate(smoothing_table, headers = ["v1, v2", "ave of 3", "average of 4", "average of 5", "average of 6", "average of 7", "iterations"], tablefmt="latex")



if __name__ == "__main__":
	main()
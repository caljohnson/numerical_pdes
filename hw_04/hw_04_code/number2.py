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
import scipy.sparse as sparse
import scipy.sparse.linalg
import tabulate
import argparse
from time import clock
from scipy.stats.mstats import gmean

from multigrid_V_cycle import V_cycle
from compute_residual import compute_residual, apply_Laplacian, get_Laplacian

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
	h= 2**(-6)
	n = int(1/h - 1)
	u = np.zeros((n+2,n+2))
	f = RHS_function_sampled(h)
	u_known = known_solution(h)

	init_error = np.amax(np.abs(u_known-u))

	#make sparse Laplacians for later computation
	Laplacians = []
	for i in range(2,10):
		Laplacians.append(get_Laplacian(2**(-i)))

	tol = 10**(-7)

	step_number=0
	#smoothing step counts to check over
	smooth_steps = [(0,1), (1, 0), (1, 1), (0,2), (2,0), (1,2), (2, 1), (3,0), (0,3), (2, 2), (1, 3), (3,1), (4,0), (0,4)]
	conv_Factor = np.zeros((15,8))
	itcounts = np.zeros(15)
	times = np.zeros(15)

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
			u = V_cycle(u_old, f, h, step[0], step[1], Laplacians)
			res = compute_residual(u, f, h, Laplacians[int(-2-np.log2(h))])

			#add error at this iterate to list of errors
			errors.append(np.amax(np.abs(u_known-u)))

			#stop when iterate differences within relative tol
			if np.amax(np.abs(res)) < tol*np.amax(np.abs(f)):
				break

		toc = clock()
		# print toc-tic

		#compute convergence factors
		conv_factors = [(errors[k]/errors[0])**(1/k) for k in range(1,6)]
		#save data for table
		for i in range(5):
			conv_Factor[step_number][i] = conv_factors[i]
		itcounts[step_number] = itcount
		times[step_number]=toc-tic

		step_number+=1

		# print errors[itcount]

	#create table of output data
	print "h = ", h
	print "tol =", tol
	smoothing_table = [[smooth_steps[i], conv_Factor[i][0],conv_Factor[i][1],conv_Factor[i][2],conv_Factor[i][3], conv_Factor[i][4], itcounts[i], times[i]] for i in range(14)]
	print tabulate.tabulate(smoothing_table, headers = ["v1, v2", "ave of 1", "average of 2", "average of 3", "average of 4", "average of 5", "iterations", "run times"], tablefmt="latex")



if __name__ == "__main__":
	main()
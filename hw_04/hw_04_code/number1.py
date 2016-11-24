#MAT228A HW 4
#Problem 1

#Use Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing 
#to find soluton to Laplacian u = -e^(-(x-.25)^2 - (y-.6)^2)
#with homogeneous Dirichlet BC's
#for different grid spacings

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
			f[i][j] = -exp(-(x[i]-0.25)**2 - (y[j]-0.6)**2)
	return f

def test_function(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	f = np.zeros((n+2,n+2))
	for i in range(1,n+1):
		for j in range(1,n+1):
			f[i][j] = -2*(pi**2)*sin(pi*x[i])*sin(pi*y[j])
	return f	

def test_solution(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	u = np.zeros((n+2,n+2))
	for i in range(1,n+1):
		for j in range(1,n+1):
			u[i][j] = sin(pi*x[i])*sin(pi*y[j])
	return u

def PARSE_ARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true", dest="test", default=False)
    #parser.add_argument("-tol", "--tolerance", type=float, nargs=1,dest="tol", default=10**(-7), required=True)
    return parser.parse_args()		

def main():
	#create grid spacings data
	grid_spacings = [2**(-5), 2**(-6), 2**(-7), 2**(-8)]
	#create iteration count holder and runtimes holder
	itcounts = []
	times = []

	args = PARSE_ARGS()
	tol = 10**(-7)

	if args.test:
		errors=[]

	#loop over different grid spacings
	for h in tqdm(grid_spacings):
		toc = clock()
		n = int(1/h - 1)
		u = np.zeros((n+2, n+2))
		if args.test:
			f = test_function(h)
			u_soln = test_solution(h)
		else:
			f = RHS_function_sampled(h)	


		#use multigrid algorithm
		itcount = 0
		while True:
			u_old = u + 0
			itcount += 1
			#use a V-cycle iteration
			u=V_cycle(u_old, f, h, 1, 1)

			#check convergence using relative tolerance
			if np.amax(np.abs(u-u_old)) < tol*np.amax(np.abs(u_old)):
				break		
		tic = clock()
		itcounts.append(itcount)
		times.append(tic-toc)
		if args.test:
			errors.append(np.amax(np.abs(u-u_soln))/np.amax(np.abs(u_soln)))

	#create table of output data
	if args.test:
		test_table = [[grid_spacings[i], itcounts[i], times[i], errors[i]] for i in range(np.size(grid_spacings)) ]
		print tabulate.tabulate(test_table, headers = ["grid spacing h", "iteration count", "run time (seconds)", "max errors"], tablefmt="latex")
	
	else:	
		table = [[grid_spacings[i], itcounts[i], times[i]] for i in range(np.size(grid_spacings)) ]
		print tabulate.tabulate(table, headers = ["grid spacing h", "iteration count", "run time (seconds)"], tablefmt="latex")
	

if __name__ == "__main__":
	main() 
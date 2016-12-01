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
import copy
import argparse
from timeit import default_timer as timer 
from time import clock
from multigrid_V_cycle import V_cycle
from compute_residual import compute_residual, make_sparse_Laplacian, apply_Laplacian_nomatrix
from GS_RB_smoother import GS_RB_smoother, GS_lex_smoother

def RHS_function_sampled(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	f = np.zeros((n+2,n+2),dtype=float)
	for i in range(1,n+1):
		for j in range(1,n+1):
			f[i][j] = -exp(-(x[i]-0.25)**2 - (y[j]-0.6)**2)
	return f

def test_function(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	f = np.zeros((n+2,n+2),dtype=float)
	# for i in range(1,n+1):
	# 	for j in range(1,n+1):
			# f[i][j] = -2*(pi**2)*sin(pi*x[i])*sin(pi*y[j])
	return f	

def test_solution(h):
	n = int(1/h)-1
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]

	u = np.zeros((n+2,n+2),dtype=float)
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
	grid_spacings = [2**(-5), 2**(-6), 2**(-7), 2**(-8), 2**(-9)]
	# grid_spacings = [2**(-2)]
	#create iteration count holder and runtimes holder
	itcounts = []
	times = []

	args = PARSE_ARGS()
	tol = 10**(-10)

	if args.test:
		errors=[]
		grid_spacings = [2**(-8)]

	#loop over different grid spacings
	for h in tqdm(grid_spacings):
		toc = clock()
		n = int(1/h - 1)
		u = np.zeros((n+2, n+2), dtype=float)
		# u = np.random.rand(n,n)
		# u= np.pad(u, ((1,1),(1,1)), mode='constant')
		# print np.amax(np.abs(u))
		if args.test:
			SOL = np.random.rand(n,n)
			SOL = np.pad(SOL, ((1,1),(1,1)), mode='constant')
			f = apply_Laplacian_nomatrix(SOL,h)

			# print f, SOL
			# f = np.random.rand(n,n)
			# f = np.pad(f, ((1,1),(1,1)), mode='constant')
			# f = test_function(h)
			# u_soln = test_solution(h)
			# soln_res = compute_residual(u_soln, f, h)
		else:
			f = RHS_function_sampled(h)	
		# res = compute_residual(u,f,h)
		# print "res = ", res

		# u = list(GS_RB_smoother(list(SOL),f,h,1))
		#use multigrid algorithm
		itcount = 0
		while True:
			# res_old = res)
			itcount += 1
			print itcount
			#use a V-cycle iteration
			u=V_cycle(u, f, h, 1,1)
			# u = GS_RB_smoother(u,f,h,1)
			# print u-SOL
			# u = list(GS_RB_smoother(list(u), f, h, 1))
			#print 'v cycle end' 
			res = compute_residual(u, f, h)
			# print "res = ", np.amax(np.abs(res))
			# print " error = " , np.amax(np.abs(SOL-u))
			# print "ratio of residuals = ", np.amax(np.abs(res))/np.amax(np.abs(res_old))
			#check convergence using relative tolerance
			# if np.amax(np.abs(u-u_old)) < tol*np.amax(np.abs(u_old)):
			# 	break
			if np.amax(np.abs(res)) < tol*np.amax(np.abs(f)):
				break

		tic = clock()
		itcounts.append(itcount)
		times.append(tic-toc)
		if args.test:
			res = compute_residual(u,f,h)
			print "final residual", np.amax(np.abs(res))
			errors.append(np.amax(np.abs(u-SOL))/np.amax(np.abs(SOL)))

	#create table of output data
	if args.test:
		test_table = [[grid_spacings[i], itcounts[i], times[i], errors[i]] for i in range(np.size(grid_spacings)) ]
		print tabulate.tabulate(test_table, headers = ["grid spacing h", "iteration count", "run time (seconds)", "max errors"], tablefmt="latex")
	
	else:	
		table = [[grid_spacings[i], itcounts[i], times[i]] for i in range(np.size(grid_spacings)) ]
		print tabulate.tabulate(table, headers = ["grid spacing h", "iteration count", "run time (seconds)"], tablefmt="latex")
	

if __name__ == "__main__":
	main() 
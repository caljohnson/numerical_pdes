#MAT228A HW 4
#Multigrid V-Cycle Solver

#Use Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing 
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
from time import clock
import GS_RB_smoother as GSRB
import full_weighting_restriction
import bilinear_interpolation
import compute_residual
import direct_solve

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
	
def V_cycle(u, f, h):
	#presmooth v1 times
	u = GSRB.GS_RB_smoother(u,f, h,1)

	#compute residual
	res = compute_residual.compute_residual(u, f, h)

	#restrict residual
	res2 = full_weighting_restriction.full_weighting_restriction(res, h)

	#solve for coarse grid error, check grid level to decide whether to solve or be recursive
	if h == 2**(-2):
		error = direct_solve.trivial_direct_solve(res2, 2*h)
	else:
		error = np.zeros((int(1/(2*h)+1), int(1/(2*h)+1)))
		error = V_cycle(error, res2, 2*h)	

	#interpolate error
	error2 = bilinear_interpolation.bilinear_interpolation(error, 2*h)

	#correct (add error back in)
	u = u+error2

	#post-smooth v2 times
	return GSRB.GS_RB_smoother(u, f, h, 1)

def main():
	h = 2**(-8)
	n = int(1/h - 1)
	u = np.zeros((n+2, n+2))
	f = RHS_function_sampled(h)

	tol = 10**(-7)

	#use multigrid algorithm
	itcount = 0
	while True:
		u_old = u + 0
		itcount += 1
		#use a V-cycle iteration
		u=V_cycle(u_old, f, h)

		#check convergence using relative tolerance
		if np.amax(np.abs(u-u_old)) < tol*np.amax(np.abs(u_old)):
			break

	print itcount

if __name__ == "__main__":
	main() 
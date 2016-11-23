#MAT228A HW 4
#Multigrid V-Cycle Solver

#Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing
#for Poission equation with Dirichlet BCs 


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
from GS_RB_smoother import GS_RB_smoother
from full_weighting_restriction import full_weighting_restriction
from bilinear_interpolation import bilinear_interpolation
from compute_residual import compute_residual
from direct_solve import trivial_direct_solve
	
def V_cycle(u, f, h, v1, v2):
	#presmooth v1 times
	u = GS_RB_smoother(u,f, h, v1)

	#compute residual
	res = compute_residual(u, f, h)

	#restrict residual
	res2 = full_weighting_restriction(res, h)

	#solve for coarse grid error, check grid level to decide whether to solve or be recursive
	if h == 2**(-2):
		error = trivial_direct_solve(res2, 2*h)
	else:
		error = np.zeros((int(1/(2*h)+1), int(1/(2*h)+1)))
		error = V_cycle(error, res2, 2*h, v1, v2)	

	#interpolate error
	error2 = bilinear_interpolation(error, 2*h)

	#correct (add error back in)
	u = u+error2

	#post-smooth v2 times
	return GS_RB_smoother(u, f, h, v2)

# def main():
# 	h = 2**(-8)
# 	n = int(1/h - 1)
# 	u = np.zeros((n+2, n+2))
# 	f = RHS_function_sampled(h)

# 	tol = 10**(-7)

# 	#use multigrid algorithm
# 	itcount = 0
# 	while True:
# 		u_old = u + 0
# 		itcount += 1
# 		#use a V-cycle iteration
# 		u=V_cycle(u_old, f, h, 1, 1)

# 		#check convergence using relative tolerance
# 		if np.amax(np.abs(u-u_old)) < tol*np.amax(np.abs(u_old)):
# 			break

# 	print itcount

# if __name__ == "__main__":
# 	main() 
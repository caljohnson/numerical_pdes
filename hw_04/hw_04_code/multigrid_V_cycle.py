#MAT228A HW 4
#Multigrid V-Cycle Solver

#Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing
#for Poission equation with Dirichlet BCs 


from __future__ import division

import numpy as np
from math import exp, sin, pi

from GS_RB_smoother import GS_RB_smoother
from full_weighting_restriction import full_weighting_restriction
from bilinear_interpolation import bilinear_interpolation
from compute_residual import compute_residual
from direct_solve import trivial_direct_solve
	
def V_cycle(u, f, h, v1, v2, Laplacians):
	#presmooth v1 times
	u = GS_RB_smoother(u,f, h, v1)

	#access correct Laplacian
	L = Laplacians[int(-2-np.log2(h))]

	#compute residual
	res = compute_residual(u, f, h, L)

	#restrict residual
	res2 = full_weighting_restriction(res, h)

	#solve for coarse grid error, check grid level to decide whether to solve or be recursive
	if h == 2**(-2):
		error = trivial_direct_solve(res2, 2*h)
	else:
		error = np.zeros((int(1/(2*h)+1), int(1/(2*h)+1)), dtype=float)
		error = V_cycle(error, res2, 2*h, v1, v2, Laplacians)	

	#interpolate error
	error2 = bilinear_interpolation(error, 2*h)

	#correct (add error back in)
	u = u+error2

	#post-smooth v2 times
	u = GS_RB_smoother(u, f, h, v2)

	return u
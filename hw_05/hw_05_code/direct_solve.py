#MAT228A HW 4
#Direct Solve

#Directly solve for the error on the coarsest grid
#for use in Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing 
#to find soluton to Poisson eqn
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
import compute_residual
import full_weighting_restriction


def trivial_direct_solve(residual, h):
	#for when h=1/2 and direct solve is completely trivial
	n= int(1/h)-1
	approx_u = np.zeros((n+2, n+2),dtype=float)
	approx_u[1][1] = -h**2 * residual[1][1] / 4
	return approx_u
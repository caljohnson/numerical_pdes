#MAT228A HW 4
#Bilinear Interpolation

#Bilinear interpolation operation for
#full Multigrid V-cycle with
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

def bilinear_interpolation(u, h):
	u_c = u +0
	n = int(1/h)-1
	h2 = h/2
	n2 = int(1/h2)-1
	u_f = np.zeros((n2+2, n2+2))

	#loop over coarse mesh, holding boundary entries zero as per Dirichlet BC
	for i in range(1,n+1):
		for j in range(1,n+1):
			#the 4 contribution from the stencil
			u_f[2*i][2*j] = u_c[i][j]
			#the 2 contributions from the stencil
			u_f[2*i+1][2*j] += u_c[i][j]/2
			u_f[2*i][2*j+1] += u_c[i][j]/2
			u_f[2*i-1][2*j] += u_c[i][j]/2
			u_f[2*i][2*j-1] += u_c[i][j]/2
			#the 1 contributions from the stencil
			u_f[2*i+1][2*j+1] += u_c[i][j]/4
			u_f[2*i+1][2*j-1] += u_c[i][j]/4
			u_f[2*i-1][2*j+1] += u_c[i][j]/4
			u_f[2*i-1][2*j-1] += u_c[i][j]/4
			
	return u_f

# def test():
# 	h = 2**(-4)
# 	n = int(1/h)-1
# 	u = np.zeros((n+2, n+2))
# 	for i in range(1,n+1):
# 		for j in range(1,n+1):
# 			u[i][j] = 1
# 	u2 = bilinear_interpolation(u,h)
# 	print u
# 	print u2

# test()	
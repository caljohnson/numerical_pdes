#MAT228A HW 4
#Full-Weighting Restriction

#Full-weighting restriction operation for
#full Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing 
#to find soluton to Poisson eqn
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np
from math import exp, sin, pi

def full_weighting_restriction(u, h):
	u_f = u +0
	h2 = 2*h
	n2 = int(1/h2)-1
	u_c = np.zeros((n2+2, n2+2))

	#loop over coarse mesh, holding boundary entries zero as per Dirichlet BC
	for i in range(1,n2+1):
		for j in range(1,n2+1):
			u_c[i][j] = 1/16*(4*u_f[2*i][2*j] + 2*u_f[2*i-1][2*j] +2*u_f[2*i][2*j-1]+ 2*u_f[2*i][2*j+1] + 2*u_f[2*i+1][2*j] +u_f[2*i+1][2*j+1] + u_f[2*i-1][2*j-1] + u_f[2*i+1][2*j-1] + u_f[2*i-1][2*j+1])
	return u_c
#MAT228A HW 4
#Bilinear Interpolation

#Bilinear interpolation operation for
#full Multigrid V-cycle with
#full-weighting, bilinear interpol, GS-RB smoothing 
#to find soluton to Poisson eqn.
#with homogeneous Dirichlet BC's

from __future__ import division

import numpy as np
from math import exp, sin, pi

def bilinear_interpolation(u_c, h):
	n = int(1/h)-1
	h2 = h/2
	n2 = int(1/h2)-1
	u_f = np.zeros((n2+2, n2+2),dtype=float)

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
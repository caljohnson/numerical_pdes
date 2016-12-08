#MAT228A HW 5
#Make Matrix RHS Circle Problem

# make the matrix corresponding to the 
# discrete Laplacian on the unit square 
# excluding a specified circle inside
# the square f is the rhs corresponding
# to u=1 on the circle equations are
# solved everywhere, inside and out,
# for simplicity

from __future__ import division
import numpy as np
from compute_residual import get_Laplacian
from numpy import sin, sqrt, pi
import scipy.sparse as sparse

def make_matrix_rhs_circleproblem(h):

	#initialize A as the standard Laplacian (scaled)
	n = int(1/h)-1
	# print "n = ", n
	A = (h**2)*get_Laplacian(h)

	#initialize f to be zeros
	f = np.zeros(n*n)

	#assume boundary value constant
	Ub = 1

	#form arrays of grid point locations
	x = [i*h for i in range(1,n+1)]
	x,y = np.meshgrid(x,x)

	#parameters for the embedded circle, center coordinates and radius
	xc = 0.3
	yc = 0.4
	rad = 0.15

	#compute the signed distance function
	phi = sqrt( (x-xc)**2 + (y-yc)**2 ) - rad
	IJ = [[-1, 0], [1, 0], [0, -1], [0,1]]

	for j in range(1,n-1):
		for i in range(1,n-1):

			#skip the interior points
			if phi[i][j] < 0:
				continue

			for k in range(4):
				if phi[i+IJ[k][0]][j+IJ[k][1]]  < 0:
					#approximate distance to the boundary, scaled by h
					alpha = phi[i][j]/(phi[i][j] - phi[ i+IJ[k][0] ][ j+IJ[k][1] ])

					#compute the distance to the boundary
					kr = np.ravel_multi_index([i,j], (n,n) )
					kc = np.ravel_multi_index([i+IJ[k][0], j+IJ[k][1]], (n,n))


					#adjust RHS
					f[kr] = f[kr] - Ub/alpha

					#adjust diagonal element
					A[kr,kr] = A[kr,kr] + 1 - 1/alpha

					#adjust off-diagonal and enforce symmetry
					A[kr,kc] = 0
					A[kc,kr] = 0

	return A, f, phi

def main():
	h = 2**(-4)
	[A, f, phi] = make_matrix_rhs_circleproblem(h)
	B = (h**2)*get_Laplacian(h)
	print "\n\n", A-B
	return	

if __name__ == "__main__":
	main()	
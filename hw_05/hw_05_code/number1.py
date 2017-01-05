#MAT228A HW 5
#Number 1

#Write a program to solve the discrete Poisson equation on the unit square using preconditioned
#conjugate gradient. Set up a test problem and compare the number of iterations and efficiency
#of using (i) no preconditioning, (ii) SSOR preconditioning, (iii) multigrid preconditioning.
#Run your tests for different grid sizes. How does the number of iterations scale with the
#number of unknowns as the grid is refined? For multigrid preconditioning, compare the
#efficiency of with multigrid as a standalone solver.

from argparse import ArgumentParser
import numpy as np
from numpy import sin, pi, exp
from conjugate_gradient_descent import conjugate_gradient_descent as cgd
from make_matrix_rhs_circleproblem import make_matrix_rhs_circleproblem as mmrc
from compute_residual import get_Laplacian, apply_Laplacian
import tabulate
from time import clock

def PARSE_ARGS():
	parser = ArgumentParser()
	parser.add_argument("-p", "--power", type=int, default=7, dest="power")
	parser.add_argument("-s", "--shampoo", type=int, default=1, dest="shampoo")
	parser.add_argument("-m", "--maxits", type=int, default=2000, dest="max_iterations")
	parser.add_argument("-t", "--tol", type=float, default=1e-7, dest="tol")
	return parser.parse_args()	

def RHS(h):
	n = int(1/h)-1
	u = np.zeros((n+2,n+2))

	for i in range(1,n+1):
		for j in range(1,n+1):
			# u[i][j] = sin(pi*i*h)*sin(pi*j*h)
			u[i][j] = -exp(-(i*h-0.25)**2 - (j*h-0.6)**2)
	return u

def main():
	ARGS = PARSE_ARGS()

	#make Laplacians for V cycle
	Laplacians = []
	for i in range(2,10):
		Laplacians.append(get_Laplacian(2**(-i)))

	grid_spacings = [2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)]
	itcounts = []
	times = [] 
	
	for h in grid_spacings:
		print "h = ", h
		#compute grid point number n and power
		n = int(1/h) -1
		power = -np.log2(h)

		#compute RHS to f=Au
		A = Laplacians[int(-2-np.log2(h))]
		f = RHS(h)
		
		#use CGD or PCG algorithm
		toc = clock()
		[u,iterations] = cgd(power, ARGS.shampoo, ARGS.tol, ARGS.max_iterations, A, Laplacians, f)
		tic = clock()
		times.append(tic-toc)
		itcounts.append(iterations)

	test_table = [[grid_spacings[i], itcounts[i], times[i]] for i in range(np.size(grid_spacings)) ]
	if ARGS.shampoo == 1:
		print "CG with no pre-conditioning"
	if ARGS.shampoo == 2:
		print "PCG with SSOR"
	if ARGS.shampoo == 3:
		print "PCG with MG"	
	print tabulate.tabulate(test_table, headers = ["grid spacing h", "iteration count", "run time (seconds)"], tablefmt="latex")




if __name__ == "__main__":
	main()	


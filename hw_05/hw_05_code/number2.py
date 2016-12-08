#MAT228A HW 5
#Number 2
#Unique Domain Poisson Solver Using CG

#give a matrix and right hand side for a discretized Poisson equation on
#a domain which is the intersection of the interior of the unit square and exterior of a circle
#centered at (0.3, 0.4) with radius 0.15. The boundary conditions are zero on the square and
#1 on the circle.
#Use your preconditioned conjugate gradient code to solve this problem. Explore the performance
#of no preconditioning and multigrid preconditioning for different grid sizes. Comment
#on your results
from argparse import ArgumentParser
import numpy as np

from conjugate_gradient_descent import conjugate_gradient_descent as cgd
from make_matrix_rhs_circleproblem import make_matrix_rhs_circleproblem as mmrc
from compute_residual import get_Laplacian


def PARSE_ARGS():
	parser = ArgumentParser()
	parser.add_argument("-p", "--power", type=int, default=7, dest="power")
	parser.add_argument("-s", "--shampoo", type=int, default=1, dest="shampoo")
	parser.add_argument("-m", "--maxits", type=int, default=1000, dest="max_iterations")
	parser.add_argument("-t", "--tol", type=float, default=1e-7, dest="tol")
	return parser.parse_args()	

def main():
	ARGS = PARSE_ARGS()
	#make special Laplacian and RHS for problem
	h = 2**(-ARGS.power)
	[A, f, phi] = mmrc(h)

	#make Laplacians for V cycle
	Laplacians = []
	for i in range(2,ARGS.power+1):
		Laplacians.append(get_Laplacian(2**(-i)))

	n = int(1/(2**(-ARGS.power)))-1
	f = f.reshape(n, n)
	f = np.pad(f, ((1,1),(1,1)), mode='constant')


	u = cgd(ARGS.power, ARGS.shampoo, ARGS.tol, ARGS.max_iterations, A, Laplacians, f)



if __name__ == "__main__":
	main()	
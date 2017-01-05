## MAT 228A Homework 5
## Conjugate Gradient Descent Algorithm

from __future__ import division

import numpy as np
import copy
import scipy.sparse.linalg as linalg
from time import clock
from numpy import sin, exp, pi
from tqdm import tqdm
from argparse import ArgumentParser
from compute_residual import apply_Laplacian, get_Laplacian, compute_residual
from multigrid_V_cycle import V_cycle


def MG_preconditioner(r, h, Laplacians):
	n = int(1/h)-1
	z = np.zeros((n+2,n+2))
	z = V_cycle(z,r,h,1,1,Laplacians)
	# print z
	return z

def SSOR_preconditioner(r, h):
	n = int(1/h)-1
	z = np.zeros((n+2,n+2))

	w = 2/(1+sin(pi*h))

	for i in range(1,n+1):
		for j in range(1,n+1):
			z[i][j] = (w/4)*(z[i-1][j] + z[i][j-1] + z[i+1][j] + z[i][j+1] - (h**2)*r[i][j]) + (1-w)*z[i][j]

	for i in range(n,0,-1):
		for j in range(n,0,-1):
			z[i][j] = (w/4)*(z[i-1][j] + z[i][j-1] + z[i+1][j] + z[i][j+1] - (h**2)*r[i][j]) + (1-w)*z[i][j]
	
	return z

def exact_solve_preconditioner(r,h,L):
	R = r[1:-1, 1:-1].flatten(order='C')
	N = int(1/h)-1
	z = linalg.spsolve(L,R).reshape(N,N)
	return np.pad(z, ((1,1),(1,1)), mode='constant')

def conjugate_gradient_descent(power, shampoo, tol, max_iterations, A, Laplacians, f):
	#set grid size h and grid points N
	h = 2**(-power)
	n = int(1/h) -1

	#initial guess
	u = np.zeros((n+2,n+2))

	#initialize residual
	r = f - apply_Laplacian(u,h,A)

	tic = clock()
	#apply pre-conditioner/ shampoo
	if shampoo == 1:
		z = copy.deepcopy(r)
	if shampoo == 2:
		z = SSOR_preconditioner(r, h)
	if shampoo == 3:
		z = MG_preconditioner(r,h, Laplacians)	
	if shampoo == 4:
		z = exact_solve_preconditioner(r,h,A)

	#initial search direction (residual/negative gradient)
	p = copy.deepcopy(z)


	#loop until solved
	t = tqdm(xrange(max_iterations))
	for k in t:
		w = apply_Laplacian(p, h, A)

		alpha_num = np.dot(z.flatten(),r.flatten())
		alpha_denom = np.dot(p.flatten(),w.flatten())
		alpha =  alpha_num/alpha_denom
		u = u+ alpha*p

		r_new = r - alpha*w

		if np.amax(np.abs(r_new)) <= tol*np.amax(np.abs(f)):
			print "iteration count =", k+1
			print "||r|| =", np.amax(np.abs(r_new))
			toc = clock()
			print "runtime = ", toc-tic
			break;

		#apply pre-conditioner
		if shampoo == 1:
			z_new = copy.deepcopy(r_new)
		if shampoo == 2:
			z_new = SSOR_preconditioner(r_new, h)
		if shampoo == 3:
			z_new = MG_preconditioner(r_new, h, Laplacians)
		if shampoo == 4:
			z_new = exact_solve_preconditioner(r_new,h,A)

		beta = np.dot(z_new.flatten(), r_new.flatten()) / np.dot(z.flatten(),r.flatten())

		p = z_new + beta*p

		r = copy.deepcopy(r_new)
		normr = np.amax(np.abs(r))
		t.set_description("||res||=%.10f"%normr)
		z = copy.deepcopy(z_new)

	if k == max_iterations-1:
		print "max iterations exceeded"

	return u, k+1			

def PARSE_ARGS():
	parser = ArgumentParser()
	parser.add_argument("-p", "--power", type=int, default=7, dest="power")
	parser.add_argument("-s", "--shampoo", type=int, default=1, dest="shampoo")
	parser.add_argument("-m", "--maxits", type=int, default=1000, dest="max_iterations")
	parser.add_argument("-t", "--tol", type=float, default=1e-7, dest="tol")
	return parser.parse_args()	

def main():
	ARGS = PARSE_ARGS()
	#make sparse Laplacians for later computation
	Laplacians = []
	for i in range(2,ARGS.power+1):
		Laplacians.append(get_Laplacian(2**(-i)))
	
	#set "known" solution u
	u_sol = known_solution(h)

	#compute RHS f=Au
	A = Laplacians[int(-2-np.log2(h))]
	f = apply_Laplacian(u_sol,h,A)

	u = conjugate_gradient_descent(ARGS.power,ARGS.shampoo,ARGS.tol,ARGS.max_iterations, A, Laplacians, RHS)
	return

if __name__ == "__main__":
	main()
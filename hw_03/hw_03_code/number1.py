#MAT228A Homework 3
#Question 1 

#Use Jacobi, GS-Lex, and SOR (on top of GS-lex)
#to find soluton to Laplacian u = -e^(-(x-.25)^2 - (y-.6)^2)
#with homogeneous Dirichlet BC's

#use mesh spacings h=2^-5, 2^-6, 2^-7
#track iteration counts for each method

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

#use command line input to decide which 
parser = argparse.ArgumentParser(description='Pick an iterative method')
parser.add_argument("-j", "--Jac", help = "turn on Jacobi iteration", action="store_true", default=0)
parser.add_argument("-g", "--GS", help = "turn on GS iteration", action="store_true", default=0)
parser.add_argument("-s", "--SOR", help = "turn on SOR iteration",action="store_true", default=0)
args = parser.parse_args()

do_Jac = args.Jac
do_GS = args.GS
do_SOR = args.SOR

#set empty array to hold iteration counts for different mesh spacings
counts = np.zeros((3,3))
#initialize mesh spacing number, to access proper counts row
mesh_number = 0
#set mesh spacings
mesh_spacings = [2**(-5), 2**(-6), 2**(-7)]

#loop through mesh spacings
for h in tqdm(mesh_spacings):

	#start timer
	start = timer()
	#set relative error tolerance
	tol = h**2

	#set number of grid points in each row/column
	n = int(1/h - 1)

	
	#set x,y grid point vectors (n x 1)
	x = [i*h for i in range(n+2)]
	y = [j*h for j in range(n+2)]
	
	#set empty Jacobi, GS, and SOR solution matrices
	u_Jac = np.zeros((n+2, n+2))
	u_GS = np.zeros((n+2, n+2))
	u_SOR = np.zeros((n+2, n+2))
	old_Jac = np.zeros((n+2, n+2))
	old_GS = np.zeros((n+2, n+2))
	old_SOR = np.zeros((n+2, n+2))


	#set optimum w for SOR
	w = 2/(1+sin(pi*h))

	#sample f = -e^(-(x-.25)^2 - (y-.6)^2) at grid points
	f = [ [-exp(-(x[i]-0.25)**2 - (y[j]-0.6)**2) for i in range(n+2)] for j in range(n+2)]
	

	#initialize method finished flags and method iteration counts to 0
	flag_Jac = do_Jac
	itcount_Jac = 0
	flag_GS = do_GS
	itcount_GS = 0
	flag_SOR = do_SOR
	itcount_SOR = 0

	#begin iterative schemes
	while  flag_GS or flag_Jac or flag_SOR:

		#update iteration counts
		if flag_Jac:
			itcount_Jac += 1
		if flag_GS:
			itcount_GS += 1
		if flag_SOR:
			itcount_SOR += 1

		for j in tqdm(range(1,n+1)):
			for i in range(1,n+1):
				#do if Jacobi iteration not done
				if flag_Jac:
					u_Jac[i][j] = (1/4)*(old_Jac[i-1][j]+old_Jac[i][j-1]+old_Jac[i+1][j] + old_Jac[i][j+1] - (h**2)*f[i][j])
				#do if GS iteration not done
				if flag_GS:
					u_GS[i][j] = 0.25*(u_GS[i-1][j]+u_GS[i][j-1]+u_GS[i+1][j] + u_GS[i][j+1]) - 0.25*(h**2)*f[i][j]
				#do if SOR iteration not done
				if flag_SOR:
					u_SOR[i][j] = (w/4)*(u_SOR[i-1][j]+u_SOR[i][j-1]+u_SOR[i+1][j] + u_SOR[i][j+1] - (h**2)*f[i][j]) + (1-w)*u_SOR[i][j]

		#check whether these methods have relative error within the tolerance THIS IS A BAD ERROR TO LOOK AT

		if np.amax(np.abs(u_Jac-old_Jac)) <=  tol*np.amax(np.abs(old_Jac)) and flag_Jac:
			flag_Jac = 0
		if np.amax(np.abs(u_GS-old_GS)) <=  tol*np.amax(np.abs(old_GS)) and flag_GS:
			flag_GS = 0			
		if np.amax(np.abs(u_SOR-old_SOR)) <= tol*np.amax(np.abs(old_SOR)) and flag_SOR:
			flag_SOR = 0

		#remember (+0 is so that they dont BECOME THE SAME VARIABLE ?!?!?! WTF WHY)
		old_Jac = u_Jac+0
		old_GS = u_GS +0
		old_SOR = u_SOR + 0

	#store iteration counts
	counts[mesh_number, 0] = itcount_Jac
	counts[mesh_number, 1] = itcount_GS
	counts[mesh_number, 2] = itcount_SOR
	# print(counts[mesh_number,0])
	# print(counts[mesh_number,1])
	# print(counts[mesh_number,2])
	mesh_number+=1
	end = timer()
	print(end-start)

#make table of iteration counts
table = [[1/mesh_spacings[i], counts[i][0], counts[i][1], counts[i][2]] for i in range(3)]
print tabulate.tabulate(table, headers = ["Grid size NxN (N shown)", "Jacobi", "GS", "SOR"], tablefmt="latex")


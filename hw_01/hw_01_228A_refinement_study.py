#hw_01_228A Refinement Study (problem 3 part b)
#Carter Johnson
#10/11/16

import numpy as np
from math import exp
import matplotlib.pyplot as plt

#my problem is x^3/3 -> second derivative is x
#evaluate at x=10
x=10
actual = 10

#2nd derivative approximate operator as function of grid spacing h
def D2_approx(h):
	return (2/(3*h**2))*(2*(x-h/2)**3/3 + (x+h)**3/3 - (x)**3)

#evaluate my 2nd derivative approximation for various h	
H = np.linspace(1, .001, 100)
D2_approxes = [D2_approx(h) for h in H]

#find abs errors of these approximations
max_errors = [abs(approx - actual) for approx in D2_approxes]

#plot as log-log
plt.figure()
plt.loglog(H, max_errors, label="Absolute Error")
plt.title("2nd derivative of $x^3/3$ at $x=10$", fontsize=12)
plt.xlabel("grid spacing ($h$)")
plt.ylabel("absolute error")
plt.savefig("problem3_refinement_study.png", dpi=300)
plt.close()


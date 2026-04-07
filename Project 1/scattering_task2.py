##This code asks the user for k_0 and calculates the phase shift and the pecentage error compared to the analytical expression,
##for a given interval of mesh points N,
##for the delta potential scattering problem.

import numpy as np
import matplotlib.pyplot as plt

#define constants
mu = 0.5
lamb= 1.5
b=10

#ask for user's input
N_i = int(input("Enter the initial number of mesh points: "))
N_f = int(input("Enter the final number of mesh points: "))
step = 2
k_0 = float(input("Enter k_0: "))
C = 1


#initialize the vector which will hold the value of phase shifts as function of N
phase_shift = []

#calculate phase shift
for N in range(N_i, N_f+1, step):
    k, w = np.polynomial.legendre.leggauss(N)
    w = C * w*np.pi/4/np.cos(np.pi/4*(k+1))**2
    k = C * np.tan(np.pi/4*(k+1))

    if k_0 not in k: #if k_0 is among the mesh points, skip this iteration
        u = 4*mu*w*k**2/(np.pi*(k_0**2-k**2))
        K= np.append(k, k_0)
        u = np.append(u, -np.sum(4*mu*w*k_0**2/(np.pi*(k_0**2-k**2))))
        V = -lamb*np.outer(np.sin(K*b)/K, np.sin(K*b)/K)
        A = np.eye(N+1)-V*u
        R = np.linalg.solve(A, V)
        R_k0_k0 = R[N][N]
        phase_shift.append(-np.arctan(2*mu*k_0*R_k0_k0))


#for each N, calculate the difference between the calculated phase shift and the theoretical one
phase_shift_theory = np.arctan(lamb*mu*b*2*(np.sin(k_0*b))**2/(k_0*b - lamb*mu*b*2*np.sin(k_0*b)*np.cos(k_0*b)))
delta_percent = abs((np.array(phase_shift)-phase_shift_theory)/phase_shift_theory)*100


#plot phase shift and percentual error as a function of mesh points N
plt.scatter(range(N_i, N_f+1, step), phase_shift, label="Phase shift (calculated)")
plt.axhline(y=phase_shift_theory, color='r', linestyle='--', label="Phase shift (analytical)")
plt.xlabel("Number of mesh points")
plt.ylabel("Phase shift")
plt.title("Phase shift in function of the number of mesh points")
plt.legend()

plt.figure(2)
plt.scatter(range(N_i, N_f+1, step), delta_percent, label="Percentage error")
plt.xlabel("Number of mesh points")
plt.ylabel("Percentage error (%)")
plt.title("Percentage error in function of the number of mesh points")

#calculate the number of mesh points that minimizes the error and show it in the graph
N_optimal = np.argmin(delta_percent)
delta_min = delta_percent[N_optimal]
plt.plot(range(N_i, N_f+1, step)[N_optimal], delta_min, 'ro', label="Minimal error")
plt.legend(title=f"Minimal error: {delta_min:.2f}% a N={range(N_i, N_f+1, step)[N_optimal]}")
plt.show()
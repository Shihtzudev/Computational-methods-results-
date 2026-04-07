##This code tries to verify whether there is a correlation between the error in phase shift and the number of mesh points close to k_0
##for the delta potential scattering problem.

import numpy as np
import matplotlib.pyplot as plt

#define constants
mu = 0.5
lamb = 1.5
b = 10.0

#problem parameters
N_range = np.arange(10, 50, 1)
k_0_range = np.arange(0.01, 0.30 + 0.01, 0.01)
C_range = [0.001, 0.01, 0.1, 1, 10, 100, 500]
relative_radius = 50  #expressed in percentile

#calculate phase shift and error for each triplet (N,k0,c)
error_vector = []
N_local_vector = []
for k_0 in k_0_range:
    phase_shift = []
    N_local = []
    for N in N_range:
        for C in C_range:
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
                    N_local.append(np.sum(np.abs(k - k_0) < k_0*relative_radius/100))
    phase_shift_theory = np.arctan(lamb*mu*b*2*(np.sin(k_0*b))**2/(k_0*b - lamb*mu*b*2*np.sin(k_0*b)*np.cos(k_0*b)))
    delta_percent = abs((np.array(phase_shift)-phase_shift_theory)/phase_shift_theory)*100
    error_vector.append(delta_percent)
    N_local_vector.append(N_local)

error_vector = np.concatenate(error_vector)
N_local_vector = np.concatenate(N_local_vector)

#plot delta_percent vs N_local
plt.figure(figsize=(10, 6))
plt.scatter(N_local_vector, error_vector, alpha=0.5)
#color in red the points with error lesser than 5%
plt.scatter(N_local_vector[error_vector < 5], error_vector[error_vector < 5], color='red', alpha=0.5)
#color in green the absolute minimum
min_index = np.argmin(error_vector)
plt.scatter(N_local_vector[min_index], error_vector[min_index], color='green', s=100, label='Minimum error')
plt.xlabel(f'Number of mesh points within {relative_radius:.0f}% of k_0')
plt.ylabel('Relative error in phase shift (%)')
plt.title('Correlation between error in phase shift and number of mesh points near k_0')
plt.grid()

#plot in another figure the percentage of red points vs N_local
plt.figure(2)
percent_red = []
for N_local in np.unique(N_local_vector):
    percent_red.append(np.sum((N_local_vector == N_local) & (error_vector < 5))/np.sum(N_local_vector == N_local)*100)
plt.bar(np.unique(N_local_vector), percent_red, color='red')
plt.xlabel(f'Number of mesh points within {relative_radius:.0f}% of k_0')
plt.ylabel('Percentage of points with error < 5 (%)')
plt.title('Distribution of errors in phase shift')
plt.grid()
plt.show()
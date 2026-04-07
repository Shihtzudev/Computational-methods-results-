##This code plots sin^2(phase shift) for a given interval of energies E
##picking for each E the optimal N (number of mesh points) that minimizes the error compared to the analytical expression
##for the delta potential scattering problem.

import numpy as np
import matplotlib.pyplot as plt

#define constants
b=10
mu = 0.5
lamb= 1.5

#ask for user's input
N_i = int(input("Enter initial number of mesh points: "))
N_f = int(input("Enter final number of mesh points: "))
N_step = 2
k_0_f = np.pi/b #the final value of k_0, which is expected to be the first zero of phase shift (from the analytical expression)
k_0_step = float(input("Enter step for k_0: ")) #this will also be used for the initial value of k_0

phase_shift_optimal=[] #initialize a vector of the optimal phase shifts for each value of k_0

for k_0 in np.arange(k_0_step, k_0_f+k_0_step, k_0_step):
    phase_shift=[] #initialize the vector which will hold the value of phase shifts as function of N

    for N in range(N_i, N_f+1, N_step):
        k, w = np.polynomial.legendre.leggauss(N)
        w = w*np.pi/4/np.cos(np.pi/4*(k+1))**2
        k = np.tan(np.pi/4*(k+1))

        if k_0 not in k: #if k_0 is among the mesh points, skip this value of N
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

    #for this value of k_0, calculate the optimal N that minimizes the error and save the corresponding phase shift
    N_optimal= np.argmin(delta_percent)
    phase_shift_optimal.append(phase_shift[N_optimal])


#plot sin^2(phase shift) in function of energy E

k_vector=np.arange(k_0_step, k_0_f+k_0_step, k_0_step)
E=k_vector**2/(2*mu)

plt.scatter(E, np.sin(phase_shift_optimal)**2, marker='o')
plt.xlabel("Energy (GeV)")
plt.ylabel("sin²(δ₀)")
plt.title("sin²(δ₀) as a function of E")
plt.grid()
plt.show()       
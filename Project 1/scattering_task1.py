##This code asks the user for k_0 and N (mesh points) and calculates the phase shift and the pecentage error compared to the analytical expression,
##for the delta potential scattering problem.

import numpy as np

mu = 0.5
lamb= 1.5
b=10
C = 1 #rescaling factor for the mesh points and weights
while True:
    N = int(input("Enter the number of mesh points N: "))
    k_0 = float(input("Enter k_0: "))

    k, w = np.polynomial.legendre.leggauss(N) #calculate mesh points and weights and rescale them to the interval [0, +inf]
    w = C * w*np.pi/4/np.cos(np.pi/4*(k+1))**2
    k = C * np.tan(np.pi/4*(k+1))

    if k_0 in k:
        print("Please choose another value for N so that K_0 is not among the mesh points.")
    else:
            break 

K = np.append(k, k_0) #create a vector with k and k_0

#solve the problem to find R(k_0, k_0)
u = 4*mu*w*k**2/(np.pi*(k_0**2-k**2))
u = np.append(u, -np.sum(4*mu*w*k_0**2/(np.pi*(k_0**2-k**2))))

V = -lamb*np.outer(np.sin(K*b)/K, np.sin(K*b)/K)
A = np.eye(N+1)-V*u
R = np.linalg.solve(A, V)
R_k0_k0 = R[N][N]

#print the results
print("R(k₀, k₀) =", R_k0_k0, "GeV⁻²")
phase_shift = -np.arctan(2*mu*k_0*R_k0_k0)
print("Phase shift =", phase_shift)
phase_shift_theory = np.arctan(lamb*mu*b*2*(np.sin(k_0*b))**2/(k_0*b - lamb*mu*b*2*np.sin(k_0*b)*np.cos(k_0*b)))
print("Phase shift (from analytical expression) =", phase_shift_theory)
delta_percent = abs((phase_shift-phase_shift_theory)/phase_shift_theory)*100
print("Percentage error =", delta_percent, "%")

#print the values of k (useful for debugging)
#print("k =", k)
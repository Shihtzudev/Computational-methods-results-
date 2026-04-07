##This code plots the phase shift as a function of energt E for a given N (number of mesh points)
##and compares it with data provided by "pwa93.xlsx"
##for the NP scattering problem.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #necessary to import the data from "pwa93.xlsx" - install with "pip install pandas openpyxl"

#define constants
mu = 0.7*197.3 #mu written in MeV
mass = 469.459 #reduced mass in MeV
coeff = [1, 4, 7]
V_0 = [-10.463, -1650.6, 6484.3]
C = 200

#energy is in the lab frame
E_i = 0.5
E_f = 350.0
E_step = 0.5
N = int(input("Enter number of mesh points: "))

##CALCULATION OF PHASE SHIFT

#initialize the vector which will hold the value of phase shifts as function of k_0
phase_shift = [] 
for E in np.arange(E_i, E_f+E_step, E_step):
    k_0 = np.sqrt(mass*E) #calculate k_0 from E - there is no 2 factor since the data are provided in terms of E_lab = 2 E_cm (approximatively)
    k, w = np.polynomial.legendre.leggauss(N)
    w = C* w*np.pi/4/np.cos(np.pi/4*(k+1))**2
    k = C* np.tan(np.pi/4*(k+1))

    if k_0 not in k: #if k_0 is among the mesh points, skip this value of k_0
        K= np.append(k, k_0)
        u = 4*mass*w*k**2/(np.pi*(k_0**2-k**2))
        u = np.append(u, -np.sum(4*mass*w*k_0**2/(np.pi*(k_0**2-k**2))))
        V = np.zeros((N+1, N+1))

        for i in range(N+1):
            for j in range(N+1):
                for q in range(3):
                    V[i][j] += V_0[q]/(4*K[i]*K[j]*mu)*np.log(((K[i]+K[j])**2+(mu*coeff[q])**2)/((K[i]-K[j])**2+(mu*coeff[q])**2))
    
        A = np.eye(N+1)-V*u
        R = np.linalg.solve(A, V)
        R_k0_k0 = R[N][N]
        phase_shift.append(-np.arctan(2*mass*k_0*R_k0_k0))
    else:
        phase_shift.append(np.nan) #if k_0 is among the mesh points, save NaN as phase shift to avoid plotting it

##COMPARISON WITH EXTERNAL DATA

#import the data from "pwa93.xlsx"
pwa_data = pd.read_excel("pwa93.xlsx", skiprows=1)
#extract first column (energy) and second column (phase shift) and convert them to lists
phase_shift_pwa = pwa_data.iloc[:, 1].tolist()
phase_shift_pwa = np.deg2rad(phase_shift_pwa) #convert phase shift from degrees to radians
absolute_error=np.abs((np.array(phase_shift) - np.array(phase_shift_pwa)))
percent_error = absolute_error/np.abs(phase_shift_pwa)*100

##PLOTTING

#plot on the same graph the phase shift calculated with our code and the one from "pwa93.xlsx"
E_vector=np.arange(E_i, E_f+E_step, E_step)
E_vector = E_vector/2 #we want to plot in the center-of-mass frame
plt.scatter(E_vector, phase_shift, marker='o', label="Phase shift (calculated)", s=5)
plt.plot(E_vector, phase_shift_pwa, label="Phase shift (PWA93)")
plt.xlabel("Energy (MeV)")
plt.ylabel("Phase shift")
plt.title("Phase shift as a function of E")
plt.grid()
plt.legend()


#plot in another graph the percentage error as a function of E
plt.figure(2)
plt.plot(E_vector, percent_error)
plt.xlabel("Energy (MeV)")
plt.ylabel("Percent error (%)")
#find the maximum and minimum error and show them in the graph and in the legend
max_error_index = np.argmax(percent_error)
min_error_index = np.argmin(percent_error)
plt.plot(E_vector, percent_error)
plt.scatter(E_vector[max_error_index], percent_error[max_error_index], color='red', label="Max error", s=50)
plt.scatter(E_vector[min_error_index], percent_error[min_error_index], color='green', label="Min error", s=50)
plt.legend()
print("Max error (%): ", percent_error[max_error_index], "% at E =", E_vector[max_error_index], "MeV")
print("Min error (%): ", percent_error[min_error_index], "% at E =", E_vector[min_error_index], "MeV")
plt.title("Percentage error as a function of E")
plt.grid()

#plot the absolute error as a function of E
plt.figure(3)
plt.plot(E_vector, absolute_error)
plt.xlabel("Energy (MeV)")
plt.ylabel("Absolute error")
#find the maximum and minimum error and show them in the graph and in the legend
max_error_index = np.argmax(absolute_error)
min_error_index = np.argmin(absolute_error)
plt.plot(E_vector, absolute_error)
plt.scatter(E_vector[max_error_index], absolute_error[max_error_index], color='red', label="Max error", s=50)
plt.scatter(E_vector[min_error_index], absolute_error[min_error_index], color='green', label="Min error", s=50)
plt.legend()
print("Max error (absolute):", absolute_error[max_error_index], " at E =", E_vector[max_error_index], "MeV")
print("Min error (absolute):", absolute_error[min_error_index], " at E =", E_vector[min_error_index], "MeV")
plt.title("Absolute error as a function of E")
plt.grid()

plt.show()
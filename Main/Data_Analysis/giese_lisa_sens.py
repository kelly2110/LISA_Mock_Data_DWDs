# Based on Giese et al, 2021.

import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Constants
h = 1
H_0 = (100 * h)/(3.09e19) #km sec^-1 Mpc^-1 --> 1/s --> Hz
c = 3.0e8 #m/s
L = 2.5e9 #m
f_s = c/(2*pi*L)
# Noise Parameters, P = 15, A = 3


# This is now h^2*Omega_N, without h^2 would be a factor of 4                   
def S_n(f, A, P): #Strain sensitivity
    return ((10/3)*(1+ 0.6*((2*pi*f*L/c)**2))*(P_oms(f, P) + (3 + np.cos((4*pi*f*L)/c))*P_acc(f, A)))/(((2*pi*f*L/c))**2)

def P_oms(f,P):
    return (P**2)*((10**-12)**2)*((1 + (0.002/f)**4))*(((2*pi*f)/c)**2)

def P_acc(f,A):
    return (A**2)*((10**-15)**2)*(1 + ((0.0004/f)**2))*(1 + (f/0.008)**4)*(1/(4*pi*pi*f*f*c*c))

def Omega_N(f, A, P):
    return ((2*(pi**2))/(3*H_0**2))*(f**3)*S_n(f, A, P)

if __name__ == "__main__":
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))
    Noise = np.sqrt(S_n(frequencies, 3, 15))
    Omega = Omega_N(frequencies, 3, 15)
    np.savetxt('Omega_Noise_test.csv', Omega, delimiter=',') # @Jroinde, ook hier heb ik de data in een csv file gezet om te kunen vergelijken met ptplot


# Plotting Sensitivity Curve
    plt.loglog(frequencies, Omega)
    plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Signal$')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    #plt.savefig('Lisa h^2Omega Curve Giese et al')
    plt.show()


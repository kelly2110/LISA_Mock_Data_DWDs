import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from giese_lisa_sens import Omega_N


# Constants
c = 3.0e8 #m/s
L = 2.5e9 #m
f_s = c/(2*pi*L)
h = 0.678
H_0 = (100 * h)/(3.09e19)

# DWD Noise Boileau et al.
def lisa_noise_1(f, A1, A2, alpha1, alpha2, A, P):
    
    def DWD_noise_1():
        return (A1*(f/f_s)**alpha1)/(1 + A2*(f/f_s)**alpha2) # Broken power law
            
    def S_n(): #Strain sensitivity
        return ((10/3)*(1+ 0.6*((2*pi*f*L/c)**2))*(P_oms() + (3 + np.cos((4*pi*f*L)/c))*P_acc()))/(((2*pi*f*L/c))**2)
        
    def P_oms():
        return (P**2)*((10**-12)**2)*((1 + (0.002/f)**4))*(((2*pi*f)/c)**2)

    def P_acc():
        return (A**2)*((10**-15)**2)*(1 + ((0.0004/f)**2))*(1 + (f/0.008)**4)*(1/(4*pi*pi*f*f*c*c))  
            
    def Omega_N():
        return ((2*(pi**2))/(3*H_0**2))*(f**3)*S_n()

    DWD = DWD_noise_1()
    Sens = Omega_N()
    lisa_noise_1 = DWD + Sens
    return lisa_noise_1


# DWD Noise Robson & Cornish
def lisa_noise_2(f, alpha, beta, gamma, k, A, P):
    def DWD_noise_2(): 
        def Sc():
            A = 9e-45
            return A*f**(-7/3)*np.exp(-f**alpha + beta*f*np.sin(k*f))*(1 + np.tanh(gamma*(0.00113 - f)))

        return (4*(pi)**2*f**3/(3*H_0**2))*Sc()
    
    def S_n(): #Strain sensitivity
        return ((10/3)*(1+ 0.6*((2*pi*f*L/c)**2))*(P_oms() + (3 + np.cos((4*pi*f*L)/c))*P_acc()))/(((2*pi*f*L/c))**2)
        
    def P_oms():
        return (P**2)*((10**-12)**2)*((1 + (0.002/f)**4))*(((2*pi*f)/c)**2)

    def P_acc():
        return (A**2)*((10**-15)**2)*(1 + ((0.0004/f)**2))*(1 + (f/0.008)**4)*(1/(4*pi*pi*f*f*c*c))  
            
    def Omega_N():
        return ((2*(pi**2))/(3*H_0**2))*(f**3)*S_n()

    DWD = DWD_noise_2()
    Sens = Omega_N()
    lisa_noise_2 = DWD + Sens
    return lisa_noise_2

""" if __name__ == "__main__":
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high)) 
    DWD_Noise_1 = lisa_noise_1(frequencies, 7.44e-14, 2.96e-7, -1.98, -2.6, 3, 15)
    DWD_Noise_2 = lisa_noise_2(frequencies, 0.138, -221, 521, 1680, 3, 15)
    Sensitivity = Omega_N(frequencies, 3, 15)
    plt.loglog(frequencies, DWD_Noise_1, label='DWD Noise 1')
    plt.loglog(frequencies, DWD_Noise_2, label='DWD_Noise_2')
    plt.loglog(frequencies, Sensitivity, label='LISA Sensitivity')
    plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Signal$')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.legend()
    plt.grid = 'True'
    plt.savefig('Lisa sensitivity + DWD noise fits')
    plt.show() """
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from giese_lisa_sens import S_n, Omega_N

c = 3.0e8 #m/s
L = 2.5e9 #m
f_s = c/(2*pi*L)
h = 0.678
H_0 = (100 * h)/(3.09e19)

# Where do i get my values for A and alpha from here?
def DWD_noise_2(f, alpha, beta, gamma, k): 
    def Sc(f):
        A = 9e-45
        return A*f**(-7/3)*np.exp(-f**alpha + beta*f*np.sin(k*f))*(1 + np.tanh(gamma*(0.00113 - f)))

    return (4*(pi)**2*f**3/(3*H_0**2))*Sc(f)
    
if __name__ == "__main__":
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high))   
    Omega = Omega_N(frequencies, 3, 15)
    DWD_Noise_2 = DWD_noise_2(frequencies, 0.138, -221, 521, 1680)
    total_noise = (Omega + DWD_Noise_2)

# Plotting Sensitivity Curve

    plt.loglog(frequencies, Omega)
    plt.loglog(frequencies, DWD_Noise_2)
    plt.loglog(frequencies, total_noise)
    plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Signal$')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    plt.savefig('Lisa Sensitivity + DWD Noise 2')
    plt.show()
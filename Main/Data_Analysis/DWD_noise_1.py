import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from giese_lisa_sens import S_n, Omega_N

c = 3.0e8 #m/s
L = 2.5e9 #m
f_s = c/(2*pi*L)

# Where do i get my values for A and alpha from here?
def DWD_noise_1(frequencies, A1, A2, alpha1, alpha2):
    return (A1*(frequencies/f_s)**alpha1)/(1 + A2*(frequencies/f_s)**alpha2) # Broken power law
    

if __name__ == "__main__":
    f_low = np.arange(0.00003, 0.001, 0.000001)
    f_middle = np.arange(0.001, 0.01, 0.00005)
    f_high = np.arange(0.01, 0.5, 0.001)
    frequencies = np.concatenate((f_low, f_middle, f_high)) 
    Omega = Omega_N(frequencies, 3, 15)
    DWD_Noise_1 = DWD_noise_1(frequencies, 7.44e-14, 2.96e-7, -1.98, -2.6)
    total_noise = Omega + DWD_Noise_1

# Plotting Sensitivity Curve

    plt.loglog(frequencies, Omega)
    plt.loglog(frequencies, DWD_Noise_1)
    plt.loglog(frequencies, total_noise)
    plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Signal$')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    plt.savefig('Lisa sensitivity + DWD noise 1')
    plt.show()
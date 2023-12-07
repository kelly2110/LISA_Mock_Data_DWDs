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
def DWD_noise_2(f, alpha, beta, gamma, fk): 
    def Sc(f):
        A = 9e-45
        return A*f**(-7/3)*np.exp**(-f**0.138 - 221*f*np.sin(521*f)*((1 + np.tanh(1680*(0.00113 - f)))))

    return 4*(pi)**2*f**3/(3*H_0**2)*Sc(f)
    
if __name__ == "__main__":
    frequencies = np.logspace(-6, 1, 8000)
    Noise = np.sqrt(S_n(frequencies, 3, 15))
    Omega = Omega_N(frequencies, 3, 15)
    np.savetxt('Omega_Noise_test.csv', Omega, delimiter=',')
    print(Omega)


# Plotting Sensitivity Curve

    plt.loglog(frequencies, Omega)
    plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Signal$')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    plt.savefig('Lisa h^2Omega Curve Giese et al')
    plt.show()
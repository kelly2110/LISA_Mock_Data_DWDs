import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt

H_0 = (100 * 0.687)/3.09e19

def S_h(f):
    return (10/3)*((S_1(f)/((2*pi*f)**4))+S_2(f))*R(f)

def S_1(f):
    return ((5.76e-48)*(1 + (0.4e-3/f)**2))


def S_2(f):
    return (3.6e-41) #1/Hz

def R(f):
    return 1 + (f/25e-3)**2

def Omega_S(f): #In reality this is h^2*Omega_S
    return ((2*pi**2)/(3*H_0**2))*(f**3)*S_h(f)

f = np.logspace(-6, 1, 8000)
noise = np.sqrt(S_h(f))
Omega = Omega_S(f)
plt.loglog(f, Omega)
plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Signal$')
plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
plt.ylabel(r'$h^{2}\Omega(f)$')
plt.grid = 'True'
plt.savefig(r'Lisa h^2Omega Curve')
plt.show()
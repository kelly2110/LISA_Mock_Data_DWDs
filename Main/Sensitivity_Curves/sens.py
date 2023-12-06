# Robson et al.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from math import pi


h = 0.678
H_0 = 100 * h
f = np.arange(0.0001, 1, 0.0001)
c = 3e8 # m/s
L = 2.5e6 # M km
f_s = (c/(2*pi*L))

def Omega_noise(f):
    S_p = 8.9e-23 #m^2 Hz^-1
    S_a = 9e-30*(1+16*(10e-4/f)**2 + (2e-5/f)**10)
    for frequency in f:
           Omega = (20/3)*(1/L**2)*(1 +(frequency/1.29*f_s)**2)*(S_p + (4*S_a/(2*pi*frequency)**4))

    return np.sqrt(Omega)

print(Omega_noise(f))

plt.plot(f, Omega_noise(f))
plt.xlabel('Frequency (Hz)')
plt.ylabel('OmegaSens')
plt.title('LISA Sensitivity Curve')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.savefig("Sensitivity.png")
plt.show()

#Why does it flatten off like that?
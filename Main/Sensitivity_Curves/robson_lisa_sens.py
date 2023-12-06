# Based on Robson et al, 2019.
import numpy as np
from math import pi
import matplotlib.pyplot as plt

f_s = 0.019 # Equal to (c/2*pi*L)
L = 2.5e9 # Lisa arm length
f = np.logspace(-5, 1, 8000) # Frequency range
P = 15
A = 3

def P_oms(f, P): # Single link optrical metrology noise
    return (P**2)*((10**-12)**2)*(1+(0.002/f)**4)

def P_acc(f, A): # Single test mass acceleration noise
    return (A**2)*((10**-15)**2)*(1+(0.0004/f)**2)*(1+(f/0.008)**4)

def P_n(f): # Total noise in Michelson-style LISA data channel, not necessary
      return (P_oms(f,P) + 2*(1 + np.cos(f/f_s)**2)*P_acc(f, A)/(2*pi*f)**4)/L**2

def R(f): # Transfer function, not necessary
    return 3/10/(1 + 6/10*(f/f_s)**2)

def S_n(f): # Sensitivity curve analytic fit
    #return P_n(f)/R(f)
    return (10/(3*L**2))*((P_oms(f,P))+2*(1+(np.cos(f/f_s)**2))*(P_acc(f,A)/((2*pi*f)**4)))*(1+(0.6*((f/f_s)**2)))


''' Plotting Sensitivity Curve'''

noise = np.sqrt(S_n(f))
print(noise[0:20])
plt.loglog(f, noise)
plt.title('LISA Noise Signal')
plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
plt.ylabel(r'$\sqrt{S_{n}(f)}$' + "  " +r'$(1/\sqrt{Hz})$')
plt.grid = 'True'
plt.savefig('Lisa Noise Signal Robson et al.')
plt.show()
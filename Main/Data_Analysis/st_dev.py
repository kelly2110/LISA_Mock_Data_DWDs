import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Constants
h = 0.678
H_0 = (100 * h)/(3.09e19)#km sec^-1 Mpc^-1 --> 1/s --> Hz
c = 3.0e8 #m/s
L = 2.5e9 #m
f_s = c/(2*pi*L)
# Noise Parameters, P = 15, A = 3

def Omega_N(f, A, P):
    return ((2*(pi**2))/(3*H_0**2))*(f**3)*S_n(f, A, P)
                   
def S_n(f, A, P): #Strain sensitivity
    return ((10/3)*(1+ 0.6*((2*pi*f*L/c)**2))*(P_oms(f, P) + (3 + np.cos((4*pi*f*L)/c))*P_acc(f, A)))/(((2*pi*f*L/c))**2)

def P_oms(f,P):
    return (P**2)*((10**-12)**2)*((1 + (0.002/f)**4))*(((2*pi*f)/c)**2)

def P_acc(f,A):
    return (A**2)*((10**-15)**2)*(1 + ((0.0004/f)**2))*(1 + (f/0.008)**4)*(1/(4*pi*pi*f*f*c*c))


def calculate_std_deviation(frequencies, num_trials=1):
    std_deviation_per_frequency = []

    for _ in range(num_trials):
        noise_signal = Omega_N(frequencies, 3, 15)
        std_deviation = np.std(noise_signal)
        std_deviation_per_frequency.append(std_deviation)

    return std_deviation_per_frequency

# Example usage:
# Replace frequencies_array with the actual array of frequencies you want to analyze
frequencies_array = np.logspace(-6, 1, 10000) # Example frequencies from 10 to 1000 Hz
std_deviation_values = calculate_std_deviation(frequencies_array)

# Print or use std_deviation_values as needed
print("Standard Deviation for each frequency:")
for freq, std_dev in zip(frequencies_array, std_deviation_values):
    print(f"Frequency: {freq} Hz, Standard Deviation: {std_dev}")
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
import scipy.integrate as integrate
from scipy.integrate import trapz
T_obs = 9.46e7 #s

# Testing whether SNR gives the correct results

columns = ['f', ' omegaSens', ' omegaSW']
data = pd.read_csv("curvedata.csv", usecols=columns)
array = data.to_numpy()
f = array[:,0]
N= array[:,1]
GW = array[:,2]

# SNR Function
def calculate_snr_(GW, Noise, frequency):
        integrand = (GW**2)/(Noise**2)
        integral_result = trapz(integrand, frequency)
        snr = np.sqrt(T_obs*integral_result)
        print(f"The SNR for this GW signal is {snr}")
        return snr

if  __name__ == '__main__':
        SNR = calculate_snr_(f, N, GW)
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
import scipy.integrate as integrate
from scipy.integrate import trapz, quad, trapezoid

T_obs = 9.46e7  # s

# Testing whether SNR gives the correct results with PTPLOT data
""" columns = ['f', ' omegaSens', ' omegaSW']
data = pd.read_csv("ptplotdata.csv", usecols=columns)
array = data.to_numpy()
f = array[:, 0]
N = array[:, 1]
GW = array[:, 2]
 """
# SNR Function
def calculate_snr_(GW, Noise, frequencies):
        integrand = (GW/Noise)**2
        print(integrand.shape)
        integral_result = trapezoid(integrand, frequencies)
        snr = np.sqrt(T_obs * integral_result)
        print(f"The SNR for this GW signal is {snr}")
        return snr


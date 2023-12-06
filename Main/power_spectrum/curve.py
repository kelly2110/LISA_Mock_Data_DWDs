#Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import *
from math import pi
from scipy.integrate import odeint
from scipy.integrate import simps

#Import code for calculating K
#from power_spectrum.KineticEnergyFraction import KineticEnergyFraction



#Constants
g_s = 106.75
T_s = 200 #GeV
z_p = 10
H_s = 3
R_s = 5
c_s = 343 #m/s
alpha = 0.4
v_w = 0.9
beta = 50 


#Equation 30
def C(s):
    return s**3*(7/(4+3*s**2))**(7/2)

#Equation 31
Beta = 1/(H_s*R_s)
def f_p(z_p, T_s, g_s):
    return Beta*(z_p/10)*(T_s/100)*(g_s/100) #muHz

#Equation 20
def F_GW(g_s):
    return (3.57e-5)*(100/g_s)**(1/3) #g_* is constant during the entire course of this project

#Equation 6 From Beta --> R_star
def B_to_R_s(beta, v_w):
    return (8*pi)**(1/3)*v_w/beta

# From R_star --> Beta
def R_s_to_B(r_s, v_w):
  return (8*pi)**(1/3)*v_w/r_s


#Equation 29
def Power_Spectra(H_s, R_s, c_s, f):
    return 0.687*F_GW(g_s)*KineticEnergyFraction(alpha, v_w)*(H_s*R_s(Beta, v_w, c_s))*10e-2*C/f_p

print(F_GW(g_s))




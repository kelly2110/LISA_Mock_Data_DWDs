# Based on Caprini et al., 


# Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi, sqrt
from kinetic_energy_fraction import KineticEnergyFraction
from giese_lisa_sens import Omega_N

# Frequency range
f_low = np.arange(0.00003, 0.001, 0.000001)
f_high = np.arange(0.001, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_high))
H_0 = (100 * 0.687)/3.09e19 # Hubble Constant

# Conversions of Bubble radius to beta and vice versa

def Rstar_to_Beta(rstar, vw):
  return (8*pi)**(1/3)*vw/rstar

def Beta_to_Rstar(beta, vw):
  return (8*pi)**(1/3)*vw/beta

# Define class for Powerspectrum, to be able to create multiple objects later

class PowerSpectrum:

  def __init__(self,
               alpha, 
               betaoverH,
               Tstar,
               vw):

      self.alpha = alpha
      self.betaoverH = betaoverH
      self.Tstar = Tstar
      self.vw = vw
      self.H_rstar = Beta_to_Rstar(self.betaoverH, self.vw) 
      self.h = 0.687
      self.zp = 10
      self.gs = 100
      self.H_0 = 68.7/3.086e19 #????
      self.cs = 1.0/sqrt(3.0) 
      self.K = KineticEnergyFraction(self.alpha, self.vw)
      self.hstar = 16.5e-6*(self.Tstar/100)*np.power(self.gs/100, (1/6))  

   # Not dependent on the power spectrum itself, therefore static
  @staticmethod
  def C(s):
    return (s**3)*((7/(4+3*(s**2)))**(7/2))
                                  
  #def RxH(self):
    #return self.H_0*(8*pi)**(1/3)*self.vw/(self.betaoverH)
                
  def fp_0(self):
    return ((26e-6)*(1/self.H_rstar)*(self.zp/10)*(self.Tstar/100)**(self.gs/100)**(1/6))#milliHz
    
  @staticmethod                           
  def Fgw_0():
    return 3.57e-5*((100/106.75))**(1/3) # this is a constant

#Returns h^2 Omega_GW
  def Omega_GW(self, f):  
    fp = f/self.fp_0()
    return self.h*self.h*2.061*self.Fgw_0()*0.012*(self.K)**2*self.H_rstar*self.C(fp)
  

if __name__ == "__main__":

  # Object Creation
  P1 = PowerSpectrum(0.6, 50, 180, 0.8)
  GW = np.array(P1.Omega_GW(frequencies))
  np.savetxt('Gravitational_Wave.csv', GW, delimiter=',')
  print(GW)
  Noise = Omega_N(frequencies, 3, 15) 
  print("The peak frequency of the PS is:", P1.fp_0()) #Peak Frequency

# PS plot
  plt.loglog(frequencies, GW, color = 'g', label='GW signal')
  plt.loglog(frequencies, Noise, color ='m', label='Noise signal')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel(r'$\Omega$')
  plt.title('Power Spectrum')
  plt.grid(True)
  plt.savefig("PS.png")
  plt.show()

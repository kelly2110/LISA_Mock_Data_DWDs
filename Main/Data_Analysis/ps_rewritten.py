# Based upon the equations given in "Detecting gravitational waves from a cosmological first-order phase transition with LISA: an update', Caprini et al., 2020.
# and "Shape of the acoustic gravitational wave power spectrum from a first order phase transition", Hindmarsh et al., 2020.

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt
from kinetic_energy_fraction import KineticEnergyFraction #Jorinde's K calculation
from giese_lisa_sens import Omega_N # Dit is de Sensitivity curve uit jouw paper, die ik overal gebruik @Jorinde
import time
import pandas as pd
from SNR_calc import calculate_snr_ # Mijn SNR functie staat hierin!

# Frequency range
""" f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high)) """
frequencies = np.logspace(-6, 1, 2000)
H_0 = (100 * 1) / 3.09e19  # Hubble Constant

# Conversions of Bubble radius to beta and vice versa
def Rstar_to_Beta(rstar, vw):
    return (8*pi)**(1/3)*vw/rstar

def Beta_to_Rstar(beta, vw):
    return ((8.0*pi)**(1.0/3.0))*vw/beta

# Define class for Powerspectrum, to be able to create multiple objects later
class PowerSpectrum:
    def __init__(self, alpha, betaoverH, Tstar, vw):
        self.alpha = alpha
        self.betaoverH = betaoverH
        self.Tstar = Tstar
        self.vw = vw
        self.H_rstar = Beta_to_Rstar(self.betaoverH, self.vw)
        self.cs =  1/np.sqrt(3)
        self.h = 0.678
        self.zp = 10
        self.gs = 106.75
        self.H_0 = 100 / 3.09e19  
        self.T_sh = 2 # Not the correct value because I would need Ubarf, but works as long as << 1
        self.K = KineticEnergyFraction(self.alpha, self.vw) # Changes per PS
        self.hstar = 16.5e-6 * (self.Tstar / 100) * np.power(self.gs / 100, (1 / 6))
        self.Fgw_0 = 3.57e-5*((100/self.gs))**(1/3) # Constant
        self.Amp = self.calculate_amplitude()
        

    def calculate_amplitude(self): 
        if self.H_rstar*self.T_sh < 1:
            return self.h*self.h*2.061*self.Fgw_0*0.012*(self.K)**(2)*(self.H_rstar/self.cs)
        else:
            return self.h*self.h*2.061*self.Fgw_0*0.012*(self.K)**(3/2)*(self.H_rstar/np.sqrt(self.cs))**2

    @staticmethod  # Not dependent on the power spectrum itself, therefore static
    def C(s):
        return (s**3)*((7/(4+3*(s**2)) )**(7/2))
    
    def fp_0(self):
        return ((26.0e-6)*(1.0/self.H_rstar)*(self.zp/10.0)*(self.Tstar/100)*(self.gs/100)**(1.0/6.0))
  
    # Returns h^2 Omega_GW
    def Omega_GW(self, frequencies, Amp, f_peak):
        return Amp * self.C(frequencies/f_peak) # Amplitude times spectral shape


if __name__ == "__main__":

  # Object Creation
  start_time = time.time()
  P1 = PowerSpectrum(0.001, 50, 180, 0.8)
  GW = (P1.Omega_GW(frequencies, P1.Amp, P1.fp_0())) # Dit is mijn GW signaal
  np.savetxt('Omega_GW_test.csv', GW, delimiter=',') # Slaat op als csv zodat ik de waardes kon vergelijken met PTPLOT
  print(GW)
  Noise = Omega_N(frequencies, 3, 15)  # Mijn sensitivity curve
  print(f"The peak frequency of the PS is: {P1.fp_0()} mHz")
  print(f"The amplitude of the PS is: {P1.calculate_amplitude()}")


# @Jorinde dit is de csv file van PTPLOT met dezelfde parameters, ik lees hem nu in via pandas
  columns = ['f', ' omegaSens', ' omegaSW']
  data = pd.read_csv("alpha0_001.csv", usecols=columns)
  array = data.to_numpy()
  f = array[:, 0]
  N_ptp = array[:, 1]
  GW_ptp = array[:, 2]
  SNR_1 = calculate_snr_(GW, N_ptp, frequencies) 
  SNR_2 = calculate_snr_(GW_ptp, N_ptp, f) # SNR calculation gebaseerd op PTPLOT data, dit komt inderdaad overeen met wat in de grafiek van PTPLOT staat
 
 
 # Timing runtime
  end_time = time.time()
  run_time = end_time - start_time
  print("Runtime:", run_time)


 # @Jorinde hier staat de ratio tussen mijn signaal en het PTPLOT signaal, voor zowel GW als Sensitivity
  print("Ratio GW signal:", 1/(GW_ptp[0:1]/GW[0:1]))
  print("Ratio Noise signal:", 1/(N_ptp[0:1]/Noise[0:1]))
 
# PS plot
  plt.loglog(frequencies, GW, color = 'g', label='GW signal')
  plt.loglog(frequencies, Noise, color ='m', label='Noise signal')
  plt.loglog(f, N_ptp, color = 'r', label='PTPLOT Sensitivity')
  plt.loglog(f, GW_ptp, color = 'b', label= 'PTPLOT GW signal')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel(r'$\Omega$')
  plt.title('Power Spectrum')
  plt.legend()
  plt.show()
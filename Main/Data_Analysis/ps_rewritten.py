import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt
from kinetic_energy_fraction import KineticEnergyFraction
from giese_lisa_sens import Omega_N
import time


# Frequency range
f_low = np.arange(0.00003, 0.001, 0.000001)
f_middle = np.arange(0.001, 0.01, 0.00005)
f_high = np.arange(0.01, 0.5, 0.001)
frequencies = np.concatenate((f_low, f_middle, f_high))
H_0 = (100 * 0.687) / 3.09e19  # Hubble Constant

# Conversions of Bubble radius to beta and vice versa
def Rstar_to_Beta(rstar, vw):
    return (8*pi) ** (1/3)*vw/rstar

def Beta_to_Rstar(beta, vw):
    return (8*pi)**(1/3)*vw/beta

# Define class for Powerspectrum, to be able to create multiple objects later
class PowerSpectrum:
    def __init__(self, alpha, betaoverH, Tstar, vw):
        self.alpha = alpha
        self.betaoverH = betaoverH
        self.Tstar = Tstar
        self.vw = vw
        self.H_rstar = Beta_to_Rstar(self.betaoverH, self.vw)
        self.h = 0.687
        self.zp = 10
        self.gs = 106.75
        self.H_0 = 68.7 / 3.086e19  
        self.cs = 1.0 / sqrt(3.0)
        self.K = KineticEnergyFraction(self.alpha, self.vw) # Changes per PS
        self.hstar = 16.5e-6 * (self.Tstar / 100) * np.power(self.gs / 100, (1 / 6))
        self.Fgw_0 = 3.57e-5*((100/self.gs))**(1/3) # Constant
        self.Amp = self.calculate_amplitude()

    def calculate_amplitude(self): 
        return self.h*self.h*2.061*self.Fgw_0*0.012*(self.K)**2*self.H_rstar 

    @staticmethod  # Not dependent on the power spectrum itself, therefore static
    def C(s):
        return (s**3)*((7/(4+3*(s**2)))**(7/2))
    
    def fp_0(self):
        return ((26e-6)*(1/self.H_rstar)*(self.zp/10)*(self.Tstar/100)**(self.gs/100)**(1/6))
  
    # Returns h^2 Omega_GW
    def Omega_GW(self, frequencies, Amp, f_peak):
        return Amp * self.C(frequencies/f_peak) # Amplitude times spectral shape
    

if __name__ == "__main__":

  # Object Creation
  start_time = time.time()
  P1 = PowerSpectrum(0.6, 50, 180, 0.8)
  GW = np.array(P1.Omega_GW(frequencies, P1.Amp, P1.fp_0()))
  np.savetxt('Omega_GW_test.csv', GW, delimiter=',')
  print(GW.shape)
  print(GW)
  Noise = Omega_N(frequencies, 3, 15) 
  print(f"The peak frequency of the PS is: {P1.fp_0()} mHz")
  print(f"The amplitude of the PS is: {P1.calculate_amplitude()}")
  end_time = time.time()
  run_time = end_time - start_time
  print("Runtime:", run_time)

# PS plot
  plt.loglog(frequencies, GW, color = 'g', label='GW signal')
  plt.loglog(frequencies, Noise, color ='m', label='Noise signal')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel(r'$\Omega$')
  plt.title('Power Spectrum')
  plt.grid(True)
  plt.savefig("PS.png")
  plt.show()
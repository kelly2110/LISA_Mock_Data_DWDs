# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Used to minimize for the parameters
from giese_lisa_sens import S_n, P_oms, P_acc, Omega_N
from ps_rewritten import PowerSpectrum
from combined_data_gen import make_data_DWD_1, make_data_DWD_2, make_data_no_DWD
from SNR_calc import calculate_snr_
import time



    # Create objects with different alphas
    # calculate the three chi squared functions
    # save results for each chi squared in a separate file/folder
    # loop for beta/vw as well
    
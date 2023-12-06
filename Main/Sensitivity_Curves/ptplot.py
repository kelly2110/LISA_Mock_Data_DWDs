import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from math import *

columns = ['f', ' omegaSens', ' omegaSW']
data = pd.read_csv("test.csv", usecols=columns)
print(data)
data.plot(x='f', y=[' omegaSens', ' omegaSW'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('OmegaSens/OmegaSW')
plt.title('LISA Sensitivity Curve + GW Signal')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.savefig("a_01.png")
plt.show()


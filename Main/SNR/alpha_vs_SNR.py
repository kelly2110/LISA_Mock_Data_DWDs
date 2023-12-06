import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import *
from math import pi

alpha = np.arange(0.1, 1.1, 0.1)
SNR = [71.06715661647287, 415.6760305997594, 1060.9948009258226, 1951.84884669769, 3021.39188410795, 4211.143194271651, 5474.978812710167, 6778.205964038759, 8095.420319891287, 9408.379182264154]



plt.plot(alpha, SNR)
plt.xlabel('Alpha')
plt.ylabel('SNR')
plt.title('Alpha vs SNR')
plt.grid(True)
plt.savefig("ok.png")
plt.show
import numpy as np
import matplotlib.pyplot as plt


def DWD_noise_1(frequencies):
    return 

if __name__ == "__main__":
    frequencies = np.logspace(-6, 1, 8000)
    Noise = np.sqrt(S_n(frequencies, 3, 15))
    Omega = Omega_N(frequencies, 3, 15)
    np.savetxt('Omega_Noise_test.csv', Omega, delimiter=',')
    print(Omega)


# Plotting Sensitivity Curve

    plt.loglog(frequencies, Omega)
    plt.title(r'$LISA$' + " " + r'$\Omega$' + " " + '$Signal$')
    plt.xlabel(r'$Frequency$' + "  " + r'$(Hz)$')
    plt.ylabel(r'$h^{2}\Omega(f)$')
    plt.grid = 'True'
    plt.savefig('Lisa h^2Omega Curve Giese et al')
    plt.show()
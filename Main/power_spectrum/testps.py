class PowerSpectrum:

    def __init__(self,
                 BetaoverH = None,
                 Tstar = 180.0,
                 gstar = 100,
                 vw = None,
                 adiabaticRatio = 4.0/3.0,
                 zp = 10,
                 alpha = None,
                 kturb = 1.97/65.0,
                 H_rstar = None,
                 ubarf_in = None):


        self.BetaoverH = BetaoverH
        self.Tstar = Tstar
        self.gstar = gstar
        self.vw = vw
        self.adiabaticRatio = adiabaticRatio
        self.zp = zp
        self.alpha = alpha
        self.kturb = kturb

        self.hstar = 16.5e-6*(self.Tstar/100.0) \
                     *np.power(self.gstar/100.0,1.0/6.0)

        # Either take ubarf_in as-is, or calculate ubarf from the wall velocity
        if (vw is not None) and (ubarf_in is None):
            self.ubarf = ubarf(vw, alpha, adiabaticRatio)
        elif (vw is None) and (ubarf_in is not None):
            self.ubarf = ubarf_in
        else:
            raise ValueError("Either ubarf_in or vw must be set, but not both")

        # Calculate typical bubble radius
        if (H_rstar is None) and (BetaoverH is not None):
            self.H_rstar = beta_to_rstar(self.BetaoverH, self.vw)
        elif (H_rstar is not None) and (BetaoverH is None):
            self.H_rstar = H_rstar
        else:
            raise ValueError("Either H_rstar or BetaoverH must be set, but not both")

        # Compute shock time
        self.H_tsh = self.H_rstar/self.ubarf

    # This function does not depend on the power spectrum itself, and so does
    # not inherit the class instance information (no self in arguments).
    # Note that this function used to be called Ssw, but it was renamed in 2023
    # to match the notation used in equation 36 of 1704.05871.
    # Function C(f)
    @staticmethod
    def Csw(fp, norm=1.0):
        """Calculate spectral shape for gw from sound waves

        For a given peak frequency, calculate spectral shape of a single broken
        power law fit to simulation results for gw from sound waves.
        """

        return norm*np.power(fp,3.0) \
            *np.power(7.0/(4.0 + 3.0*np.power(fp,2.0)),7.0/2.0)

    def get_shocktime(self):
        """Calculate shock time
        """

        return self.H_tsh
    
    # This follows equation 43 in 1704.05871. Note that the numerical prefactor
    # is absorbed in the definition of beta_to_rstar() above;
    # (1/(H_n*R_*)) = 1/((8*pi)^{1/3}*vw/BetaoverH) .
    def fsw(self):
        """Calculate true peak frequency
        """

        return (26.0e-6)*(1.0/self.H_rstar)*(self.zp/10.0) \
            *(self.Tstar/100)*np.power(self.gstar/100,1.0/6.0)        

    # This follows equations 39 - 45 in 1704.05871 (and the paper erratum)
    def power_spectrum_sw(self, f):
        """Calculate power spectrum from sound waves for a given frequency f

        This function follows equation 45 (erratum equation 2) of
        1704.05871.
        """

        # This is based on equation 45 in 1704.05871, with the numerical
        # prefactor coming from 0.68*(3.57e-5)*(8*pi)^(1/3)*0.12 = 8.5e-6
        # (=0.68*Fgw0*geometric*Omtil)
        #
        # Using equation R_* = (8*pi)^{1/3}*vw/beta (section VI, same paper),
        # thus: H_n*R_* = (8*pi)^{1/3}*vw/BetaoverH
        #
        # See fsw() method, and definition of beta_to_rstar()
        fp = f/self.fsw()

        # Some of the equations below were derived assuming this value for h,
        # we add it here to remove the h dependence from the final results
        h_planck = 0.678

        # Equations 39 and 45 in 1704.05871 are missing the factor of 3 [typo];
        # and there is no h_planck in eq 45 (it is implicit in the RHS).
        # Note also typo below eq 45, 0.12 -> 0.012 for OmTilde.
        #
        # The resulting 3*0.687 = 2.061 prefactor is also explained in equation
        # 2 of the erratum.
        #
        # Fgw0 is 3.57e-5*(100/hstar)^(1/3), and implicitly includes
        # Omega_photons. The implicit Hubble constant dependence of equation
        # 45 (erratum equation 2) comes from Omega_photons. Multiplying both
        # sides by h_planck removes that, and so the result does not depend
        # on a particular measurement of the Hubble constant.
        #
        # Thus, this returns h^2 OmGW, which does not depend on a
        # particular value of the Hubble constant.
        return h_planck*h_planck*3.0 \
            *0.687*3.57e-5*0.012*np.power(100.0/self.gstar,1.0/3.0) \
            *self.adiabaticRatio*self.adiabaticRatio \
            *np.power(self.ubarf,4.0)*self.H_rstar*self.Csw(fp)    


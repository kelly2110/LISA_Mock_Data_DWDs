# AIC calculation, adjust k as necessary
def calculate_aic(chi,k):
    AIC = chi + 2*k
    print("The AIC value is:", AIC)
    return AIC

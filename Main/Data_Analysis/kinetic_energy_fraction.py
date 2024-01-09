""" Calculation of the Kinetic Energy Fraction for the FOPT GW signal"""
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import simps
import matplotlib.pyplot as plt 

def mu(a,b):
  return (a-b)/(1.-a*b)

# If the wall velocity is smaller than the sos, solution is deflagration, so vm = vw
# Compare eq. 19 of bag eos paper: disc is the argument of the sqrt appearing in the expression
# for vm in terms of vp
# If the solution would be a detonation, vp=vw. Whenever the sqrt becomes imaginary for vp=vw
# is the transition to hybrid.
def getvm(al,vw,cs2b):
  if vw**2<cs2b:
    return (vw,0)
  cc = 1.-3.*al+vw**2*(1./cs2b+3.*al)
  disc = -4.*vw**2/cs2b+cc**2
  if (disc<0.)|(cc<0.):
    return (np.sqrt(cs2b), 1)
  return ((cc+np.sqrt(disc))/2.*cs2b/vw, 2)

#This is al+
def getal(vp,vm,cs2b):
  return (vp/vm-1.)*(vp*vm/cs2b - 1.)/(1-vp**2)/3.

#similar reasoning as above for vm
def getvp(al,vm,cs2b):
  cc = 0.5*(vm/cs2b + 1./vm)
  disc = (1./cs2b+3.*al)*(3.*al-1.)+cc**2
  if (disc<0.):
    print("neg disc in vp: ",al,vm,cs2b,cc)
    return 0.
  return (cc-np.sqrt(disc))/(1./cs2b+3.*al)

#Differential equations for v, xi and w.
def dfdv(xiw, v, cs2):
  xi, w = xiw
  dxidv = (mu(xi,v)**2/cs2-1.)
  dxidv *= (1.-v*xi)*xi/2./v/(1.-v**2) 
  dwdv = (1.+1./cs2)*mu(xi,v)*w/(1.-v**2)
  return [dxidv,dwdv]

def getwow(a,b):
  return a/(1.-a**2)/b*(1.-b**2)

def getKandWow(vw,v0,cs2):
   
  
  if v0==0:
    return 0,1

  n = 8*1024  # change accuracy here
  vs = np.linspace(v0, 0, n)
#solution of differential equation for xi and wow in terms of v
#initial conditions corresponds to v=v0, xi = vw and w = 1 (which is random)
  sol = odeint(dfdv, [vw,1.], vs, args=(cs2,))
  xis, wows = (sol[:,0],sol[:,1])

  ll=-1
  if mu(vw,v0)*vw<=cs2:
    #find position of the shock
    ll=max(int(sum(np.heaviside(cs2-(mu(xis,vs)*xis),0.0))),1)
    vs = vs[:ll]
    xis = xis[:ll]
    wows = wows[:ll]/wows[ll-1]*getwow(xis[-1], mu(xis[-1],vs[-1]))

  Kint = simps(wows*(xis*vs)**2/(1.-vs**2), xis)

  return (Kint*4./vw**3, wows[0])

def alN(al,wow,cs2b,cs2s):
  da = (1./cs2b - 1./cs2s)/(1./cs2s + 1.)/3.
  return (al+da)*wow -da

def getalNwow(vp,vm,vw,cs2b,cs2s):
  Ksh,wow = getKandWow(vw,mu(vw,vp),cs2s) 
  #print (Ksh,wow)
  return (alN(getal(vp,vm,cs2b),wow,cs2b,cs2s), wow) 

def kappaNuMuModel(cs2b,cs2s,al,vw):
  #print (cs2b,cs2s,al,vw)
  vm, mode = getvm(al,vw,cs2b)
  #print (vm**2, mode)
  if mode<2:
    almax,wow = getalNwow(0,vm,vw,cs2b,cs2s)
    if almax<al:
      print ("alpha too large for shock")
      return 0;

    vp = min(cs2s/vw,vw) #check here
    almin,wow = getalNwow(vp,vm,vw,cs2b,cs2s)
    if almin>al: #minimum??
      print ("alpha too small for shock")
      return 0;

    iv = [[vp,almin],[0,almax]]
    while (abs(iv[1][0]-iv[0][0])>1e-7):
      vpm = (iv[1][0]+iv[0][0])/2.
      alm = getalNwow(vpm,vm,vw,cs2b,cs2s)[0]
      if alm>al:
        iv = [iv[0],[vpm,alm]]
      else:
        iv = [[vpm,alm],iv[1]]
      #print iv
    vp = (iv[1][0]+iv[0][0])/2.
    Ksh,wow = getKandWow(vw,mu(vw,vp),cs2s)
  
  else:
    vp = vw 
    Ksh,wow = (0,1)
  
  if mode>0:
    Krf,wow3 = getKandWow(vw,mu(vw,vm),cs2b)
    Krf*= -wow*getwow(vp,vm)
  else:
    Krf = 0

  return (Ksh + Krf)/al

def KineticEnergyFraction(al,vw):
  return kappaNuMuModel(1/3.,1/3.,al,vw)*al/(al+1.)

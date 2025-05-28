#coding: utf-8
"""
Created on 14/05/2025
Domaine : Mécanique
Sous-Domaine : Mécanique MPSI
Chapitre : Gravitation
Appellation : 3eme loi de Kepler et régression linéaire
@author: Antonin
"""
#-Clean Workspace
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
#
#-Import libraries
import numpy as np
import scipy
# import scipy.interpolate as spinterp
#
#-Plot figures
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.close('all')
# mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.family'] = 'STIXGeneral'
# mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['figure.dpi'] = 110
# mpl.rcParams['axes.grid'] = True
# mpl.rcParams['axes.grid.axis'] = 'both'
# mpl.rcParams['axes.grid.which'] = 'minor'
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['figure.subplot.hspace']=  0.6
mpl.rcParams['lines.linewidth']  = 1.4
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['figure.figsize'] = 1.5*7.4, 1.5*5.8
from mpl_toolkits.mplot3d import Axes3D
#
nf = 0 #index for figures
"""Show linearity"""
vec_T = np.array([0.241, 0.615,1.000,1.880,11.85,29.47,84.29,164.78,247.80])
vec_a = np.array([0.387,0.723,1.000,1.523,5.202,9.554,19.218,30.109,39.60])
nf = nf + 1
plt.figure(nf)
plt.loglog(vec_a,vec_T,'*')
plt.xlabel("$x=\log((a/a_0)^2)$")
plt.ylabel("$y=\log((T/T_0)^2)$")
ax = plt.subplot()
ax.grid(which='major', color='silver', linewidth=0.5)
ax.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
ax.minorticks_on()
plt.show()
#
"""Linear Regression"""
#
vec_x = np.log10(vec_a/1)# a en u.a.
vec_y = np.log10(vec_T/1)# T en u.a.
reg = scipy.stats.linregress(vec_x,vec_y)
A_estim = float(reg.intercept)
print(A_estim)
B_estim = float(reg.slope)
print(B_estim)
r = float(reg.rvalue)
print(r)
#
"""Statistics general"""
#
m_y = np.mean(vec_y)
m_x = np.mean(vec_x)
s_yy = np.sum((vec_y - m_y)**2)
s_xx = np.sum((vec_x - m_x)**2)
df = np.size(vec_T) - 2
df = int(df)
#
P = 95/100 #probability for the student coefficient
alpha = 1 - P
df = 7 #number of degrees of freedom
p = 1 - alpha/2 #order of the quantile
qt = scipy.special.stdtrit(df,p,out=None) #student quantile corresponding to the student coefficient
qt = float(qt)
#
"""Statistics for Beta"""
s_Bestimator = np.sqrt((1-r**2)/df)*(s_yy/s_xx)**0.5
#-Compare with the direct calculation given by scipy.stats.linregress.stderr 
s_Bestimator_scipy = reg.stderr
diff = s_Bestimator - s_Bestimator_scipy
print(diff)
#
Delta_beta = qt * s_Bestimator
print(Delta_beta)
#
"""Statistics for alpha"""
N=7
s_Aestimator = np.sqrt((1-r**2)/df)*np.sqrt(s_yy*(1/N + m_x**2/s_xx))
#
Delta_alpha = qt * s_Aestimator
print(Delta_alpha)
#
"""Statistics for Co"""
#
M_sun = 1.98892*10**30 #kg
Delta_M_sun = 8.8*10**-5 * M_sun #
#
a_0 = 149597870700 #m
T_0 = 365.25636042*24*3600 #s
Co = 4*np.pi**2/M_sun
Co = Co * a_0**3/T_0**2
#
Delta_Co = Co * Delta_M_sun/M_sun
#
"""Incertitude sur G"""
Delta_G = (10**(-2*A_estim/1) * Delta_Co)**2
Delta_G = Delta_G + (((-2/1)*Co*np.log(10))**2 * 10**(-2*(-2/1)*A_estim) * Delta_alpha)**2
Delta_G = Delta_G**0.5
print(Delta_G)
#
G = Co * 10**(-2*A_estim)
print(G)
"""Plot linear regression"""
vec_linR = 10**A_estim * vec_a**B_estim
plt.loglog(vec_a,vec_linR)
plt.title("Vérification de la troisième loi de Kepler avec les orbites du système solaire")
plt.legend(["data", "$y=\\beta.x+\\alpha,\\; \\beta=%.2f \\pm %.2f$ à $68\\%%$" % (B_estim, Delta_beta)])
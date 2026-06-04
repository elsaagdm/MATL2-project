#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:05:30 2026

@author: nika
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit

#===========================================================================
# DATA 
#===========================================================================

#Extracting data: Pantheon + SH0ES
pan = np.genfromtxt("data/Pantheon+SH0ES.dat")
zHD = pan [1:,2]
z = zHD
mu = pan [1:,10]
mu_err = pan [1:,11]

#Extracting data: DES 5YR
des = np.genfromtxt("data/DES_data.csv")
probia_beams =  des [9:,9] 
des = des [9:]
des = des [probia_beams >= 0.95]
z_des = des [9:,3]
mu_des = des [9:,5]
mu_err_des = des [9:,6]

#===========================================================================
# PLOTTING data - Hubble diagram
#===========================================================================

#Normal scale - both data sets 
plt.errorbar(z, mu, yerr=mu_err, fmt = 'o', ecolor ='gray', markersize = 3, label = "Pantheon+SH0ES")
plt.errorbar(z_des, mu_des, yerr=mu_err_des, fmt = 'o', ecolor ='gray', markersize = 3, label = "DES 5YR")
plt.title("Hubble Diagram (Supernovae Ia)")
plt.xlabel("Redshift (z)")
plt.ylabel("Distance modulus ($\mu$)")
plt.legend()
file1 = input("Please enter the name of the file where the Hubble diagram (normal scale) with the two datasets will be saved: ")
plt.savefig(file1)
plt.show()

#From now on we only use the Pantheon+SH0ES dataset 

#Normal scale
plt.errorbar(z, mu, yerr=mu_err, fmt = 'o', markersize = 3, ecolor ='gray', label = "Supernova Ia")
plt.title("Hubble Diagram (Pantheon+SH0ES data)")
plt.xlabel("Redshift (z)")
plt.ylabel("Distance modulus ($\mu$)")
plt.legend()
file1 = input("Please enter the name of the file where the Hubble diagram (normal scale) will be saved: ")
plt.savefig(file1)
plt.show()

#Logarithmic scale
plt.errorbar(z, mu, yerr=mu_err, fmt = 'o', markersize = 3, ecolor ='gray', label = "Supernova Ia")
plt.title("Hubble Diagram (Pantheon+SH0ES data and theoretical models)")
plt.xscale('log')
plt.xlabel("Redshift (z) (log scale)")
plt.ylabel("Distance modulus ($\mu$)")
plt.legend()
file2 = input("Please enter the name of the file where the Hubble diagram (logarithmic scale) will be saved: ")
plt.savefig(file2)
plt.show()

#===========================================================================
#THEORETICAL FRAMEWORK MODEL PREDICTIONS
#===========================================================================
 #Constants 
hubble = 73 #Units: km/s/Mpc (Hubble constant)
c = 299792.458 #Units: km/s (Speed of light in vacuum)

 #Equations for model-based predictions
def luminosity_distance(z, matter, dark_e, hubble):
    if matter == 0 and dark_e == 0:   #Milne
        return ((c*z)/hubble)*(1+z/2)
    else:   #Flat universes
        integral = quad(lambda z_prime: 1/(np.sqrt(matter*(1+z_prime)**3+dark_e)), 0, z)[0]
        return ((c*(1+z))/hubble)*integral #Radiation contribution is negligible

def mu_theory(z, matter, dark_e, hubble):
    dL = np.array([luminosity_distance(z, matter, dark_e, hubble) for z in zHD])
    return 5*np.log10(dL)+25

#DEFINING THE MODELS

# ------Flat universe--------
 
#LambdaCDM:
matter_lcdm = 0.28
dark_e_lcdm = 0.72
mu_theory_lcdm = mu_theory(z, matter_lcdm, dark_e_lcdm, hubble)

#Einstein-de Sitter model:
matter_eds = 1.0
dark_e_eds = 0.0
mu_theory_eds= mu_theory(z, matter_eds, dark_e_eds, hubble)

#de Sitter model
matter_ds = 0.0
dark_e_ds = 1.0
mu_theory_ds = mu_theory(z, matter_ds, dark_e_ds, hubble)

#----Non-flat universe-------

#Milne 
matter_mil = 0.0
dark_e_mil = 0.0
mu_theory_mil = mu_theory(z, matter_mil, dark_e_mil, hubble)

#PLOTTING THE MODELS' PREDICTIONS

 #Normal scale
plt.errorbar(z, mu, yerr=mu_err, fmt = 'o', markersize = 3, ecolor ='gray', label = "Supernova Ia")
plt.title("Hubble Diagram (Pantheon+SH0ES data and theoretical models)")
plt.xlabel("Redshift (z)")
plt.ylabel("Distance modulus ($\mu$)")
plt.plot(z, mu_theory_lcdm, label = "$\Lambda$CDM model")
plt.plot(z, mu_theory_eds, label = "Einstein-de Sitter model")
plt.plot(z, mu_theory_ds, label = "de Sitter model")
plt.plot(z, mu_theory_mil, label = "Milne model")
plt.legend()
file3 = input("Please enter the name of the file where the model comparison plot will be saved: ")
plt.savefig(file3)
plt.show()

 #Logarithmic scale
plt.errorbar(z, mu, yerr=mu_err, fmt = 'o', markersize = 3, ecolor ='gray', label = "Supernova Ia")
plt.title("Hubble Diagram (Pantheon+SH0ES data)")
plt.xscale('log')
plt.xlabel("Redshift (z) (log scale)")
plt.ylabel("Distance modulus ($\mu$)")
plt.plot(z, mu_theory_lcdm, label = "$\Lambda$CDM model")
plt.plot(z, mu_theory_eds, label = "Einstein-de Sitter model")
plt.plot(z, mu_theory_ds, label = "de Sitter model")
plt.plot(z, mu_theory_mil, label = "Milne model")
plt.legend()
file4 = input("Please enter the name of the file where the model comparison plot logarithmic will be saved:\n")
plt.savefig(file4)
plt.show()

#CHI SQUARED OF BEST FIT

print("To assess which model fits the data best, the chi2 of best fit will now be performed.\n")

chi2_lcdm = np.sum((mu-mu_theory_lcdm)**2/mu_err**2)
print(f"Chi2 Statistic for ΛCDM: {chi2_lcdm}")

chi2_mil = np.sum((mu-mu_theory_mil)**2/mu_err**2)
print(f"Chi2 Statistic for Milne: {chi2_mil}")

chi2_eds = np.sum((mu-mu_theory_eds)**2/mu_err**2)
print(f"Chi2 Statistic for Einstein-de Sitter: {chi2_eds}")

chi2_ds = np.sum((mu-mu_theory_ds)**2/mu_err**2)
print(f"Chi2 Statistic for de Sitter: {chi2_ds}\n")

print("Based on the chi2 test, the ΛCDM model best fits the data.\n") #We can see this from the smalles chi2

#===========================================================================
# FINDING THE BEST FIT PARAMETERS (LAMBDA CDM MODEL)
#===========================================================================

#VARIABLES

# Initial guess: 
p0 = (0.3, 70)
#Adding boundaries to avoid guesses out of the domain: 
lower_b = [0.0, 50]
upper_b = [2.0, 100]

# FUNCTIONS ADAPTED FOR BEST FIT

# We use the same as the previous part, but we replace dark_e by 1-matter to reduce the number of variables
def luminosity_distance_bf(z, matter, hubble):
    integral = quad(lambda z_prime: 1/(np.sqrt(matter*(1+z_prime)**3+(1-matter))), 0, z)[0]
    return ((c*(1+z))/hubble)*integral

def mu_theory_bf (z, matter, hubble): 
    dL_predictions = [luminosity_distance_bf (z, matter, hubble) for z in zHD]
    dL = np.array(dL_predictions)
    return 5*np.log10(dL)+25

# FINDING THE BEST FIT PARAMETERS

fit_params, error = curve_fit(mu_theory_bf, z, mu, sigma = mu_err, p0=p0, bounds=(lower_b, upper_b))
error = perr = np.sqrt(np.diag(error))
matter_fit = fit_params [0]
hubble_fit = fit_params [1]
dark_e_fit = 1 - matter_fit 
err_matter = error [0]
err_hubble = error [1]

print("The values we obtain from the full Pantheon+SH0ES dataset are the following:")
print(" Hubble constant: %.4f \u00B1 %.4f km/s/Mpc" %(hubble_fit,  err_hubble))
print(" Matter density parameter: %.4f \u00B1 %.4f" %(matter_fit,  err_matter))
print(" Cosmological constant density parameter: %.4f \u00B1 %.4f\n" %(dark_e_fit,  err_matter))

# PLOTING THE BEST FIT CURVE AGAINST THE DATA  

file5 = input("Enter the name of the file where the best fit plot will be saved: ")
y_data = mu_theory_bf (z, matter_fit, hubble_fit)
plt.plot(z, y_data, label = "Best Fit Curve for $\Lambda$CDM model " )
plt.errorbar(z, mu, yerr=mu_err, fmt = 'o', ecolor ='gray', markersize=3, label = "Supernovae Ia")
box = dict(boxstyle = 'round,pad=0.5', facecolor = 'white', edgecolor = 'gray')
plt.text(1.27, 30.95, "$H_0 = 73.02 \pm 0.18 km/s/Mpc$ \n$\Omega_m = 0.35 \pm 0.17$ \n$\Omega_\Lambda = 0.65 \pm 0.01$", bbox = box)
plt.title("Hubble Diagram (Pantheon+SH0ES data)")
plt.xlabel("Redshift (z)")
plt.ylabel("Distance modulus ($\mu$)")
plt.legend()
plt.savefig(file5)
plt.show()




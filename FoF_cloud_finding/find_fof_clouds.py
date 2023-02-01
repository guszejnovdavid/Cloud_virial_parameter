#!/usr/bin/env python
import numpy as np
import h5py
import load_from_snapshot #loading GIZMO snapshots
import os
import pickle
from sklearn.cluster import AgglomerativeClustering
import bz2
import shutil

#Constants and units conversion from standard GIZMO units
#pc=3.08567758E16 #pc to m, if needed
#yr=31556926.0 #yr to s if needed
kb=1.38*1E-23 #Boltzmann constant
mp=1.67*1E-27 #proton mass in kg
#mH2=2.1*1.67*1E-27 #assumin it is mostly H2
#H_massfrac_correction=1.2195 # correction from using default H mass fraction in sim
#msolar=2E30 #solar mass in kg
#mu0=np.pi*4E-7 #vacuum permeability, SI
#eps0=8.854E-12 #vacuum permittivity, SI
#G=6.67384*1E-11 #Gravitational constant in SI
#gauss=1E-4 #Guass to Tesla
#self.NDegreesofFreedom=5 #assuming it acts like diatomic gas
gamma=5.0/3.0 #adibatic index
MassConv=(1E10)/0.7 #converting to M_solar from 10^10 h^-1 M_sun
VelocityConv=1.0 #Converting to m/s from km/s units
#self.TimeConv=1.0/0.978; #Converting to Gyr from 0.978 Gyr units
LengthConv=1000.0/0.7   #Converting to pc from 1 h^-1 kpc
InternalEnergyConv=1E6 #converting from (km/s)^2 to kg (m/s)^2 by multiplying with mass
NumberDensConv=20.3813*2 #Converts Msolar/pc^3 to cm^(-3) for H
SolarMetallicity=0.02 #solar metallicity in code units (by mass)
ptype=0

snapdir='D:\Work\Projects\GalaxySim\highmass_galaxy\m12i_MHD'
snapnum=600
linking_length=20
nmin = 10 #cm-3
Tmax = 1e4 #K










#Converting internal energy from GIZMO into temperature
def energy_to_temp(internalenergy,metallicity, electronabundance,density ):
    global const
    #Get H2 dissociation temperature
    Tmol=100+np.zeros(len(internalenergy)) #temperature above which H2 breaks up
    Tmol[density>0] *= (density[density>0]*NumberDensConv) / 100.0 #based on Glover Clark 2012, lifted from cooling.c
    Tmol[Tmol>8000.0] = 8000.0; #max temperature for Tmol
    #Get mass fractions
    if (len(metallicity.shape)==2 ): 
        helium_mass_fraction=metallicity[:,1]
        metal_mass_fraction=0*metallicity[:,0]
    else:
        helium_mass_fraction=0.24+np.zeros(len(internalenergy)) #default mass fraction of He in GIZMO
        metal_mass_fraction=np.zeros(len(internalenergy))
    fmol=np.zeros(len(internalenergy)) #assume no H2 to start
    Htotfrac=1.0-helium_mass_fraction-metal_mass_fraction #all H and H2, ignore metals, small correction
    for iter in range(10): #10 iterations to converge
        #get mean molecular weight
        mu=1.0/( Htotfrac*(1-0.5*fmol) + helium_mass_fraction/4.0 + Htotfrac*electronabundance + metal_mass_fraction/(16.0+12.0*fmol) ) #from GIZMO get_mu routine in cooling.c
        #get temperature
        temp=internalenergy*(gamma-1.0)*(mp/kb)*mu
        #recalculate molecular fraction
        fmol=1.0/(1.0+ (temp/Tmol)**2 )
    return temp

def pickle_load(fname, use_bz2=False):
    if use_bz2:
        with bz2.BZ2File(fname, 'rb') as pickle_in:
            var = pickle.load(pickle_in, encoding='bytes')
    else:
        with open(fname,'rb') as pickle_in:
            var = pickle.load(pickle_in)
    return var

def pickle_dump(fname, var, use_bz2=False, make_backup=True):
    if make_backup:
        if os.path.exists(fname):
            shutil.copy(fname, fname+'.bak')
    if use_bz2:
        with bz2.BZ2File(fname, 'wb') as pickle_out:
            pickle.dump(var, pickle_out, protocol=2)    
    else:
        with open(fname,'wb') as pickle_out:
            pickle.dump(var, pickle_out)
            


#ID
allids = load_from_snapshot.load_from_snapshot("ParticleIDs",ptype,snapdir,snapnum)
#Position
x = load_from_snapshot.load_from_snapshot("Coordinates",ptype,snapdir,snapnum) * LengthConv # in pc
x = x - np.mean(x,axis=0)
#Density
density = load_from_snapshot.load_from_snapshot("Density",ptype,snapdir,snapnum) * MassConv/(LengthConv**3)*NumberDensConv # in cm^-3
#Temperature
electron_abundance = load_from_snapshot.load_from_snapshot("ElectronAbundance",ptype,snapdir,snapnum) #Electron Abundance
Z_all = load_from_snapshot.load_from_snapshot("Metallicity",ptype,snapdir,snapnum)# #Metallicity, take all species 
internalenergy = load_from_snapshot.load_from_snapshot("InternalEnergy",ptype,snapdir,snapnum) * InternalEnergyConv # internal energy in (m/s)^2
temp=energy_to_temp(internalenergy,(Z_all[:,0]/SolarMetallicity),electron_abundance,density) #K
#Select n>10 cm^-3 and T<10^4K
select = (density>nmin) & (temp<Tmax)
#Use sklearn
print("Data loaded")
labels = -np.ones_like(internalenergy)
cl=AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=linking_length).fit(x[select])
labels[select] = cl.labels_
print("Clustering done")


# #FoF clustering
# groups_temp = pyfof.friends_of_friends(x[select,:], linking_length)
# #Get IDs
# groups = []
# for g in groups_temp:
#     groups.append( allids[select][g] )

pickle_dump("m12i_MHD_snap600_FoF_labels.pickle", labels, use_bz2=True, make_backup=True)
            
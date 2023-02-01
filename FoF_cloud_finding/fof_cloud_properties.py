#!/usr/bin/env python
import numpy as np
#import h5py
import load_from_snapshot #loading GIZMO snapshots
import os
import pickle
import bz2
#from matplotlib import pyplot as plt
import shutil
import pytreegrav
import pandas as pd
from tqdm import tqdm



#Constants and units conversion from standard GIZMO units
pc=3.08567758E16 #pc to m, if needed
#yr=31556926.0 #yr to s if needed
kb=1.38*1E-23 #Boltzmann constant
mp=1.67*1E-27 #proton mass in kg
#mH2=2.1*1.67*1E-27 #assumin it is mostly H2
#H_massfrac_correction=1.2195 # correction from using default H mass fraction in sim
msolar=msun=2E30 #solar mass in kg
mu0=np.pi*4E-7 #vacuum permeability, SI
#eps0=8.854E-12 #vacuum permittivity, SI
G=6.67384*1E-11 #Gravitational constant in SI
Gcode=G/(pc/msun) #Gravitational constant in code units
gauss=1E-4 #Guass to Tesla
#self.NDegreesofFreedom=5 #assuming it acts like diatomic gas
gamma=5.0/3.0 #adibatic index
MassConv=(1E10)/0.7 #converting to M_solar from 10^10 h^-1 M_sun
VelocityConv=1000.0 #Converting to m/s from km/s units
#self.TimeConv=1.0/0.978; #Converting to Gyr from 0.978 Gyr units
LengthConv=1000.0/0.7   #Converting to pc from 1 h^-1 kpc
InternalEnergyConv=1E6 #converting from (km/s)^2 to kg (m/s)^2 by multiplying with mass
NumberDensConv=20.3813*2 #Converts Msolar/pc^3 to cm^(-3) for H
SolarMetallicity=0.02 #solar metallicity in code units (by mass)
ptype=0

snapdir='D:\Work\Projects\GalaxySim\highmass_galaxy\m12i_MHD'
snapnum=600
N_min = 20


def start_or_append_in_dict(dictionary, label, value):
    if label in dictionary:
        if not  hasattr(dictionary[label], "__iter__"):
            dictionary[label] = [dictionary[label]] #assume that it already has the first value
        dictionary[label].append(value)
    else:
        dictionary[label] = [value]


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

def weight_percentile(data,weights,p,verbose=False):
    'Weighted percentile calculation'
    weightsum=np.sum(weights)
    if weightsum:
        if (np.size(data)==1):
            if hasattr(data, "__iter__"):
                return data[0]
            else:
                return data
        else:
            #reorder data
            sortind=np.argsort(data)
            cdf=np.cumsum(weights[sortind])/np.sum(weights)
            if(p>1.0):
                p /= 100.0
                if verbose: print("Large percentile value given, converting %g to %g"%(p*100,p))
            percentile_index=np.argmax(cdf>=p)
            return data[sortind[percentile_index]]
    else:
        return 0

def weight_avg(data,weights=None):
    'Weighted average'
    if weights is None:
        return np.mean(data)
    else:
        weightsum=np.sum(weights)
        if weightsum:
            return np.sum(data*weights)/weightsum
        else:
            return 0*data

def galactic_distance(x, galaxy_center=np.array([0,0,0])):
    return np.linalg.norm(np.mean(x,axis=0) - galaxy_center)

def Sphere_Potential_Energy(M,R,G=Gcode):
    return (-0.5*G*M*M/R)*msolar

def Potential_Energy(x,m,h,G=Gcode, include_self=True):
    E_pot = 0.5*np.sum(m*pytreegrav.Potential(x,m,h,G=Gcode))*(msolar)
    if include_self: 
        E_pot += np.sum(Sphere_Potential_Energy(m,h)) #gravitational energy in each cell
    return E_pot

def Kinetic_Energy(m, v):
    dv = v - np.average(v, weights=m,axis=0)
    vSqr = np.sum(dv**2,axis=1)
    return 0.5*np.sum(m*vSqr)*msolar  

def Internal_Energy(m,u):
    return np.sum(m*u)*msolar

def Magnetic_Energy(m,Bv,rho):
    B=np.sqrt(Bv[:,0]**2+Bv[:,1]**2+Bv[:,2]**2)
    return 1/(2*mu0)*np.sum(B*B*(m/rho)*(pc**3))

def Potential_Energy_obs(x,m):
    R=R_eff(x,only_2D=True)
    M=np.sum(m)
    return Sphere_Potential_Energy(M,R,G=Gcode)

def Kinetic_Energy_obs(m,v):
    return 0.5*np.sum(m*msolar)*3*(sigma_1D(m,v,z_only=True)**2)

def Internal_Energy_obs(m,temp):
    Tmean = weight_avg(temp, m )
    return Tmean/((gamma-1.0)*(mp/kb)*2)*np.sum(m)*msolar

def Magnetic_Energy_obs(x,m,Bv,rho):
    R=R_eff(x,only_2D=True)
    Bmean = B_mean(Bv,m,rho,only_2D=True)
    return 1/(2*mu0)*(Bmean**2)*(4*np.pi/3)*(R**3)*(pc**3)


def alpha(x,m,v,rho,u,Bv,h,temp, only_2D=False):
    if only_2D:
         return np.abs( 2*(Kinetic_Energy_obs(m, v) + Internal_Energy_obs(m,temp) + 0.5*Magnetic_Energy_obs(x,m,Bv,rho)) /  Potential_Energy_obs(x,m) )
    else:
         return np.abs( 2*(Kinetic_Energy(m, v) + Internal_Energy(m,u) + 0.5*Magnetic_Energy(m,Bv,rho)) /  Potential_Energy(x,m,h) )  

def alpha_thermal(x,m,u,h,temp, only_2D=False):
    if only_2D:
        return np.abs( 2*Internal_Energy_obs(m,temp) /  Potential_Energy_obs(x,m) )
    else:
        return np.abs( 2*Internal_Energy(m,u) /  Potential_Energy(x,m,h) )

def alpha_turb(x,m,v,h, only_2D=False):
    if only_2D:
        return np.abs( 2*Kinetic_Energy_obs(m, v  ) /  Potential_Energy_obs(x,m) )
    else:
        return np.abs( 2*Kinetic_Energy(m, v  ) /  Potential_Energy(x,m,h) )

def alpha_magnetic(x,m,rho,Bv,h, only_2D=False):
    if only_2D:
        return np.abs( Magnetic_Energy_obs(x,m,Bv,rho) /  Potential_Energy_obs(x,m) )
    else:
        return np.abs( Magnetic_Energy(m,Bv,rho) /  Potential_Energy(x,m,h) )

def aspect_ratio(x,m):
    dx = x - np.mean(x,axis=0)
    R = np.linalg.norm(dx,axis=1)
    ndim = x.shape[1]
    I = np.eye(ndim)*np.mean(R**2)
    for i in range(ndim):
        for j in range(ndim):
            I[i,j] -= np.mean(dx[:,i]*dx[:,j])
    w, v = np.linalg.eig(I)
    R_p = np.sqrt(w)
    return np.max(R_p)/np.min(R_p), np.sum(R_p/R_p.max()), R_p       

def ellipsoid_surface(a,b,c):
    return 4*np.pi*( ( ( (a*b)**1.6 + (a*c)**1.6 + (b*c)**1.6 )/3  )**(1/1.6) )

def sphere_surface(R):
    return 4*np.pi*(R**2)
  
def sphericity(x,m):
    _,_,Rp = aspect_ratio(x,m)
    a=Rp[0]; b=Rp[1];
    if len(Rp)==3:
        c=Rp[2]
    else:
        c=np.sqrt(a*b)
    R_sph = (a*b*c)**(1/3) #volume equivalent sphere
    return sphere_surface(R_sph)/ellipsoid_surface(a,b,c)
             
def sigma_1D(m,v, z_only=False):   
    if z_only:
        dv_sq = ((v - np.mean(v,axis=0))**2)[:,2]
        return np.sqrt(np.sum(dv_sq*m)/np.sum(m))
    else:     
        dv_sq = np.sum((v - np.mean(v,axis=0))**2,axis=1) 
        return np.sqrt(np.sum(dv_sq*m)/np.sum(m)/3)

def cs(temp,mu):
    return np.sqrt(gamma*(kb/mp)*temp/mu)

def Mach_eff(m,v,temp,mu):
    Mach_sq = ( np.linalg.norm((v - np.mean(v,axis=0)),axis=1) / cs(temp,mu) )**2            
    return np.sqrt(np.sum(Mach_sq*m)/np.sum(m)/3)

def frac_above_num_density(m,num_density,n_lim):
    return np.sum(m[num_density>n_lim])/np.sum(m)


def R_mass_percentile(x,m,p):
    dx = x - np.mean(x,axis=0)
    R = np.linalg.norm(dx,axis=1)
    return weight_percentile(R,m,p)
    
def R_eff(x,only_2D=False):  
    dx = x - np.mean(x,axis=0)
    if not only_2D: 
        return  np.sqrt(np.sum(np.var(dx, axis=0)))
    else:
        dx[:,2] *= 0
        return  np.sqrt(1.5*np.sum(np.var(dx, axis=0)))

def B_mean(Bv,m,rho,only_2D=False):
    if not only_2D:
        B=np.sqrt(Bv[:,0]**2+Bv[:,1]**2+Bv[:,2]**2)
        return weight_avg(B,(m/rho))
    else:
        return weight_avg(np.abs(Bv[:,2]),(m/rho))*np.sqrt(3)
        

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
    return temp, mu, fmol


def load_data(snapnum,snapdir,ptype):
    #Get the index array that we can use to pick out the particles that have the given ids
    allids = load_from_snapshot.load_from_snapshot("ParticleIDs",ptype,snapdir,snapnum)
    #Softening
    m = load_from_snapshot.load_from_snapshot("Masses",ptype,snapdir,snapnum) * MassConv # in M_sun
    #Mass
    h = load_from_snapshot.load_from_snapshot("SmoothingLength",ptype,snapdir,snapnum) * LengthConv # in pc
    #Position
    x = load_from_snapshot.load_from_snapshot("Coordinates",ptype,snapdir,snapnum) * LengthConv # in pc
    #Velocity
    v = load_from_snapshot.load_from_snapshot("Velocities",ptype,snapdir,snapnum) * VelocityConv # in m/s
    #Density
    density = load_from_snapshot.load_from_snapshot("Density",ptype,snapdir,snapnum) * MassConv/(LengthConv**3)*NumberDensConv # in cm^-3
    #Star formation rate
    sfr = load_from_snapshot.load_from_snapshot("StarFormationRate",ptype,snapdir,snapnum) #this should be M_sun/yr by default
    #Electron Abundance
    electron_abundance = load_from_snapshot.load_from_snapshot("ElectronAbundance",ptype,snapdir,snapnum)
    #Metallicity
    Z_all = load_from_snapshot.load_from_snapshot("Metallicity",ptype,snapdir,snapnum)# take all species 
    #Internal Energy and Temperature
    internalenergy = load_from_snapshot.load_from_snapshot("InternalEnergy",ptype,snapdir,snapnum) * InternalEnergyConv # in (m/s)^2
    temp, mu, fmol=energy_to_temp(internalenergy,(Z_all[:,0]/SolarMetallicity),electron_abundance,density) #K
    #Magnetic field
    try:
        Bv=load_from_snapshot.load_from_snapshot("MagneticField",ptype,snapdir,snapnum)*gauss #should be in gauss, we convert to Tesla
        #B=np.sqrt(Bv[:,0]**2+Bv[:,1]**2+Bv[:,2]**2)
    except:
        print('\t No magnetic field present setting it to 0')
        Bv = 0*v
        #B=np.zeros(len(density)) #set it to zero
    return allids,x,m,v,density,sfr,Z_all,temp,Bv,mu,fmol,h,internalenergy
            
def find_galaxy_center_and_orientation(snapnum,snapdir, ptype=4, Rmax_kpc = 100):
    #Mass
    m = load_from_snapshot.load_from_snapshot("Masses",ptype,snapdir,snapnum) * MassConv # in M_sun
    #Position
    x = load_from_snapshot.load_from_snapshot("Coordinates",ptype,snapdir,snapnum) * LengthConv # in pc
    #Velocity
    v = load_from_snapshot.load_from_snapshot("Velocities",ptype,snapdir,snapnum) * VelocityConv # in m/s
    #Median of total
    x_med = np.median(x,axis=0)
    R = np.linalg.norm(x-x_med,axis=1)
    ind = R<(Rmax_kpc*1000)
    #Return CoM after removing far away objects
    x_com = np.sum(x[ind,:]*m[ind,None],axis=0)/np.sum(m[ind])   
    v_com = np.sum(v[ind,:]*m[ind,None],axis=0)/np.sum(m[ind])
    #Get angular momentum
    dv = v[ind,:]-v_com[None,:]
    dx = x[ind,:]-x_com[None,:]
    L = -np.cross(dx,dv)
    L_tot = np.sum(L,axis=0); L_tot/=np.linalg.norm(L_tot)
    return x_com,v_com,L_tot
 
# def FO_transform_and_Center(x, v, B, FO_transform, Center,snapdir,snapnum):
#     aaaaaa  #Nem jo erre a snpashotra
    
#     import h5py
#     cenFile = snapdir+"\snap_"+str(snapnum)+"_cents.hdf5"
#     cFile = h5py.File(cenFile,'r')
#     gc = cFile.attrs['StellarCenter']*1000 # Stellar center found with FIREMapper (kpc, physical)
#     Vcm = cFile.attrs['StellarCMVel']
#     print('CoM: ',gc, ' v:', Vcm)
    
#     print('Transforming to Face-on coordinates...')
#     phi = cFile.attrs['Phi']; theta = cFile.attrs['Theta']
#     print('phi %g theta %g'%(phi, theta))
    
#     px = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
#     py = np.array([-np.sin(phi), np.cos(phi), 0])
#     pz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            
#     x_FO = np.zeros_like(x)
#     v_FO = np.zeros_like(v)
#     B_FO = np.zeros_like(B)

#     print('Centering..')
#     xg = x - gc[None,:]
#     vg = x - Vcm[None,:]
    
#     x_FO[:,0] = np.sum(xg*px[None,:],axis=1)
#     x_FO[:,1] = np.sum(xg*py[None,:],axis=1)
#     x_FO[:,2] = np.sum(xg*pz[None,:],axis=1)
#     v_FO[:,0] = np.sum(vg*px[None,:],axis=1)
#     v_FO[:,1] = np.sum(vg*py[None,:],axis=1)
#     v_FO[:,2] = np.sum(vg*pz[None,:],axis=1)
#     B_FO[:,0] = np.sum(B*px[None,:],axis=1)
#     B_FO[:,1] = np.sum(B*py[None,:],axis=1)
#     B_FO[:,2] = np.sum(B*pz[None,:],axis=1)   

#     return x_FO, v_FO, B_FO
   

#Init
cloud_properties = {}
#Load data
allids,x,m,v,density,sfr,Z_all,temp,Bv,mu,fmol,h,u = load_data(snapnum,snapdir,ptype) 
rho = density/NumberDensConv
#Face on transform and center
#x,v,Bv = FO_transform_and_Center(x, v, Bv, True, True, snapdir, snapnum)

galaxy_center,v_center,L_norm = find_galaxy_center_and_orientation(snapnum,snapdir)
#Transform
x = x - galaxy_center[None,:]
v = v - v_center[None,:]
e1 = np.cross([0,1,0],L_norm); e1/= np.linalg.norm(e1)
e2 = np.cross(L_norm,e1); e2/= np.linalg.norm(e2)
Tranform_mtx = np.vstack((e1,e2,L_norm)) #Transformation mtx
x = (Tranform_mtx @ x.T).T
v = (Tranform_mtx @ v.T).T
Bv = (Tranform_mtx @ Bv.T).T
#Load clustering
labels = np.int64(pickle_load('m12i_MHD_snap600_FoF_labels.pickle', use_bz2=True))
print('Loading finsihed')
#Find sizeable clouds
all_cloud_labels, cloud_pop = np.unique(labels[labels>-1], return_counts=True)
cloud_labels = all_cloud_labels[cloud_pop>=N_min]
#Calculate properties
for cloud_label in tqdm(cloud_labels):
    ind = (labels==cloud_label)
    cloud_mass = np.sum(m[ind])
    #Cloud mass
    start_or_append_in_dict(cloud_properties, 'Mass', cloud_mass )
    #Effective radius
    start_or_append_in_dict(cloud_properties, 'R_eff', R_eff(x[ind,:]) )
    start_or_append_in_dict(cloud_properties, 'R_eff_obs', R_eff(x[ind,:],only_2D=True) )
    #Half-mass radius
    start_or_append_in_dict(cloud_properties, 'R_50', R_mass_percentile(x[ind,:],m[ind],50) )
    start_or_append_in_dict(cloud_properties, 'R_50_obs', R_mass_percentile(x[ind,:2],m[ind],50)*np.sqrt(3/2) )
    #90%-mass radius
    start_or_append_in_dict(cloud_properties, 'R_90', R_mass_percentile(x[ind,:],m[ind],90) )
    start_or_append_in_dict(cloud_properties, 'R_90_obs', R_mass_percentile(x[ind,:2],m[ind],90)*np.sqrt(3/2) )
    #Aspect ratio
    start_or_append_in_dict(cloud_properties, 'aspect_ratio', aspect_ratio( x[ind,:],m[ind])[0] )
    start_or_append_in_dict(cloud_properties, 'aspect_ratio_obs', aspect_ratio( x[ind,:2],m[ind])[0] )
    #Effective Dimension
    start_or_append_in_dict(cloud_properties, 'effective dimension', aspect_ratio( x[ind,:],m[ind])[1] )
    start_or_append_in_dict(cloud_properties, 'effective dimension_obs', aspect_ratio( x[ind,:],m[ind])[1]*3/2 )
    #Sphericity
    start_or_append_in_dict(cloud_properties, 'sphericity', sphericity( x[ind,:],m[ind]) )
    start_or_append_in_dict(cloud_properties, 'sphericity_obs', sphericity( x[ind,:2],m[ind]) )
    #Fraction above 100 cm^-3
    start_or_append_in_dict(cloud_properties, 'f_n100', 1e-10+frac_above_num_density(m[ind],density[ind],100) )
    #Fraction above 1000 cm^-3
    start_or_append_in_dict(cloud_properties, 'f_n1000', 1e-10+frac_above_num_density(m[ind],density[ind],1000) )
    #Total star formation rate
    start_or_append_in_dict(cloud_properties, 'SFR', np.sum(sfr[ind])+1e-10 )
    #Mean metallicity
    start_or_append_in_dict(cloud_properties, 'Z_mean', weight_avg(Z_all[ind,0]/SolarMetallicity, m[ind]) )
    #Mean magnetic field
    start_or_append_in_dict(cloud_properties, 'B_mean', B_mean(Bv[ind,:],m[ind],rho[ind])  )
    #Mean magnetic field (obs)
    start_or_append_in_dict(cloud_properties, 'B_mean_obs', B_mean(Bv[ind,:],m[ind],rho[ind],only_2D=True)  )
    #Mean temperature
    start_or_append_in_dict(cloud_properties, 'T_mean', weight_avg(temp[ind], m[ind] ))
    #Half-mass temperature
    start_or_append_in_dict(cloud_properties, 'T_50', weight_percentile(temp[ind],m[ind],50) )
    #90%-mass temperature
    start_or_append_in_dict(cloud_properties, 'T_90', weight_percentile(temp[ind],m[ind],90) )
    #Velocity dispersion (1D)
    start_or_append_in_dict(cloud_properties, 'sigma_1D', sigma_1D(m[ind],v[ind,:]) )
    #Velocity dispersion (1D, obs)
    start_or_append_in_dict(cloud_properties, 'sigma_1D_obs', sigma_1D(m[ind],v[ind,:],z_only=True) )
    #Effective Mach number (1D)
    start_or_append_in_dict(cloud_properties, 'Mach', Mach_eff(m[ind], v[ind,:], temp[ind], mu[ind]) )
    start_or_append_in_dict(cloud_properties, 'Mach_obs', sigma_1D(m[ind],v[ind,:],z_only=True)/cs(weight_avg(temp[ind], m[ind] ),2) )
    #Molecular fraction
    start_or_append_in_dict(cloud_properties, 'f_atomic', 1-np.sum(fmol[ind] * m[ind])/cloud_mass )
    #Galactocentric distance
    start_or_append_in_dict(cloud_properties, 'R_galactic', galactic_distance(x[ind,:]) )
    for only_2D,obs_text in zip([False,True],['','_obs']):
        #Full virial parameter
        start_or_append_in_dict(cloud_properties, 'alpha'+obs_text, alpha(x[ind,:], m[ind], v[ind,:], rho[ind], u[ind], Bv[ind,:], h[ind], temp[ind], only_2D=only_2D) )
        #Thermal virial parameter
        start_or_append_in_dict(cloud_properties, 'alpha_thermal'+obs_text, alpha_thermal(x[ind,:], m[ind], u[ind], h[ind],  temp[ind], only_2D=only_2D))
        #Turbulent virial parameter
        start_or_append_in_dict(cloud_properties, 'alpha_turb'+obs_text, alpha_turb(x[ind,:], m[ind], v[ind,:], h[ind], only_2D=only_2D) )
        #Magnetic virial parameter
        start_or_append_in_dict(cloud_properties, 'alpha_magnetic'+obs_text, alpha_magnetic(x[ind,:], m[ind], rho[ind], Bv[ind,:], h[ind], only_2D=only_2D)  )
    


#Create dataframe
df = pd.DataFrame.from_dict(cloud_properties)
#Save dataframe
df_fname = 'FIRE2_m12i_MHD_FoF_clouds'
df.to_pickle(df_fname + '.p')
df.to_csv(df_fname + '.csv')
print('Dataframe saved to %s and %s'%(df_fname + '.p', df_fname + '.csv', ) )



    



#!/usr/bin/env python

#-----------------------------------------------------------------------
# PROGRAM: optimal_average.py
#-----------------------------------------------------------------------
# Version 0.1
# 21 October, 2021
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#-----------------------------------------------------------------------

# Numerics and dataframe libraries:
import numpy as np
import numpy.ma as ma
import scipy
import scipy.stats as stats    
import pandas as pd
import xarray as xr
import pickle
from netCDF4 import Dataset

# Datetime libraries:
from datetime import datetime
import nc_time_axis
import cftime
from cftime import num2date, DatetimeNoLeap

# Statistics libraries:
import scipy.linalg as la
from scipy.special import erfinv
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Plotting libraries:
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = False
import seaborn as sns; sns.set()

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#-----------------------------------------------------------------------------
# PACKAGE VERSIONS
#-----------------------------------------------------------------------------

import platform
print("python       : ", platform.python_version())
print("pandas       : ", pd.__version__)
print("matplotlib   : ", matplotlib.__version__)
#print("squarify     :  0.4.3")

#---------------------------------------------------------------------------
import ensemble_func as pca_tools # load PCA and MC sampling of Cov tools
#---------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16
country = 'australia'
#country = 'india'
use_mse = False

#------------------------------------------------------------------------------
# LOAD: pkl
#------------------------------------------------------------------------------

df_temp = pd.read_pickle('DATA/df_temp_expect.pkl', compression='bz2') # dataframe of GloSAT absolute temperatures in degrees C
df_norm = pd.read_pickle('DATA/df_normals.pkl', compression='bz2')
for i in range(1,13): df_norm[str(i)] = df_norm[str(i)].astype(float)
df_norm [ df_norm == -999 ] = np.nan

#Index(['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#       'stationcode', 'stationlat', 'stationlon', 'stationelevation',
#       'stationname', 'stationcountry', 'stationfirstyear', 'stationlastyear',
#       'stationsource', 'stationfirstreliable', 'n1', 'n2', 'n3', 'n4', 'n5',
#       'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'e1', 'e2', 'e3', 'e4',
#       'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 's1', 's2', 's3',
#       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12'],

# TRIM: to 1780 ( start of GloSAT )

df_temp = df_temp[ df_temp.year >= 1780 ]
df_temp = df_temp[ df_temp.stationcountry == country.upper()]

# EXTRACT: station metadata

station_codes = df_temp['stationcode'].unique()
station_indices = np.arange( len(station_codes) )
station_latitudes = df_temp.groupby('stationcode')['stationlat'].mean()
station_longitudes = df_temp.groupby('stationcode')['stationlon'].mean()
station_elevations = df_temp.groupby('stationcode')['stationelevation'].mean()
station_names = df_temp.groupby('stationcode')['stationname'].unique()
station_names = np.array( [ station_names[i][0] for i in range(len(station_names)) ], dtype='object')
station_countries = df_temp.groupby('stationcode')['stationcountry'].unique()
station_countries = np.array([ station_countries[i][0] for i in range(len(station_countries)) ], dtype='object')
station_firstyears = df_temp.groupby('stationcode')['stationfirstyear'].mean().astype(int)
station_lastyears = df_temp.groupby('stationcode')['stationlastyear'].mean().astype(int)
station_sources = df_temp.groupby('stationcode')['stationsource'].mean().astype(int)
station_firstreliables = df_temp.groupby('stationcode')['stationfirstreliable'].mean().astype(int)

#------------------------------------------------------------------------------
# CONSTRUCT: station and local expectation timeseries 
#------------------------------------------------------------------------------

# INITIALISE: arrays and define time axis

time = pd.date_range(start='1780', end='2021', freq='MS')[0:-1]
tas = np.zeros( [len(station_codes), len(time)] )
tas_n = np.zeros( [len(station_codes), len(time)] )
lek_n = np.zeros( [len(station_codes), len(time)] )
lek_e = np.zeros( [len(station_codes), len(time)] )
lek_s = np.zeros( [len(station_codes), len(time)] )

for i in range( len(station_codes) ):

    station_code = station_codes[i]
    df = df_temp.copy(); df = df.iloc[0:0] # copy dtypes and reset dataframe    
    df['year'] = np.arange(1780,2021)
    dg = df_temp[ df_temp.stationcode==station_code].sort_values(by='year').reset_index(drop=True).dropna() # extract station dataframe    
#   df = df.merge(dg, how='left', on='year')    
    dn = np.array([ df_norm[df_norm.stationcode==station_code][str(k)].values for k in range(1,13) ]).ravel().astype(float) # extract station normals     
   
    for j in range( len(dg) ):
        
        df.iloc[ df.year[ df.year == dg.year.values[j] ].index,:] = dg.iloc[j,:]

    o = np.array(df.iloc[:,1:13]).ravel()
    n = np.array(df.iloc[:,23:35]).ravel()
    e = np.array(df.iloc[:,35:47]).ravel()
    s = np.array(df.iloc[:,47:59]).ravel()      
    mask = ~np.isfinite(o.astype(float)) | ~np.isfinite(e.astype(float))
    o[mask] = np.nan
    n[mask] = np.nan
    e[mask] = np.nan
    s[mask] = np.nan
    lek_n[i,:] = n
    lek_e[i,:] = e
    lek_s[i,:] = s
    tas[i,:] = o
    tas_n[i,:] = np.tile(dn,int(len(time)/12))
    
tas_a = tas - tas_n # observations
    
#------------------------------------------------------------------------------
# PRUNING ALGORITHM
#------------------------------------------------------------------------------

# COMPUTE: TAS and LEK means and LEK standard deviation

tas_mean = np.nanmean(tas_a, axis=0)
lek_mean = np.nanmean(lek_e, axis=0)
lek_std = np.nanstd(lek_s, axis=0)

# SMOOTH: with MA filter

nsmooth = 24

tas_a_smooth = np.zeros( [len(station_codes), len(time)] )
for i in range( len(station_codes) ): tas_a_smooth[i,:] = pd.Series( tas_a[i,:] ).rolling(nsmooth, center=True).mean()
tas_a_smooth_mean = np.nanmean(tas_a_smooth, axis=0)
tas_mean_smooth = pd.Series( tas_mean ).rolling(nsmooth, center=True).mean().values

# COMPUTE: cost function ( MSE per station holdout )

cost_vec = []
r_vec = []
n_vec = []
for i in range(len(tas_a)):
    
#    tas_holdout = tas_a*1.0
    tas_holdout = tas_a_smooth*1.0
    tas_holdout[i,:] = tas_holdout[i,:]*np.nan
    tas_holdout_mean = np.nanmean(tas_holdout, axis=0)
    mask_i = np.isfinite(tas_mean_smooth) & np.isfinite(tas_a_smooth[i,:])
    if mask_i.sum() == 0:
        cost = np.nan
        r = np.nan
        n =  0
    else:
        cost = np.nanmean( (tas_mean_smooth[mask_i]-tas_a_smooth[i,:][mask_i])**2.0 ) 
        r = scipy.stats.pearsonr( tas_mean_smooth[mask_i], tas_a_smooth[i,:][mask_i] )[0]                     
        n = mask_i.sum()
    cost_vec.append(cost)
    r_vec.append(r)
    n_vec.append(n)

cost_vec = np.array( cost_vec )
r_vec = np.array( r_vec )
n_vec = np.array( n_vec )

# FILTER: retain series n > 30-yr and r > 0.9

mask = (r_vec >= 0.9) & (n_vec >= int(len(time)*0.25))
cost_vec = cost_vec[ mask ]
r_vec = r_vec[ mask ]
n_vec = n_vec[ mask ]

# EXTRACT: best fit station to mean of stations

if use_mse == True:

    cost_n_array = np.vstack([cost_vec,n_vec]).T
    idx = np.lexsort((cost_n_array[:,0],cost_n_array[:,1])) # sort on column 1 then on column 2
    cost_n_array_sorted = cost_n_array[idx]
#    opt = np.where(cost_n_array[:,0] == np.nanmin(np.array(cost_n_array[:,0])))[0][0] # optimum = min MSE
    opt = np.where(cost_n_array_sorted[:,0] == np.nanmin(np.array(cost_n_array_sorted[:,0])))[0][0] # optimum = min MSE
    
else:
    
    cost_n_array = np.vstack([r_vec,n_vec]).T
    idx = np.lexsort((cost_n_array[:,0],cost_n_array[:,1])) # sort on column 1 then on column 2
    cost_n_array_sorted = cost_n_array[idx]
#    opt = np.where(cost_n_array[:,0] ==   np.nanmax(np.array(cost_n_array[:,0])))[0][0] # optimum = max r
    opt = np.where(cost_n_array_sorted[:,0] == np.nanmax(np.array(cost_n_array_sorted[:,0])))[0][0] # optimum = max_r

#mask = np.isfinite(tas_mean) & np.isfinite(tas_opt)
idx_opt = idx[opt]    
tas_opt = tas_a_smooth[idx_opt,:]

idx_opt_mse = np.argsort(cost_vec)
idx_opt_r = np.argsort(r_vec)
tas_opt_mse = tas_a[idx_opt_mse[0],:]
tas_opt_r = tas_a[idx_opt_r[-1],:]

#------------------------------------------------------------------------------
# PLOT: decadal mean GloSAT.p03 v LEK
#------------------------------------------------------------------------------
        
tas_yearly = pd.Series(tas_mean).rolling(12,center=True).mean().values # yearly
lek_yearly = pd.Series(lek_mean).rolling(12,center=True).mean().values # yearly
tas_smooth = pd.Series(tas_mean).rolling(nsmooth,center=True).mean().values # decadal
lek_smooth = pd.Series(lek_mean).rolling(nsmooth,center=True).mean().values # decadal
tas_std_smooth = pd.Series(tas_mean).rolling(nsmooth,center=True).std().values # decadal
lek_std_smooth = pd.Series(lek_std).rolling(nsmooth,center=True).mean().values # decadal

figstr = 'glosat-v-lek-lsat' + '-' + country + '.png'
titlestr = 'LSAT at 2m: ' + country.title() + ', n(stations)=' + str(len(station_codes))
            
fig, ax = plt.subplots(figsize=(15,10))
plt.plot( time, tas_smooth, ls='-', lw=3, color='red', label='GloSAT.p03: mean')
plt.plot( time, lek_smooth, ls='-', lw=3, color='blue', label='LEK: mean')
#plt.axhline(y=0, ls='-', lw=1, color='black', alpha=0.5, label='pre-HadCRUT')
#plt.axvline(x=pd.to_datetime('01-01-1850', format='%d-%m-%Y'), ls='dotted', lw=2, color='black', label='Start of HadCRUT5.0.1')
#ylimits = ax.get_ylim()
#xlimits = ax.get_xlim()
#plt.fill_betweenx(ylimits, xlimits[0], pd.to_datetime('01-01-1850', format='%d-%m-%Y'), color='black', alpha=0.1)
plt.fill_between(time, tas_smooth-tas_std_smooth, tas_smooth+tas_std_smooth, color='red', alpha=0.05, label='GloSAT.p03: s.d.')
plt.fill_between(time, lek_smooth-lek_std_smooth, lek_smooth+lek_std_smooth, color='blue', alpha=0.2, label='LEK: s.d.')
plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)        
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'Anomaly ( from 1961-1990 ), $^{\circ}$C', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)   
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')        

figstr = 'glosat-v-lek-lsat' + '-' + country + '-' + 'cost_analysis' + '.png'
titlestr = 'Holdout cost function: ' + country.title() + ', n(stations)=' + str(len(station_codes))

fig, ax = plt.subplots(figsize=(15,10))
plt.bar( np.arange(len(cost_vec)), cost_vec, color='blue', label='Cost function (MSE)')
plt.bar( idx_opt_mse[0], cost_vec[idx_opt_mse[0]], color='black', label='Optimum')
#plt.plot( cost_vec, ls='-', lw=3, color='blue', label='Cost function (MSE)')
#plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)        
plt.xlabel('Station index', fontsize=fontsize)
plt.ylabel(r'Mean squared error (MSE), $^{\circ}$C', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)   
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')           

figstr = 'glosat-v-lek-lsat' + '-' + country + '-' + 'correlation' + '.png'
titlestr = 'Holdout correlation: ' + country.title() + ', n(stations)=' + str(len(station_codes))

fig, ax = plt.subplots(figsize=(15,10))
plt.bar( np.arange(len(r_vec)), r_vec, color='blue', label='Pearson correlation')
plt.bar( idx_opt_r[-1], r_vec[idx_opt_r[-1]], color='black', label='Optimum')
#plt.bar( r_vec, ls='-', lw=3, color='blue', label='Pearson correlation')
#plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)      
plt.ylim(-1,1)  
plt.xlabel('Station index', fontsize=fontsize)
plt.ylabel('Correlation', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)   
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')           
            
figstr = 'glosat-v-lek-lsat' + '-' + country + '-' + 'cost_analysis' + '-' + 'bestfit' + '.png'
titlestr = ' LSAT at 2m vs most representative station=' +  station_codes[idx_opt_r[-1]] + ': ' + country.title() + ', n(stations)=' + str(len(station_codes))
                
fig, ax = plt.subplots(figsize=(15,10))
for i in range(len(tas_a)):
#    plt.plot( time, pd.Series(tas_a[i,:]).rolling(nsmooth,center=True).mean(), ls='-', lw=1, alpha=0.2)   
    plt.plot( time, pd.Series(tas_a_smooth[i,:]).rolling(nsmooth,center=True).mean(), ls='-', lw=1, alpha=0.2)   
plt.plot( time, pd.Series(tas_mean).rolling(nsmooth,center=True).mean(), ls='-', lw=2, color='red', label='GloSAT.p03: mean')
plt.plot( time, pd.Series(lek_mean).rolling(nsmooth,center=True).mean(), ls='-', lw=2, color='blue', label='LEK: mean')
plt.fill_between(time, lek_smooth-lek_std_smooth, lek_smooth+lek_std_smooth, color='blue', alpha=0.2, label='LEK: s.d.')
plt.plot( time, pd.Series(tas_opt_r).rolling(nsmooth,center=True).mean(), ls='-', lw=2, color='black', label='Optimal=' + station_codes[idx_opt_r[-1]] + ': MSE='+str(np.round( cost_vec[idx_opt_r[-1]],3 )) + r', $\rho$=' + str(np.round( r_vec[idx_opt_r[-1]],3 )) )
plt.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)        
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'Anomaly ( from 1961-1990 ), $^{\circ}$C', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)   
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')        
                
#------------------------------------------------------------------------------
print('** END')


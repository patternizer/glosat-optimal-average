import numpy as np
from numpy import array_equal, savetxt, loadtxt, frombuffer, save as np_save, load as np_load, savez_compressed, array
import scipy.linalg as la
from scipy.special import erfinv
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from netCDF4 import Dataset
import xarray

###########################################
# Version 0.1                             # 
# 25 July, 2019                           #
# https://patternizer.github.io/          #
# michael.taylor AT reading DOT ac DOT uk #
###########################################

#---------------------------------------------------------------------------
# FUNCTION LIST (alphabetical):
#---------------------------------------------------------------------------
# cov2draws(ds,npop): [npop] MC-draws using multi-normal sampling (MNS)
# cov2ev(X,c): Eigenvalue decomposition (EVD) of the covariance matrix --> nPC(c)
# cov2svd(X,c): Singular value decomposition (SVD) of the covariance matrix --> nPC(c)
# cov2u(Xcov,N): MC-estimate of uncertainty from covariance matrix
# da2pc12(ev,n): Project deltas onto 2 leading principal components (PC12) 
# draws2ensemble(draws,nens): [nens] ensemble members using draw-decile norm
# ensemble2BT(da,har,mmd,lut,channel,idx_,cci): ensemble BT from ensemble deltas 
# ev2da(ev,n): [2*n] deltas using CRS
# find_nearest(array,value): Find value and idx closest to target value
# fmt(x,pos): Expoential notation in colorbar labels
# fpe(X1,X2): Fractional percentage error (FPE) between two arrays
# generate_n_single(n): [n] positive-normal random sampling using inverse ERF
# generate_n(n): [n] Constrained random sampling (CRS) with 3 sets of positive-normal trials

#---------------------------------------------------------------------------
import convert_func as convert # measurement equations & L<-->BT conversion
#---------------------------------------------------------------------------

def cov2draws(ds,npop):
    '''
    Sample from the N-normal distribution using the harmonisation parameters as th\
e mean values (best case) and the covariance matrix as the N-variance

    # The multivariate normal, multinormal or Gaussian distribution is a
    # generalization of the 1D-normal distribution to higher dimensions.
    # Such a distribution is specified by its mean and covariance matrix.
    # These parameters are analogous to the mean (average or “center”) and
    # variance (standard deviation, or “width,” squared) of the 1D-normal distribu\
tion.

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivaria\
te_normal.html

    # Harmonisation parameters: (nsensor x npar,)
    # Harmonisation parameter covariance matrix: (nsensor x npar, nsensor x npar)
    '''
    a_ave = ds['parameter']
    a_cov = ds['parameter_covariance_matrix']
    draws = np.random.multivariate_normal(a_ave, a_cov, npop)
    return draws

def cov2ev(X,c):
    '''
    Eigenvalue decomposition of the covariance matrix --> nPC(c=0.99)
    '''
    eigenvalues,eigenvectors = np.linalg.eig(X)
    eigenvalues_cumsum = (eigenvalues/eigenvalues.sum()).cumsum()
    nPC = np.where(eigenvalues_cumsum > c)[0][0] # NB: python indexing
    nPC_variance = eigenvalues_cumsum[nPC]
    print('nPC=',(nPC+1))
    print('nPC_variance=',nPC_variance)
    ev = {}
    ev['eigenvectors'] = eigenvectors
    ev['eigenvalues'] = eigenvalues
    ev['eigenvalues_sum'] = eigenvalues.sum()
    ev['eigenvalues_norm'] = eigenvalues/eigenvalues.sum()
    ev['eigenvalues_cumsum'] = eigenvalues_cumsum
    ev['eigenvalues_prod'] = eigenvalues.prod() # should be ~ det(Xcov)
    ev['eigenvalues_rank'] = np.arange(1,len(eigenvalues)+1) # for plotting
    ev['nPC'] = nPC
    ev['nPC_variance'] = nPC_variance
    return ev

def cov2svd(X,c):
    '''
    Singular value decomposition of the covariance matrix: use PCA if non-square  --> nPC(c=0.99)
    '''
#    if X.shape[0] ~= X.shape[1]:
    if X.shape[0] != X.shape[1]:
        if X.shape[0] < X.shape[1]:
            X = X.T
        X_mean = np.mean(X, axis=0)
        X_sdev = np.std(X, axis=0)
        X_centered = X - X_mean
        X_cov = np.cov(X_centered.T)
        U,S,V = np.linalg.svd(X_centered.T, full_matrices=True)
        eigenvalues, eigenvectors = np.sqrt(S), V # or U.T
    else:
        pca = PCA(n_components=X.shape[1])
        X_transformed = pca.fit_transform(X)
        eigenvalues, eigenvectors = pca.explained_variance_, pca.components_.T
    eigenvalues_cumsum = (eigenvalues/eigenvalues.sum()).cumsum()
    nPC = np.where(eigenvalues_cumsum > c)[0][0] # NB: python indexing
    nPC_variance = eigenvalues_cumsum[nPC]
    print('nPC=',(nPC+1))
    print('nPC_variance=',nPC_variance)
    svd = {}
    svd['eigenvectors'] = eigenvectors
    svd['eigenvalues'] = eigenvalues
    svd['eigenvalues_sum'] = eigenvalues.sum()
    svd['eigenvalues_norm'] = eigenvalues/eigenvalues.sum()
    svd['eigenvalues_cumsum'] = eigenvalues_cumsum
    svd['eigenvalues_prod'] = eigenvalues.prod() # should be ~ det(Xcov)
    svd['eigenvalues_rank'] = np.arange(1,len(eigenvalues)+1) # for plotting
    svd['nPC'] = nPC
    svd['nPC_variance'] = nPC_variance
    return svd

def cov2u(Xcov,N):
    '''
    N-sample MC-estimate of uncertainty from covariance matrix
    (adapted from get_harm.py by Jonathan Mittaz)
    # =======================================
    # Version 0.1
    # 9 July, 2019
    # https://patternizer.github.io/
    # michael.taylor AT reading DOT ac DOT uk
    # =======================================
    '''
    eigenval, eigenvec = np.linalg.eig(Xcov)
    T = np.matmul(eigenvec, np.diag(np.sqrt(eigenval)))
    ndims = Xcov.shape[1]
    position = np.zeros((N, ndims))
    draws = np.zeros((N, ndims))
    for j in range(ndims):
        position[:,:] = 0.
        position[:,j] = np.random.normal(size=N, loc=0., scale=1.)
        for i in range(position.shape[0]):
            vector = position[i,:]
            ovector = np.matmul(T,vector)
            draws[i,:] = draws[i,:]+ovector
    Xu = np.std(draws, axis=0)
    return Xu

def da2pc12(ev,n):
    '''
    Project deltas onto 2 leading principal components (PCs) [nens,npar]
    '''
    nPC = ev['nPC']
    eigenvalues = ev['eigenvalues']
    eigenvectors = ev['eigenvectors']
    nparameters = eigenvectors.shape[1]
    randomized = np.sort(np.array(generate_n(n))) # constrained
    da_pc1 = []
    da_pc2 = []
    for i in range((2*n)):        
        da_pc1.append(randomized[i] * np.sqrt(eigenvalues[0]) * eigenvectors[:,0])
        da_pc2.append(randomized[i] * np.sqrt(eigenvalues[1]) * eigenvectors[:,1])
    da_pc12 = {}
    da_pc12['da_pc1'] = np.array(da_pc1)
    da_pc12['da_pc2'] = np.array(da_pc2)
    return da_pc12

def draws2ensemble(draws,nens):
    '''
    Extract nens (decile) ensemble members
    '''
    npop = draws.shape[0]
    npar = draws.shape[1]
    draws_ave = draws.mean(axis=0)
    draws_std = draws.std(axis=0)
    Z = (draws - draws_ave) / draws_std

    # Extract deciles for each parameter from CDF of draws
    
    decile = np.empty(shape=(nens,npar))
    decile_idx = np.empty(shape=(nens,npar))
    for i in range(npar):

        # CDF (+ sort indices) of draw distribution (for each parameter)

        Z_cdf = np.sort(Z[:,i])
        i_cdf = np.argsort(Z[:,i])

        # Decile values of Z_cdf (for each parameter): use decile mid-points

        idx = (np.linspace(0, (npop-(npop/nens))-1, nens, endpoint=True) + (npop/nens)/2).astype('int')

        for j in range(len(idx)):
            decile[j,i] = Z_cdf[idx[j]]
            decile_idx[j,i] = i_cdf[idx[j]]
        decile_idx = decile_idx.astype(int)

    # Calcaulte norm of draw deltas with respect to deciles to select ensemble members

    Z_norm = np.empty(shape=(nens,npop))
    for i in range(npop):
        for j in range(0,nens):
            Z_norm[j,i] = np.linalg.norm( Z[i,:] - decile[j,:] )

    # Extract ensemble

    ensemble = np.empty(shape=(nens,npar))
    ensemble_idx = np.empty(shape=(nens))
    for j in range(nens):
        # y = np.percentile(Z_norm, deciles[j+1], interpolation='nearest')
        # i = abs(Z_norm - y).argmin()
        idx = np.argmin(Z_norm[j,:])
        ensemble[j,:] = draws[idx,:]
        ensemble_idx[j] = idx
    ensemble_idx = ensemble_idx.astype(int)
    ens = {}
    ens['ensemble'] = ensemble
    ens['ensemble_idx'] = ensemble_idx
    ens['decile'] = decile
    ens['decile_idx'] = decile_idx
    ens['Z'] = Z
    ens['Z_norm'] = Z_norm
    return ens

def ensemble2BT(da, har, mmd, lut, channel, idx_, cci):
    '''
    Function to calculate ensemble BT from ensemble deltas, da and counts
    '''
    noT = False
    a_ave = np.array(har.parameter)
    BT_ens = np.empty(shape=(len(mmd['avhrr-ma_x']),da.shape[0]))
    for i in range(da.shape[0]):
        parameters = da[i,:] + a_ave
        if channel == 3:
            npar = 3
            a0 = parameters[(idx_ *npar)]
            a1 = parameters[(idx_ *npar)+1]
            a2 = parameters[(idx_ *npar)+2]
            a3 = 0.0
            a4 = 0.0
            if noT:
                a2 = 0.0
            Ce = mmd['avhrr-ma_ch3b_earth_counts']
            Cs = mmd['avhrr-ma_ch3b_space_counts']
            Cict = mmd['avhrr-ma_ch3b_bbody_counts']
        elif channel == 4:
            npar = 4
            a0 = parameters[(idx_ *npar)]
            a1 = parameters[(idx_ *npar)+1]
            a2 = parameters[(idx_ *npar)+2]
            a3 = parameters[(idx_ *npar)+3]
            a4 = 0.0
            if noT:
                a3 = 0.0
            Ce = mmd['avhrr-ma_ch4_earth_counts']
            Cs = mmd['avhrr-ma_ch4_space_counts']
            Cict = mmd['avhrr-ma_ch4_bbody_counts']
        else:
            npar = 4
            a0 = parameters[(idx_ *npar)]
            a1 = parameters[(idx_ *npar)+1]
            a2 = parameters[(idx_ *npar)+2]
            a3 = parameters[(idx_ *npar)+3]
            a4 = 0.0
            if noT:
                a3 = 0.0
            Ce = mmd['avhrr-ma_ch5_earth_counts']
            Cs = mmd['avhrr-ma_ch5_space_counts']
            Cict = mmd['avhrr-ma_ch5_bbody_counts']
        Tict = mmd['avhrr-ma_ict_temp'] # equivalent to mmd['avhrr-ma_orbital_temp']
        T_mean = np.mean(Tict[:,3,3])
        T_sdev = np.std(Tict[:,3,3])
        Tinst = (mmd['avhrr-ma_orbital_temperature'][:,3,3] - T_mean) / T_sdev
        WV = 0.0 * Tinst
        if cci:
            Lict = convert.bt2rad_cci(Tict,channel)
            L = convert.counts2rad_cci(channel,Ce,Cs,Cict,Lict)
            BT = convert.rad2bt_cci(L,channel)[:,3,3]
        else:
            Lict = convert.bt2rad(Tict,channel,lut)
            L = convert.counts2rad(Ce,Cs,Cict,Lict,Tinst,WV,channel,a0,a1,a2,a3,a4,noT)
            BT = convert.rad2bt(L,channel,lut)[:,3,3]
        BT_ens[:,i] = BT

    return BT_ens

def ev2da(ev,n):
    '''
    Create (2*n) deltas using constrained random sampling (CRS)
    '''
    nPC = ev['nPC']
    eigenvalues = ev['eigenvalues']
    eigenvectors = ev['eigenvectors']
    nparameters = eigenvectors.shape[1]
    da = np.zeros(shape=(2*n,nparameters))
    for k in range(nPC):
        randomized = np.sort(np.array(generate_n(n)))
        da_c = np.zeros(shape=(2*n,nparameters))
        for i in range((2*n)):        
            da_c[i,:] = randomized[i] * np.sqrt(eigenvalues[k]) * eigenvectors[k,:]
        da = da + da_c
    return da

def find_nearest(array,value):
    '''
    Find value and idx in array X closest to target value
    '''
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx

def fmt(x,pos):
    '''
    Expoential notation in colorbar labels
    '''
    a,b = '{0:.3e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a,b)

def fpe(X1,X2):
    '''
    Calculate the fractional percentage error (FPE) between two arrays
    '''
    FPE = 100.*(1.-np.linalg.norm(X1)/np.linalg.norm(X2))
    return FPE

def generate_n_single(n):
    '''
    Positive-normal random sampling using inverse error function
    (generalisation of crs.py by Jonathan Mittaz to n-sample)
    '''
    # Half width as random number always +ve

    step_size = 1./n
    random_numbers=[]

    # +ve case    
    for i in range(n):
        rno = i*step_size + step_size*np.random.random()
        random_numbers.append(np.sqrt(2.)*erfinv(rno))
    # -ve case
    for i in range(n):
        rno = i*step_size + step_size*np.random.random()
        random_numbers.append(-np.sqrt(2.)*erfinv(rno))
    return random_numbers

def generate_n(n):
    '''
    Constrain 3 trials to normal parameters and re-scramble
    (generalisation of crs.py by Jonathan Mittaz to n-sample)
    '''
    rand1 = generate_n_single(n)
    rand2 = generate_n_single(n)
    rand3 = generate_n_single(n)
    dist1 = np.mean(rand1)**2 + (np.std(rand1)-1.)**2
    dist2 = np.mean(rand2)**2 + (np.std(rand2)-1.)**2
    dist3 = np.mean(rand3)**2 + (np.std(rand3)-1.)**2
    if dist1 < dist2 and dist1 < dist3:
        random_numbers = np.copy(rand1)
    elif dist2 < dist1 and dist2 < dist3:
        random_numbers = np.copy(rand2)
    elif dist3 < dist1 and dist3 < dist2:
        random_numbers = np.copy(rand3)

    # Now randomise the numbers (mix them up)

    in_index = np.arange(2*n).astype(dtype=np.int32)
    in_index_rand = np.random.random(size=(2*n))
    sort_index = np.argsort(in_index_rand)
    index = in_index[sort_index]
    random_numbers = random_numbers[index]
    return random_numbers








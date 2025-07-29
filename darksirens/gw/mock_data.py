import os
from argparse import ArgumentParser
import glob

import astropy
import numpy as np
import healpy as hp

import h5py

from tqdm import tqdm
from scipy.stats import multivariate_normal

from darksirens.utils.cosmology import *

def main():

    optp = ArgumentParser()
    optp.add_argument("--survey_path", help="path to survey data")
    optp.add_argument("--save_path", help="where to save", default='./')
    optp.add_argument("--ngw", type=int, default=1000)
    optp.add_argument("--nsamp", type=int, default=4096)
    optp.add_argument("--seed", type=int, default=22)

    opts = optp.parse_args()

    survey_path = opts.survey_path
    save_path = opts.save_path
    ngw = opts.ngw
    nsamp = opts.nsamp
    seed = opts.seed

    with h5py.File(survey_path, 'r') as f:
        ra_gal = np.asarray(f['ra_gal'])*np.pi/180
        dec_gal= np.asarray(f['dec_gal'])*np.pi/180
        z_gal = np.asarray(f['z_gal'])
        
    ngal = len(ra_gal)
        
    rng = np.random.default_rng(seed=seed)
    i_gw_gal = rng.choice(np.arange(ngal), ngw, replace=False)

    ra_gal_gw = ra_gal[i_gw_gal]
    dec_gal_gw = dec_gal[i_gw_gal]
    dL_gal_gw = dL_of_z(z_gal[i_gw_gal],H0Planck)

    m1s = np.random.normal(35,5,len(i_gw_gal))
    m2s = np.random.normal(35,5,len(i_gw_gal))
    m2s_gal_gw, m1s_gal_gw = np.sort([m1s,m2s],axis=0)

    m1sdet_gal_gw = m1s_gal_gw*(1+z_gal[i_gw_gal])
    m2sdet_gal_gw = m2s_gal_gw*(1+z_gal[i_gw_gal])

    m1dets = []
    m2dets = []
    dLs = []
    ras = []
    decs = []

    for k in tqdm(range(int(len(i_gw_gal)))):
        m1det = m1sdet_gal_gw[k]
        m2det = m2sdet_gal_gw[k]
        dL = dL_gal_gw[k]
        ra = ra_gal_gw[k]
        dec = dec_gal_gw[k]
        mean = np.array([m1det, m2det, dL, ra, dec])

        cov = np.diag([1.5**2,1.5**2,(dL),0.01**2,0.01**2])
        rv = multivariate_normal(mean, cov)
        samples = rv.rvs([256000])

        dec_samples = samples[:,4]
        mask = np.where((dec_samples>-np.pi/2)&(dec_samples<np.pi/2))
        samples = samples[mask]

        choose = np.random.randint(0,len(samples),nsamp)
        samples = samples[choose]

        m2d, m1d = np.sort([samples[:,0],samples[:,1]],axis=0)
        m1dets.append(m1d)
        m2dets.append(m2d)
        dLs.append(samples[:,2])
        ras.append(samples[:,3] % (2 * np.pi))
        decs.append(samples[:,4])

    m1dets = np.concatenate(m1dets)
    m2dets = np.concatenate(m2dets)
    ras = np.concatenate(ras)
    decs = np.concatenate(decs)
    dLs = np.concatenate(dLs)

    with h5py.File(save_path+'lognormal_pixelated_gws.h5', 'w') as f:
        f.attrs['nsamp'] = nsamp
        f.attrs['nobs'] = ngw
        f.create_dataset('m1det', data=m1dets, compression='gzip', shuffle=False)
        f.create_dataset('m2det', data=m2dets, compression='gzip', shuffle=False)
        f.create_dataset('dL', data=dLs, compression='gzip', shuffle=False)
        f.create_dataset('ra', data=ras, compression='gzip', shuffle=False)
        f.create_dataset('dec', data=decs, compression='gzip', shuffle=False)
        
if __name__ == "__main__":
    main()

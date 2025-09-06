import os
from argparse import ArgumentParser
import glob

import numpy as np
import h5py

from tqdm import tqdm

from pathos.multiprocessing import ProcessingPool, Pool
import tqdm_pathos

import healpy as hp

def main():

    optp = ArgumentParser()
    optp.add_argument("--survey_path", help="path to survey data")
    optp.add_argument("--save_path", help="where to save", default='./')
    optp.add_argument("--nside", type=int, default=64)

    opts = optp.parse_args()

    survey_path = opts.survey_path
    save_path = opts.save_path
    nside = opts.nside

    with h5py.File(survey_path, 'r') as f:
        ras_ = np.array(f['TARGET_RA'])*np.pi/180
        decs_ = np.array(f['TARGET_DEC'])*np.pi/180
        zs_ = np.array(f['Z'])
        try:
            ddzs_ = np.array(inp['ZERR'])
            wts_ = np.array(inp['WEIGHT'])
        except:
            dz = 0.033
            ddzs_ = dz*(1+zs_)
            wts_ = np.ones(len(zs_))

    ngals = len(ras_)

    npix = hp.pixelfunc.nside2npix(nside)
    pixgrid = np.arange(npix)

    ind = hp.pixelfunc.ang2pix(nside,np.pi/2-decs_,ras_)

    def calculate_galaxies_pix(pix):
        cats = []
        ngalaxies = []
        idx = np.where(ind == pix)[0]
        return idx

    p = ProcessingPool(20)
    results = tqdm_pathos.map(calculate_galaxies_pix, list(pixgrid))

    ngalaxies_ = []
    for pix in tqdm(pixgrid):
        idx = results[pix]
        gals = zs_[idx]
        ngals = gals.shape[0]
        ngalaxies_.append(ngals)
    maxgals = max(ngalaxies_)
    print(maxgals)

    cats = []
    dzcats = []
    dwcats = []
    ngalaxies = []
    for pix in tqdm(pixgrid):
        idx = results[pix]
        gals = zs_[idx]
        dgals = ddzs_[idx]
        wgals = wts_[idx]
        ngals = gals.shape[0]
        
        zgals = [gals]
        dzgals = [dgals]
        dwgals = [wgals]

        if ngals < maxgals:
            lenght = int(maxgals - ngals)
            zgals.append(100*np.ones(lenght))
            dzgals.append(1*np.ones(lenght))
            dwgals.append(np.zeros(lenght))

        cats.append(np.concatenate(zgals))
        dzcats.append(np.concatenate(dzgals))
        dwcats.append(np.concatenate(dwgals))
        ngalaxies.append(ngals)

    del results
    p.close()

    with h5py.File(save_path + 'lognormal_pixelated_nside_'+str(nside)+'_galaxies.h5', 'w') as f:
        f.attrs['nside'] = nside
        f.create_dataset('zgals', data=np.asarray(cats), compression='gzip', shuffle=False)
        f.create_dataset('dzgals', data=np.asarray(dzcats), compression='gzip', shuffle=False)
        f.create_dataset('wgals', data=np.asarray(dwcats), compression='gzip', shuffle=False)
        f.create_dataset('ngals', data=np.asarray(ngalaxies), compression='gzip', shuffle=False)
        
if __name__ == "__main__":
    main()


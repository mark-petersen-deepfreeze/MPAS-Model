#!/usr/bin/env python
'''
This script creates an initial condition file for MPAS-Ocean.
'''
import os
import shutil
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
import argparse
import math
import time
import subprocess
verbose = True

def main():
    timeStart = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default='culled_mesh.nc',
                        help='Input file, containing global mesh'
                        )
    parser.add_argument('-o', '--output_file', dest='output_file',
                        default='with_cullCell.nc',
                        help='Output file, containing initial variables'
                        )
    ds = xr.open_dataset(parser.parse_args().input_file)

    #comment('obtain dimensions and mesh variables')
    time1 = time.time()
    nCells = ds['nCells'].size
    latCell=ds['latCell']
    lonCell=ds['lonCell']
    latS = -50.0*np.pi/180.0
    latN =  50.0*np.pi/180.0

    # create reduced size mesh for SH
    cullCell = np.where(latCell<latS,0,1)
    ds['cullCell'] = (('nCells'), cullCell)
    ds.to_netcdf('temp/step1_mesh_with_cullCell_SH.nc', format='NETCDF4')
    subprocess.check_call(['MpasCellCuller.x','temp/step1_mesh_with_cullCell_SH.nc','temp/step2_subdomain_SH.nc'])
    subprocess.check_call(['MpasMeshConverter.x','temp/step2_subdomain_SH.nc','temp/step3_subdomain_SH.nc'])

    temp = np.zeros(nCells)
    temp[:] = latCell[:]
    oldCellID = np.argwhere(temp[:]<latS).flatten().astype(int)
    print('type(oldCellID)',type(oldCellID[0]))
    dsSH = xr.open_dataset('temp/step3_subdomain_SH.nc')
    dsSH['oldCellID'] = (('nCells'), oldCellID)
    dsSH.to_netcdf('subdomain_SH.nc', format='NETCDF4')  
    dsSH.close()

    # create reduced size mesh for NH
    cullCell = np.where(latCell>latN,0,1)
    ds['cullCell'] = (('nCells'), cullCell)
    ds.to_netcdf('temp/step1_mesh_with_cullCell_NH.nc', format='NETCDF4')
    ds.close()
    subprocess.check_call(['MpasCellCuller.x','temp/step1_mesh_with_cullCell_NH.nc','temp/step2_subdomain_NH.nc'])
    subprocess.check_call(['MpasMeshConverter.x','temp/step2_subdomain_NH.nc','temp/step3_subdomain_NH.nc'])

    temp = np.zeros(nCells)
    temp[:] = latCell[:]
    oldCellID = np.argwhere(temp[:]>latN).flatten().astype(int)
    dsNH = xr.open_dataset('temp/step3_subdomain_NH.nc')
    dsNH['oldCellID'] = (('nCells'), oldCellID)
    dsNH.to_netcdf('subdomain_NH.nc', format='NETCDF4')  
    dsNH.close()

    print('   time: %f' % ((time.time() - time1)))
    print('Total time: %f' % ((time.time() - timeStart)))

def comment(string):
    if verbose:
        print('***   ' + string)


if __name__ == '__main__':
    # If called as a primary module, run main
    main()

#!/usr/bin/env python
'''
This script plots a convergence test for the sea ice divergence of the stress.
'''

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import argparse
matplotlib.use('Agg')


# mrp: read from file mesh after nx,ny attributes are added:
#nx = 10  # ncfileMesh.getncattr('nx')
#ny = 10  # ncfileMesh.getncattr('ny')
#iz = 5

tests = ['convergenceTests']
nTests = len(tests)
grids = ['hex_0082x0094', 'hex_0082x0094', 'hex_0082x0094', 'hex_0082x0094']
dx = [2.5, 5, 10, 20]
order2 = [0.1, 0.4, 1.6, 6.4] 
order3 = [0.1, 0.8, 6.4, 51.2] 
nGrids = len(grids)
operators = ['stressDivergenceUCGrid', 'stressDivergenceVCGrid'] 
operatorsAnalytical = ['stressDivergenceUAnalytical', 'stressDivergenceVAnalytical'] 
nOperators = len(operators)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', dest='input_file',
                        default='grid.nc',
                        help='Input file, containing base mesh'
                        )
ds = xr.open_dataset(parser.parse_args().input_file)

nCells = ds['nCells'].size
nVertices = ds['nVertices'].size
nEdgesOnCell = ds['nEdgesOnCell'].values

l2 = np.zeros([nTests, nGrids, nOperators])
L2 = np.zeros([nTests, nGrids, nOperators])

for i in range(nTests):
    test = tests[i]

    for k in range(nOperators): 
        for j in range(nGrids):
            grid = grids[j]
            #ncfileIC = Dataset('../' + test + '_' + grid + '/init.nc', 'r') 
            ncfile = Dataset('/lustre/scratch3/turquoise/gcapodaglio/runs/seaice_runs/c-grid_seaice/testing_and_setup/seaice/testcases/square/operators_stress_divergence_c_grid/' + test + '/'  + grid + '/output.nc', 'r')
            ncfileIC = Dataset('/lustre/scratch3/turquoise/gcapodaglio/runs/seaice_runs/c-grid_seaice/testing_and_setup/seaice/testcases/square/operators_stress_divergence_c_grid/' + test + '/'  + grid + '/ic.nc', 'r')
            operator = operators[k] 
            operatorAnalytical = operatorsAnalytical[k]
            var = ncfile.variables[operator][0, :]
            sol = ncfileIC.variables[operatorAnalytical][:]
            #areas = ncfile.variables['areaCell'][:]
            dif = abs(var - sol)
            #multDen = (sol**2)*areas
            numl2 = dif**2
            #multNum = (dif**2)*areas
            l2[i, j, k] = np.sqrt(np.sum(numl2[:]))
            #denL2 = np.sum(multDen[:])/np.sum(areas[:])
            #numL2 = np.sum(multNum[:])/np.sum(areas[:]) 
            #L2[i, j, k] = np.sqrt(numL2)
            ncfile.close()
            ncfileIC.close()
            #print(L2)
            #print(l2)
for i in range(nTests):
    test = tests[i]
    for k in range(len(operators)):
        operator =operators[k]
        plt.loglog(dx, l2[i, :, k], '-x', label='l2-norm div of stress', linewidth=2.0, marker='x', markersize=10)
        #plt.loglog(dx, L2[i, :, k], '-x', label='scaled L2', linewidth=4.0, marker='+', markersize=10)
    #plt.loglog(dx, order2, 'k', label='slope=2', linestyle='dashed', linewidth=4.0)
    #plt.loglog(dx, order3, 'k', label='slope=3', linestyle='dotted', linewidth=4.0)
    #plt.xlim(1.e3, 1.e6) 
    #plt.ylim(1.e-7, 1.e-2)
    plt.title('Convergence rate of div of stress') 
    plt.ylabel('error norms')
    plt.legend()
    plt.grid()
plt.xticks(dx,dx)
plt.xlabel('dx')

plt.savefig('convOrders.png')


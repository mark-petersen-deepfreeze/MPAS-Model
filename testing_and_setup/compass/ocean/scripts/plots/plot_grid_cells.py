#!/usr/bin/env python
"""
This script creates historgram plots of the initial condition.
"""
# import modules
from netCDF4 import Dataset
import numpy as np
import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cartopy.crs as ccrs
import cartopy


def main():
    # parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input_file_name', dest='input_file_name',
        default='output.nc',
        help='MPAS file name for input of initial state.')
    parser.add_argument(
        '-m', '--mesh_file_name', dest='mesh_file_name',
        default='init.nc',
        help='MPAS file name for mesh variables.')
    parser.add_argument(
        '-o', '--output_file_name', dest='output_file_name',
        default='grid_cell_plot.png',
        help='File name for output image.')
    args = parser.parse_args()

    degToRad = 3.1415/180.
    minLat =  30 * degToRad
    maxLat =  50 * degToRad
    minLon = (-90 + 360) * degToRad
    maxLon = (-60 + 360)* degToRad
    iLev = 0

    # load mesh variables
    dataFile = Dataset(args.input_file_name, 'r')
    meshFile = Dataset(args.mesh_file_name, 'r')
    nCells = dataFile.dimensions['nCells'].size
    nEdges = dataFile.dimensions['nEdges'].size
    nVertLevels = dataFile.dimensions['nVertLevels'].size
    latCell = meshFile.variables['latCell'][:]
    lonCell = meshFile.variables['lonCell'][:]

    ind = np.where((latCell>minLat) & (latCell<maxLat) 
        & (lonCell>minLon) & (lonCell<maxLon))

    #ind = np.where(latCell<-60*degToRad) 
    fig = plt.figure()
    fig.set_size_inches(16.0, 12.0)
    plt.clf()

    print('plotting zoomed-in area to see noise...')

    varName = 'temperature'
    var = dataFile.variables[varName][0,:,iLev]
    print(var[ind])
    plt.scatter(lonCell[ind],latCell[ind],c=var[ind])
    plt.colorbar()
    plt.savefig(args.output_file_name)

    #d = datetime.datetime.today()
    #txt = \
    #    'MPAS-Ocean initial state\n' + \
    #    'date: {}\n'.format(d.strftime('%m/%d/%Y')) + \
    #    'number cells: {}\n'.format(nCells) + \
    #    'number cells, millions: {:6.3f}\n'.format(nCells / 1.e6) + \
    #    'number layers: {}\n\n'.format(nVertLevels) + \
    #    '  min val   max val  variable name\n'

    #plt.subplot(3, 3, 2)
    #varName = 'maxLevelCell'
    #var = dataFile.variables[varName]
    #maxLevelCell = var[:]
    #plt.hist(var, bins=nVertLevels - 4)
    #plt.ylabel('frequency')
    #plt.xlabel(varName)
    #txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, np.amin(var), np.amax(var), varName)

    #plt.subplot(3, 3, 3)
    #varName = 'bottomDepth'
    #var = dataFile.variables[varName]
    #plt.hist(var, bins=nVertLevels - 4)
    #plt.xlabel(varName)
    #txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, np.amin(var), np.amax(var), varName)

    #cellsOnEdge = dataFile.variables['cellsOnEdge']
    #cellMask = np.zeros((nCells, nVertLevels), bool)
    #edgeMask = np.zeros((nEdges, nVertLevels), bool)
    #for k in range(nVertLevels):
    #    cellMask[:, k] = k < maxLevelCell
    #    cell0 = cellsOnEdge[:, 0]-1
    #    cell1 = cellsOnEdge[:, 1]-1
    #    edgeMask[:, k] = np.logical_and(np.logical_and(cellMask[cell0, k],
    #                                                   cellMask[cell1, k]),
    #                                    np.logical_and(cell0 >= 0,
    #                                                   cell1 >= 0))

    #plt.subplot(3, 3, 4)
    #varName = 'temperature'
    #var = dataFile.variables[varName][0, :, :][cellMask]
    #plt.hist(var, bins=100, log=True)
    #plt.ylabel('frequency')
    #plt.xlabel(varName)
    #txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, np.amin(var), np.amax(var), varName)

    #plt.subplot(3, 3, 5)
    #varName = 'salinity'
    #var = dataFile.variables[varName][0, :, :][cellMask]
    #plt.hist(var, bins=100, log=True)
    #plt.xlabel(varName)
    #txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, np.amin(var), np.amax(var), varName)

    #plt.subplot(3, 3, 6)
    #varName = 'layerThickness'
    #var = dataFile.variables[varName][0, :, :][cellMask]
    #plt.hist(var, bins=100, log=True)
    #plt.xlabel(varName)
    #txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, np.amin(var), np.amax(var), varName)

    #rx1Edge = dataFile.variables['rx1Edge']
    #plt.subplot(3, 3, 7)
    #varName = 'rx1Edge'
    #var = dataFile.variables[varName][0, :, :][edgeMask]
    #plt.hist(var, bins=100, log=True)
    #plt.ylabel('frequency')
    #plt.xlabel('Haney Number, max={:4.2f}'.format(
    #    np.max(rx1Edge[:].ravel())))
    #txt = '{}{:9.2e} {:9.2e} {}\n'.format(txt, np.amin(var), np.amax(var), varName)

    #font = FontProperties()
    #font.set_family('monospace')
    #font.set_size(12)
    #print(txt)
    #plt.subplot(3, 3, 1)
    #plt.text(0, 1, txt, verticalalignment='top', fontproperties=font)
    #plt.axis('off')

    #plt.savefig(args.output_file_name)


if __name__ == '__main__':
    # If called as a primary module, run main
    main()

# vim: foldmethod=marker ai ts=4 sts=4 et sw=4 ft=python

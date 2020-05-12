#!/usr/bin/env python
"""
This script creates historgram plots of the initial condition.
"""
# import modules
from netCDF4 import Dataset
import numpy as np
import xarray
import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
import cartopy.crs as crs
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
    radToDeg = 180./3.1415
    km=1e3

    # load mesh variables
    dataFile = Dataset(args.input_file_name, 'r')
    dsMesh = xarray.open_dataset(args.mesh_file_name)
    meshFile = Dataset(args.mesh_file_name, 'r')
    nCells = dsMesh.sizes['nCells']
    nEdges = dsMesh.sizes['nEdges']
    nVertLevels = dsMesh.sizes['nVertLevels']
    nVerticesOnCell = dsMesh.nEdgesOnCell.values
    verticesOnCell = dsMesh.verticesOnCell.values - 1
    latCell = dsMesh.latCell.values
    lonCell = dsMesh.lonCell.values
    xCell = dsMesh.xCell.values
    yCell = dsMesh.yCell.values

    trans = crs.PlateCarree()

    # create patches
    patches = []
    #projection='latlon'
    projection='polarOrthographic'
    if projection=='latlon':
        #latMin = -80; latMax = -40; lonMin = 270; lonMax = 330; # Drake Passage, zoom out
        latMin = -65; latMax = -54; lonMin = 285; lonMax = 310; # Drake Passage, zoom in
        lonVertex = dsMesh.lonVertex.values
        latVertex = dsMesh.latVertex.values
        ind = np.where((latCell>latMin*degToRad) & (latCell<latMax*degToRad) 
            & (lonCell>lonMin*degToRad) & (lonCell<lonMax*degToRad))[0]
        extents = [lonMin,lonMax,latMin,latMax]
    elif projection=='polarOrthographic':
        xMin=-1400; xMax=-800; yMin=-1800; yMax=-1200; # Ross Sea, zoom out
        #xMin=-1300; xMax=-900; yMin=-1600; yMax=-1200; # Ross Sea
        #xMin=-3000; xMax=3000; yMin=-3000; yMax=3000; # Wide view of Antarctica
        xVertex = dsMesh.xVertex.values
        yVertex = dsMesh.yVertex.values
        ind = np.where((latCell<-40.0*degToRad)
            & (xCell>yMin*km) & (xCell<yMax*km)
            & (yCell>xMin*km) & (yCell<xMax*km))[0]
        extents = [xMin, xMax, yMin, yMax]

    for iCell in ind:
        # use mask later
        #if(not mask[iCell]):
        #    continue
        nVert = nVerticesOnCell[iCell]
        vertexIndices = verticesOnCell[iCell, :nVert]
        vertices = np.zeros((nVert, 2))
        if projection=='latlon':
            vertices[:, 0] = lonVertex[vertexIndices]*radToDeg
            vertices[:, 1] = latVertex[vertexIndices]*radToDeg
        elif projection=='polarOrthographic':
            vertices[:, 0] =  yVertex[vertexIndices]*1e-3
            vertices[:, 1] =  xVertex[vertexIndices]*1e-3
        polygon = Polygon(vertices, True)
        patches.append(polygon)
    #localPatches = PatchCollection(patches, cmap='jet', alpha=1., transform=trans)
    localPatches = PatchCollection(patches, cmap='jet', alpha=1.)

    fig = plt.figure()
    fig.set_size_inches(16.0, 12.0)
    plt.clf()

    print('plotting zoomed-in area to see noise...')

    varNames = ['temperature','divergence','vertVelocityTop','frazilLayerThicknessTendency']
    levs = [0,5,10,20]
    nCols = len(varNames)
    nRows = len(levs)
    iTime = 3
    #fig,axes = plt.subplots(nrows=nRows,ncols=nCols,figsize=(16,12))
    for iCol in range(nCols):
        for iRow in range(nRows):
            varName = varNames[iCol]
            #print(iCol,iRow)
            ax = fig.add_subplot(nRows,nCols,1+iCol+nRows*iRow)
            varData = dataFile.variables[varName][iTime,:,levs[iRow]]
            yLabel = 'level: ' + str(levs[iRow])
            if varName=='temperature':
                indLand = np.where(varData<-1e33)
                varData[indLand] = 0.0
            if varName=='frazilLayerThicknessTendency':
                if iRow==2:
                    varName='accumulatedFrazilIceMass'
                    varData = dataFile.variables[varName][iTime,:]
                    yLabel = 'column sum'
                elif iRow==3:
                    varName='accumulatedFrazilIceSalinity'
                    varData = dataFile.variables[varName][iTime,:]
                    yLabel = 'column sum'
            localPatches = PatchCollection(patches, cmap='jet', alpha=1.)
            localPatches.set_array(varData[ind])
            ax.add_collection(localPatches)
            plt.axis(extents)
            plt.title(varName)
            plt.ylabel('level: ' + str(levs[iRow]))
            plt.grid('true')
            if iRow<nRows-1:
                ax.set_xticklabels([])
            if iCol>0:
                ax.set_yticklabels([])
            plt.colorbar(localPatches)
            if varName!='temperature':
                maxAbsVal = 0.8*max(abs(varData[ind]))
                localPatches.set_clim(-maxAbsVal,maxAbsVal)

    plt.savefig(args.output_file_name)

#    #ax.set_global()
## lat/lon
#    #ax.set_extent([lonMin*radToDeg, lonMax*radToDeg, latMin*radToDeg, latMax*radToDeg])
#    #ax.gridlines()
#    plt.colorbar(localPatches)
#    varData = dataFile.variables[varName][iTime,:,iLev]
#    #ax = plt.subplot(1,1,1, projection=trans)
#    ax = plt.subplot(1,1,1)
#    localPatches.set_array(varData[ind])
#    #localPatches.set_transform(trans)
#    ax.add_collection(localPatches)
#    R=4000.
#    #plt.axis([-R, R, -R, R])
#    plt.axis([xMin, xMax, yMin, yMax])
#    #ax.set_global()
## lat/lon
#    #ax.set_extent([lonMin*radToDeg, lonMax*radToDeg, latMin*radToDeg, latMax*radToDeg])
#    #ax.gridlines()
#    plt.colorbar(localPatches)
#    #ax.coastlines()
#    #plt.axis([0, 500, 0, 1000])
#    #ax.set_aspect('equal')
#    #ax.autoscale(tight=True)


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
def _compute_cell_patches(dsMesh, mask, cmap):
    patches = []
    nVerticesOnCell = dsMesh.nEdgesOnCell.values
    verticesOnCell = dsMesh.verticesOnCell.values - 1
    #lonVertex = dsMesh.lonVertex.values
    #latVertex = dsMesh.latVertex.values
    for iCell in range(dsMesh.sizes['nCells']):
        if(not mask[iCell]):
            continue
        nVert = nVerticesOnCell[iCell]
        vertexIndices = verticesOnCell[iCell, :nVert]
        vertices = numpy.zeros((nVert, 2))
# lat/lon
        #vertices[:, 0] = 1e-3*lonVertex[vertexIndices]
        #vertices[:, 1] = 1e-3*latVertex[vertexIndices]
# polar orthographic
        vertices[:, 0] = 1e-3*lonVertex[vertexIndices]
        vertices[:, 1] = 1e-3*latVertex[vertexIndices]
        polygon = Polygon(vertices, True)
        patches.append(polygon)
    p = PatchCollection(patches, cmap=cmap, alpha=1.)
    return p

if __name__ == '__main__':
    # If called as a primary module, run main
    main()

# vim: foldmethod=marker ai ts=4 sts=4 et sw=4 ft=python

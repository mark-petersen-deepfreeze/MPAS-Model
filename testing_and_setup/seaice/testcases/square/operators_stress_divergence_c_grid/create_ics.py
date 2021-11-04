from netCDF4 import Dataset
import numpy as np
from math import sin, cos, pi, radians

#-------------------------------------------------------------

def velocities_strains_stress_divergences(x, y):

    A = 2.56#np.random.uniform(2,4)
    B = 2.56#np.random.uniform(2,4)
    C = 2.56#np.random.uniform(2,4)
    D = 2.56#np.random.uniform(2,4)

    Lx = 1.0
    Ly = 1.0

    u = sin((2.0 * pi * x * A) / Lx) * sin((2.0 * pi * y * B) / Ly)
    v = sin((2.0 * pi * x * C) / Lx) * sin((2.0 * pi * y * D) / Ly)

    dudx = ((2.0 * pi * A) / Lx) * cos((2.0 * pi * x * A) / Lx) * sin((2.0 * pi * y * B) / Ly)
    dudy = ((2.0 * pi * B) / Ly) * sin((2.0 * pi * x * A) / Lx) * cos((2.0 * pi * y * B) / Ly)

    dvdx = ((2.0 * pi * C) / Lx) * cos((2.0 * pi * x * C) / Lx) * sin((2.0 * pi * y * D) / Ly)
    dvdy = ((2.0 * pi * D) / Ly) * sin((2.0 * pi * x * C) / Lx) * cos((2.0 * pi * y * D) / Ly)

    d2udx2  = -((2.0 * pi * A) / Lx)*((2.0 * pi * A) / Lx) * sin((2.0 * pi * x * A) / Lx) * sin((2.0 * pi * y * B) / Ly)
    d2udy2  = -((2.0 * pi * B) / Ly)*((2.0 * pi * B) / Ly) * sin((2.0 * pi * x * A) / Lx) * sin((2.0 * pi * y * B) / Ly)
    d2udxdy =  ((2.0 * pi * A) / Lx)*((2.0 * pi * B) / Ly) * cos((2.0 * pi * x * A) / Lx) * cos((2.0 * pi * y * B) / Ly)

    d2vdx2  = -((2.0 * pi * C) / Lx)*((2.0 * pi * C) / Lx) * sin((2.0 * pi * x * C) / Lx) * sin((2.0 * pi * y * D) / Ly)
    d2vdy2  = -((2.0 * pi * D) / Ly)*((2.0 * pi * D) / Ly) * sin((2.0 * pi * x * C) / Lx) * sin((2.0 * pi * y * D) / Ly)
    d2vdxdy =  ((2.0 * pi * C) / Lx)*((2.0 * pi * D) / Ly) * cos((2.0 * pi * x * C) / Lx) * cos((2.0 * pi * y * D) / Ly)

    e11 = dudx
    e22 = dvdy
    e12 = 0.5 * (dudy + dvdx)

    de11dx = d2udx2

    de12dy = 0.5 * (d2udy2  + d2vdxdy)
    de12dx = 0.5 * (d2udxdy + d2vdx2)

    de22dy = d2vdy2

    divu = de11dx + de12dy
    divv = de12dx + de22dy

    return u, v, e11, e22, e12, divu, divv

#-------------------------------------------------------------

def create_ic(gridfile, icfile):

    # load grid file
    grid = Dataset(gridfile, "r")
    with open('./namelist.seaice.stress_divergence') as namelistFile:
            namelistTxt = namelistFile.read().split('\n')
            for line in namelistTxt:
                if 'config_use_c_grid = ' in line:
                    config_use_c_grid = line.split()[-1]
                    break

    nCells = len(grid.dimensions["nCells"])
    nVertices = len(grid.dimensions["nVertices"])
    nEdges = len(grid.dimensions["nEdges"])
    maxEdges = len(grid.dimensions["maxEdges"])
    vertexDegree = len(grid.dimensions["vertexDegree"])

    nEdgesOnCell = grid.variables["nEdgesOnCell"][:]
    verticesOnCell = grid.variables["verticesOnCell"][:]
    edgesOnCell = grid.variables["edgesOnCell"][:]
    edgesOnVertex = grid.variables["edgesOnVertex"][:]
    verticesOnCell[:] = verticesOnCell[:] - 1
    edgesOnCell[:] = edgesOnCell[:] - 1
    edgesOnVertex[:] = edgesOnVertex[:] - 1

    xCell = grid.variables["xCell"][:]
    yCell = grid.variables["yCell"][:]

    xVertex = grid.variables["xVertex"][:]
    yVertex = grid.variables["yVertex"][:]

    xEdge = grid.variables["xEdge"][:]
    yEdge = grid.variables["yEdge"][:]

    grid.close()

    xMin = np.amin(xVertex)
    xMax = np.amax(xVertex)
    yMin = np.amin(yVertex)
    yMax = np.amax(yVertex)

    xMinEdge = np.amin(xEdge)
    xMaxEdge = np.amax(xEdge)
    yMinEdge = np.amin(yEdge)
    yMaxEdge = np.amax(yEdge)

    # calculate output variables
    uVelocity = np.empty(nVertices)
    vVelocity = np.empty(nVertices)
    uVelocityCGrid = np.empty(nEdges)
    vVelocityCGrid = np.empty(nEdges)

    stressDivergenceU = np.empty(nVertices)
    stressDivergenceV = np.empty(nVertices)
    stressDivergenceUCGrid = np.empty(nEdges)
    stressDivergenceVCGrid = np.empty(nEdges)

    strain11Vertex = np.zeros(nVertices)
    strain22Vertex = np.zeros(nVertices)
    strain12Vertex = np.zeros(nVertices)
    strain11Edge = np.zeros(nEdges)
    strain22Edge = np.zeros(nEdges)
    strain12Edge = np.zeros(nEdges)

    solveVelocity = np.ones(nVertices,dtype="i")
    solveVelocityPrevious = np.ones(nVertices,dtype="i")
    solveVelocityCGrid = np.ones(nEdges,dtype="i")
    solveVelocityPreviousCGrid = np.ones(nEdges,dtype="i")
    solveStress = np.ones(nCells,dtype="i")
    solveStressTri = np.ones(nVertices,dtype="i")

    for iVertex in range(0,nVertices):

        x = (xVertex[iVertex] - xMin)
        y = (yVertex[iVertex] - yMin)

        u, v, e11, e22, e12, divu, divv = velocities_strains_stress_divergences(x, y)

        uVelocity[iVertex] = u
        vVelocity[iVertex] = v

        strain11Vertex[iVertex] = e11
        strain22Vertex[iVertex] = e22
        strain12Vertex[iVertex] = e12

        stressDivergenceU[iVertex] = divu
        stressDivergenceV[iVertex] = divv

    for iEdge in range(0,nEdges):

        x = (xEdge[iEdge] - xMinEdge)
        y = (yEdge[iEdge] - yMinEdge)

        u, v, e11, e22, e12, divu, divv = velocities_strains_stress_divergences(x, y)

        uVelocityCGrid[iEdge] = u
        vVelocityCGrid[iEdge] = v

        strain11Edge[iEdge] = e11
        strain22Edge[iEdge] = e22
        strain12Edge[iEdge] = e12

        stressDivergenceUCGrid[iEdge] = divu
        stressDivergenceVCGrid[iEdge] = divv

    stress11var = np.zeros((nCells, maxEdges))
    stress22var = np.zeros((nCells, maxEdges))
    stress12var = np.zeros((nCells, maxEdges))
    stress11varTri = np.zeros((nVertices, vertexDegree))
    stress22varTri = np.zeros((nVertices, vertexDegree))
    stress12varTri = np.zeros((nVertices, vertexDegree))

    if (config_use_c_grid == 'true') :
       # if C-grid
       for iCell in range(0, nCells):
           for iEdgeOnCell in range(0,nEdgesOnCell[iCell]):
               iEdge = edgesOnCell[iCell,iEdgeOnCell]
               stress11var[iCell,iEdgeOnCell] = strain11Edge[iEdge]
               stress22var[iCell,iEdgeOnCell] = strain22Edge[iEdge]
               stress12var[iCell,iEdgeOnCell] = strain12Edge[iEdge]

       for iVertex in range(0, nVertices):
           for iDegree in range(0,vertexDegree):
              iEdge = edgesOnVertex[iVertex,iDegree]

              stress11varTri[iVertex,iDegree] = strain11Edge[iEdge]
              stress22varTri[iVertex,iDegree] = strain22Edge[iEdge]
              stress12varTri[iVertex,iDegree] = strain12Edge[iEdge]

    else :
       # if B-grid
       for iCell in range(0, nCells):
          for iVertexOnCell in range(0,nEdgesOnCell[iCell]):
             iVertex = verticesOnCell[iCell,iVertexOnCell]
             stress11var[iCell,iVertexOnCell] = strain11Vertex[iVertex]
             stress22var[iCell,iVertexOnCell] = strain22Vertex[iVertex]
             stress12var[iCell,iVertexOnCell] = strain12Vertex[iVertex]

    # create output file
    fileOut = Dataset(icfile, "w", format="NETCDF3_CLASSIC")

    fileOut.createDimension("nVertices", nVertices)
    fileOut.createDimension("nEdges", nEdges)
    fileOut.createDimension("nCells", nCells)
    fileOut.createDimension("maxEdges", maxEdges)
    fileOut.createDimension("vertexDegree", vertexDegree)

    var = fileOut.createVariable("uVelocity","d",dimensions=["nVertices"])
    var[:] = uVelocity[:]

    var = fileOut.createVariable("vVelocity","d",dimensions=["nVertices"])
    var[:] = vVelocity[:]

    var = fileOut.createVariable("uVelocityCGrid","d",dimensions=["nEdges"])
    var[:] = uVelocityCGrid[:]

    var = fileOut.createVariable("vVelocityCGrid","d",dimensions=["nEdges"])
    var[:] = vVelocityCGrid[:]

    var = fileOut.createVariable("solveVelocity","i",dimensions=["nVertices"])
    var[:] = solveVelocity[:]

    var = fileOut.createVariable("solveVelocityPrevious","i",dimensions=["nVertices"])
    var[:] = solveVelocityPrevious[:]

    var = fileOut.createVariable("solveVelocityCGrid","i",dimensions=["nEdges"])
    var[:] = solveVelocityCGrid[:]

    var = fileOut.createVariable("solveVelocityPreviousCGrid","i",dimensions=["nEdges"])
    var[:] = solveVelocityPreviousCGrid[:]

    var = fileOut.createVariable("solveStress","i",dimensions=["nCells"])
    var[:] = solveStress[:]

    var = fileOut.createVariable("solveStressTri","i",dimensions=["nVertices"])
    var[:] = solveStressTri[:]

    var = fileOut.createVariable("strain11var","d",dimensions=["nCells","maxEdges"])
    var[:] = stress11var[:]

    var = fileOut.createVariable("strain22var","d",dimensions=["nCells","maxEdges"])
    var[:] = stress22var[:]

    var = fileOut.createVariable("strain12var","d",dimensions=["nCells","maxEdges"])
    var[:] = stress12var[:]

    var = fileOut.createVariable("strain11varTri","d",dimensions=["nVertices","vertexDegree"])
    var[:] = stress11varTri[:]

    var = fileOut.createVariable("strain22varTri","d",dimensions=["nVertices","vertexDegree"])
    var[:] = stress22varTri[:]

    var = fileOut.createVariable("strain12varTri","d",dimensions=["nVertices","vertexDegree"])
    var[:] = stress12varTri[:]

    var = fileOut.createVariable("stress11var","d",dimensions=["nCells","maxEdges"])
    var[:] = stress11var[:]

    var = fileOut.createVariable("stress22var","d",dimensions=["nCells","maxEdges"])
    var[:] = stress22var[:]

    var = fileOut.createVariable("stress12var","d",dimensions=["nCells","maxEdges"])
    var[:] = stress12var[:]

    var = fileOut.createVariable("stress11varTri","d",dimensions=["nVertices","vertexDegree"])
    var[:] = stress11varTri[:]

    var = fileOut.createVariable("stress22varTri","d",dimensions=["nVertices","vertexDegree"])
    var[:] = stress22varTri[:]

    var = fileOut.createVariable("stress12varTri","d",dimensions=["nVertices","vertexDegree"])
    var[:] = stress12varTri[:]

    #if (config_use_c_grid == 'true') :
    #   #if C-grid
    #   var = fileOut.createVariable("stressDivergenceUAnalytical","d",dimensions=["nEdges"])
    #   var[:] = stressDivergenceUCGrid[:]
 
    #   var = fileOut.createVariable("stressDivergenceVAnalytical","d",dimensions=["nEdges"])
    #   var[:] = stressDivergenceVCGrid[:]

    #else :
    var = fileOut.createVariable("stressDivergenceUAnalytical","d",dimensions=["nVertices"])
    var[:] = stressDivergenceU[:]
 
    var = fileOut.createVariable("stressDivergenceVAnalytical","d",dimensions=["nVertices"])
    var[:] = stressDivergenceV[:]

    fileOut.close()

#-------------------------------------------------------------

def create_ics():

    gridTypes = ["hex","quad"]
#    gridTypes = ["hex"]

#    grids = {"hex": ["0082x0094",
#                     "0164x0188",
#                     "0328x0376",
#                     "0656x0752"],
#             "quad":["0080x0080",
#                     "0160x0160",
#                     "0320x0320",
#                     "0640x0640"]}

    grids = {"hex": ["0082x0094",
                     "0164x0188",
                     "0328x0376"],
             "quad":["0080x0080"]}

#    grids = {"hex": ["0082x0094"],
#             "quad":["0080x0080"]}

#    grids = {"hex": ["0082x0094",
#                     "0164x0188"]}

    for gridType in gridTypes:
        for grid in grids[gridType]:

            gridfile = "grid_%s_%s.nc" %(gridType,grid)
            icfile = "ic_%s_%s.nc" %(gridType,grid)

            create_ic(gridfile, icfile)

#-------------------------------------------------------------------------------

if __name__ == "__main__":

    create_ics()

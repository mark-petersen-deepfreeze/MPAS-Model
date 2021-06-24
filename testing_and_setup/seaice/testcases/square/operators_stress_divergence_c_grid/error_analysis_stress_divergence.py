from netCDF4 import Dataset
import numpy as np

#-----------------------------------------------------------------------

def error_analysis_stress_divergence():

    # taylor series terms:
    # 0: f(x0,y0)
    # 1: df/dx(x0,y0) \Delta x
    # 2: df/dy(x0,y0) \Delta y
    # 3: 1/2 d2f/dx2(x0,y0) \Delta x^2
    # 4: 1/2 d2f/dy2(x0,y0) \Delta y^2
    # 5: d2f/dxdy(x0,y0) \Delta x \Delta y
    nTerms = 6
    dxPow = [0,1,0,2,1,0]
    dyPow = [0,0,1,0,1,2]
    coeff = [1.0,1.0,1.0,0.5,1.0,0.5]
    terms = ["f","df/dx","df/dy","d2f/dx2","d2f/dxdy","d2f/dy2"]


    fileout = open("error_report_stress_divergence.txt","w")


    #gridTypes = ["hex","quad"]
    #grids = {"hex" :"0082x0094",
    #         "quad":"0080x0080"}
    gridTypes = ["hex"]
    grids = {"hex" :"0082x0094"}
    #grids = {"hex" :"0164x0188"}
    for gridType in gridTypes:

        print("GridType: ", gridType)
        fileout.write("GridType: %s" %(gridType))


        # grid file
        filegrid = Dataset("grid_%s_%s.nc" %(gridType,grids[gridType]),"r")

        nVertices = len(filegrid.dimensions["nVertices"])
        vertexDegree = len(filegrid.dimensions["vertexDegree"])

        nEdgesOnCell = filegrid.variables["nEdgesOnCell"][:]
        cellsOnVertex = filegrid.variables["cellsOnVertex"][:]
        cellsOnVertex[:] = cellsOnVertex[:] - 1
        cellsOnEdge = filegrid.variables["cellsOnEdge"][:]
        cellsOnEdge[:] = cellsOnEdge[:] - 1
        verticesOnCell = filegrid.variables["verticesOnCell"][:]
        verticesOnCell[:] = verticesOnCell[:] - 1
        verticesOnEdge = filegrid.variables["verticesOnEdge"][:]
        verticesOnEdge[:] = verticesOnEdge[:] - 1
        edgesOnVertex = filegrid.variables["edgesOnVertex"][:]
        edgesOnVertex[:] = edgesOnVertex[:] - 1
        edgesOnCell = filegrid.variables["edgesOnCell"][:]
        edgesOnCell[:] = edgesOnCell[:] - 1

        xVertex = filegrid.variables["xVertex"][:]
        yVertex = filegrid.variables["yVertex"][:]
        xCell = filegrid.variables["xCell"][:]
        yCell = filegrid.variables["yCell"][:]
        xEdge = filegrid.variables["xEdge"][:]
        yEdge = filegrid.variables["yEdge"][:]
        dcEdges = filegrid.variables["dcEdge"][:]
        dvEdge = filegrid.variables["dvEdge"][:]
        areaCell = filegrid.variables["areaCell"][:]
        areaTriangle = filegrid.variables["areaTriangle"][:]

        filegrid.close()


        # variational
        print("  Variational:")
        fileout.write("  Variational:\n")

        iVertexTest = 1123
        print("  iVertexTest: ",iVertexTest)
        fileout.write("  iVertexTest: %i\n" %(iVertexTest))
        for iCellOnVertex in range(0, vertexDegree):
            print("  iCell: ", iCellOnVertex, cellsOnVertex[iVertexTest,iCellOnVertex])
            fileout.write("  iCell: %i %i\n" %(iCellOnVertex, cellsOnVertex[iVertexTest,iCellOnVertex]))

        print(edgesOnVertex[iVertexTest,0])
        print(edgesOnVertex[iVertexTest,1])
        print(edgesOnVertex[iVertexTest,2])

        iEdgeTest = 1554 #it is one of the edgesOnVertex of vertex 1123 (with both enumerations starting from 0)
        #iEdgeTest = 1436  #it is one of the edgesOnVertex of vertex 1123 (with both enumerations starting from 0) USE THIS WITH FINER MESH
        print("  iEdgeTest: ",iEdgeTest)
        fileout.write("  iEdgeTest: %i\n" %(iEdgeTest))
        for iCellOnEdge in range(0, 2):
            print("  iCell: ", iCellOnEdge, cellsOnEdge[iEdgeTest,iCellOnEdge])
            fileout.write("  iCell: %i %i\n" %(iCellOnEdge, cellsOnEdge[iEdgeTest,iCellOnEdge]))

        print("xVertex[iVertexTest]=",xVertex[iVertexTest],"yVertex[iVertexTest]=",yVertex[iVertexTest])
        print("xEdge[iEdgeTest]=",xEdge[iEdgeTest],"yEdge[iEdgeTest]=",yEdge[iEdgeTest])

        #basises = ["wachspress","pwl"]
        basises = ["wachspress"]
        for basis in basises:

            print("   ",basis)
            fileout.write("   %s\n" %(basis))

            # variational
            filein = Dataset("./output_%s_%s_%s/output.2000.nc" %(gridType,basis,grids[gridType]),"r")

            basisIntegralsU = filein.variables["basisIntegralsU"][:]
            basisIntegralsV = filein.variables["basisIntegralsV"][:]
            basisIntegralsMetric = filein.variables["basisIntegralsMetric"][:]
            basisIntegralsUTri = filein.variables["basisIntegralsUTri"][:]
            basisIntegralsVTri = filein.variables["basisIntegralsVTri"][:]
            basisIntegralsMetricTri = filein.variables["basisIntegralsMetricTri"][:]
            cellEdgesAtEdge = filein.variables["cellEdgesAtEdge"][:]
            cellEdgesAtEdge[:] = cellEdgesAtEdge[:] - 1
            triangleEdgesAtEdge = filein.variables["triangleEdgesAtEdge"][:]
            triangleEdgesAtEdge[:] = triangleEdgesAtEdge[:] - 1
            varDenCGrid = filein.variables["variationalDenominatorCGrid"][:]
            areaCellCGrid = filein.variables["areaCellCGrid"][:]
            areaTriCGrid = filein.variables["areaTriCGrid"][:]

            filein.close()

            # gradient taylor terms
            dfdx = np.zeros(nTerms) # ideal: (0,1,0,0,0,0)
            dfdy = np.zeros(nTerms) # ideal: (0,0,1,0,0,0)

            for iTerm in range(0,nTerms):

                # loop over surrounding cells
                for iSurroundingCell in range(0,2):

                    # get the cell number of this cell
                    iCell = cellsOnEdge[iEdgeTest,iSurroundingCell]

                    # get the local index of iEdge on iSurroundingCell
                    iVelocityEdge = cellEdgesAtEdge[iEdgeTest,iSurroundingCell]

                    # loop over the vertices of the surrounding cell
                    for iStressEdge in range(0,nEdgesOnCell[iCell]):

                        iEdge = edgesOnCell[iCell,iStressEdge]

                        dx = xEdge[iEdge] - xEdge[iEdgeTest]
                        dy = yEdge[iEdge] - yEdge[iEdgeTest]

                        fTerm = coeff[iTerm] * pow(dx,dxPow[iTerm]) * pow(dy,dyPow[iTerm])

                        if iTerm == 2:
                           #print("before=",dfdy[iTerm])
                           print("iCell=",iCell,"iStressEdge=",iStressEdge)
                           #print("cell",dfdy[iTerm], fTerm, basisIntegralsV[iCell,iVelocityEdge,iStressEdge], -fTerm * basisIntegralsV[iCell,iVelocityEdge,iStressEdge]/ varDenCGrid[iEdgeTest])
                        dfdx[iTerm] -= (fTerm * basisIntegralsU[iCell,iVelocityEdge,iStressEdge]) / varDenCGrid[iEdgeTest]
                        dfdy[iTerm] -= (fTerm * basisIntegralsV[iCell,iVelocityEdge,iStressEdge]) / varDenCGrid[iEdgeTest]
                        if iTerm == 2:
                           #print("after=",dfdy[iTerm])
                           print("after=",(-fTerm * basisIntegralsV[iCell,iVelocityEdge,iStressEdge]) / varDenCGrid[iEdgeTest])

                    #CHECK
                    areaCheck = 0.0
                    for i in range(0,nEdgesOnCell[iCell]):
                       for j in range(0,nEdgesOnCell[iCell]):
                          areaCheck += basisIntegralsMetric[iCell,i,j]
                    #if iTerm == 0:
                       #print("cell=", iCell, "areaCell=", areaCell[iCell], "areaCheck=", areaCheck, "areaCellCGrid=", areaCellCGrid[iCell], "diff=", abs(areaCheck-areaCellCGrid[iCell]))
           

                # loop over surrounding triangles
                for iSurroundingTri in range(0,2):

                    # get the vertex number of this triangle
                    iVertex = verticesOnEdge[iEdgeTest,iSurroundingTri]

                    # get the local index of iEdge on iSurroundingTri
                    iVelocityEdge = triangleEdgesAtEdge[iEdgeTest,iSurroundingTri]

                    # loop over the edges of the surrounding triangle
                    for iStressEdge in range(0,vertexDegree):

                        iEdge = edgesOnVertex[iVertex,iStressEdge]

                        dx = xEdge[iEdge] - xEdge[iEdgeTest]
                        dy = yEdge[iEdge] - yEdge[iEdgeTest]

                        fTerm = coeff[iTerm] * pow(dx,dxPow[iTerm]) * pow(dy,dyPow[iTerm])

                        if iTerm == 2:
                           #print("before=",dfdy[iTerm])
                           print("iTriangle=",iVertex,"iStressEdge=",iStressEdge)                         
                           #print("triangle", dfdy[iTerm], fTerm, basisIntegralsVTri[iVertex,iVelocityEdge,iStressEdge], -fTerm * basisIntegralsVTri[iVertex,iVelocityEdge,iStressEdge]/ varDenCGrid[iEdgeTest])
                        dfdx[iTerm] -= (fTerm * basisIntegralsUTri[iVertex,iVelocityEdge,iStressEdge]) / varDenCGrid[iEdgeTest]
                        dfdy[iTerm] -= (fTerm * basisIntegralsVTri[iVertex,iVelocityEdge,iStressEdge]) / varDenCGrid[iEdgeTest]
                        if iTerm == 2:
                           #print("after=",dfdy[iTerm])
                           print("after=",(-fTerm * basisIntegralsVTri[iVertex,iVelocityEdge,iStressEdge]) / varDenCGrid[iEdgeTest])

                    #CHECK
                    areaCheck = 0.0
                    for i in range(0,vertexDegree):
                       for j in range(0,vertexDegree):
                          areaCheck += basisIntegralsMetricTri[iVertex,i,j]
                    #if iTerm == 0:
                    #   print("triangle=", iVertex, "areaCheck=", areaCheck, "areaTriCGrid=", areaTriCGrid[iVertex], "diff=", abs(areaCheck-areaTriCGrid[iVertex]))

            dfdxAvg = np.mean(dfdx,axis=0)
            dfdyAvg = np.mean(dfdy,axis=0)
            print("          # Term       df/dx                  df/dy")
            print("          ---------------------------------------------------------")
            fileout.write("          # Term       df/dx                  df/dy\n")
            fileout.write("          ---------------------------------------------------------\n")
            for iTerm in range(0,nTerms):
                print("      %5i %-8s: % 20.15e % 20.15e" %(iTerm, terms[iTerm], dfdx[iTerm], dfdy[iTerm]))
                fileout.write("      %5i %-8s: % 20.15e % 20.15e\n" %(iTerm, terms[iTerm], dfdx[iTerm], dfdy[iTerm]))


    fileout.close()

#-------------------------------------------------------------------------------

if __name__ == "__main__":

    error_analysis_stress_divergence()

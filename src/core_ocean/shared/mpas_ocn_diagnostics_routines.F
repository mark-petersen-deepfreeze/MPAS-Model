! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_diagnostics_routines
!
!> \brief MPAS ocean diagnostics driver
!> \author Mark Petersen
!> \date   November 2013
!> \details
!>  This module contains the routines for computing individual diagnostic variables
!
!-----------------------------------------------------------------------

module ocn_diagnostics_routines

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timer

   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public parameters
   !
   !--------------------------------------------------------------------

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_relativeVorticity_circulation, &
             computeLatLonEdgeMid, &
             xyzEdgeMid, & 
             compute_weightsOnEdge_baricentric, &
             weightsHEDGE, &
             weightsPV, &
             sphericalTriangleArea, &
             sphere_distance

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_relativeVorticity_circulation
!
!> \brief   Computes relative vorticity and circulation
!> \author  Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date    November 2013
!> \details
!>  Computes relative vorticity and circulation
!
!-----------------------------------------------------------------------

   subroutine ocn_relativeVorticity_circulation(relativeVorticity, circulation, meshPool, normalVelocity, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         meshPool

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalVelocity

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(out) :: &
         relativeVorticity

      real (kind=RKIND), dimension(:,:), intent(out) :: &
         circulation

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer :: iVertex, iEdge, i, k
      integer, pointer :: nEdges, nVertices, vertexDegree
      integer, dimension(:), pointer :: maxLevelVertexBot
      integer, dimension(:,:), pointer :: edgesOnVertex, edgeSignOnVertex

      real (kind=RKIND) :: invAreaTri1, r_tmp
      real (kind=RKIND), dimension(:), pointer :: &
              dcEdge, areaTriangle

      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
      call mpas_pool_get_dimension(meshPool, 'nVertices', nVertices)
      call mpas_pool_get_dimension(meshPool, 'vertexDegree', vertexDegree)

      call mpas_pool_get_array(meshPool, 'maxLevelVertexBot', maxLevelVertexBot)
      call mpas_pool_get_array(meshPool, 'edgesOnVertex', edgesOnVertex)
      call mpas_pool_get_array(meshPool, 'edgeSignOnVertex', edgeSignOnVertex)

      call mpas_pool_get_array(meshPool, 'areaTriangle', areaTriangle)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)

      err = 0

      !$omp parallel
      !$omp do schedule(runtime) private(invAreaTri1, i, iEdge, k, r_tmp)
      do iVertex = 1, nVertices
         circulation(:, iVertex) = 0.0_RKIND
         relativeVorticity(:, iVertex) = 0.0_RKIND
         invAreaTri1 = 1.0_RKIND / areaTriangle(iVertex)
         do i = 1, vertexDegree
            iEdge = edgesOnVertex(i, iVertex)
            do k = 1, maxLevelVertexBot(iVertex)
              r_tmp = dcEdge(iEdge) * normalVelocity(k, iEdge)

              circulation(k, iVertex) = circulation(k, iVertex) + edgeSignOnVertex(i, iVertex) * r_tmp
              relativeVorticity(k, iVertex) = relativeVorticity(k, iVertex) + edgeSignOnVertex(i, iVertex) * r_tmp * invAreaTri1
            end do
         end do
      end do
      !$omp end do
      !$omp end parallel

   !--------------------------------------------------------------------

   end subroutine ocn_relativeVorticity_circulation!}}}

   subroutine computeLatLonEdgeMid(meshPool)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! computeLatLonEdgeMid computes the latitudine and longitude of the midpoint
      ! of a Voronoi edge
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (mpas_pool_type), intent(inout) :: meshPool
      integer, pointer :: nEdges
      integer :: iEdge, vertex1, vertex2
      real (kind=RKIND) :: laVertex1, loVertex1, laVertex2, loVertex2, deltaLon, Bx, By
      real (kind=RKIND), dimension(:), pointer :: latVertex, lonVertex
      real (kind=RKIND), dimension(:), pointer :: latEdgeMid, lonEdgeMid
      integer, dimension(:,:), pointer :: verticesOnEdge

      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
      call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)
      call mpas_pool_get_array(meshPool, 'latVertex', latVertex)
      call mpas_pool_get_array(meshPool, 'lonVertex', lonVertex)
      call mpas_pool_get_array(meshPool, 'latEdgeMid', latEdgeMid)
      call mpas_pool_get_array(meshPool, 'lonEdgeMid', lonEdgeMid)

      latEdgeMid(:) = 0.0
      lonEdgeMid(:) = 0.0

      do iEdge = 1, nEdges

         vertex1 = verticesOnEdge(1,iEdge)
         vertex2 = verticesOnEdge(2,iEdge)
         laVertex1 = latVertex(vertex1)
         loVertex1 = lonVertex(vertex1)
         laVertex2 = latVertex(vertex2)
         loVertex2 = lonVertex(vertex2)
         deltaLon = loVertex2 - loVertex1
         Bx = cos(laVertex2) * cos(deltaLon)
         By = cos(laVertex2) * sin(deltaLon)
         latEdgeMid(iEdge) = atan2(sin(laVertex1) + sin(laVertex2), sqrt((cos(laVertex1) + Bx) * (cos(laVertex1) + Bx) + By * By))
         lonEdgeMid(iEdge) = loVertex1 + atan2(By, cos(laVertex1) + Bx)

      end do

   end subroutine computeLatLonEdgeMid

   subroutine xyzEdgeMid(meshPool)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Computation of xEdgeMid, yEdgeMid and zEdgeMid, i.e. computation of the 
      ! x-coordinate, y-coordinate and z-coordinate of the midpoint of a Voronoi
      ! edge 
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    

      implicit none

      type (mpas_pool_type), intent(inout) :: meshPool
      integer, pointer :: nEdges
      integer :: iEdge, vertex1, vertex2
      real (kind=RKIND) :: xVertex1, yVertex1, zVertex1, xVertex2, yVertex2, zVertex2, length, radius
      real (kind=RKIND), dimension(:), pointer :: xVertex, yVertex, zVertex
      real (kind=RKIND), dimension(:), pointer :: xEdgeMid, yEdgeMid, zEdgeMid, xEdgeMidPlanar, yEdgeMidPlanar, zEdgeMidPlanar
      integer, dimension(:,:), pointer :: verticesOnEdge

      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
      call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)
      call mpas_pool_get_array(meshPool, 'xVertex', xVertex)
      call mpas_pool_get_array(meshPool, 'yVertex', yVertex)
      call mpas_pool_get_array(meshPool, 'zVertex', zVertex)
      call mpas_pool_get_array(meshPool, 'xEdgeMidPlanar', xEdgeMidPlanar)
      call mpas_pool_get_array(meshPool, 'yEdgeMidPlanar', yEdgeMidPlanar)
      call mpas_pool_get_array(meshPool, 'zEdgeMidPlanar', zEdgeMidPlanar)
      call mpas_pool_get_array(meshPool, 'xEdgeMid', xEdgeMid)
      call mpas_pool_get_array(meshPool, 'yEdgeMid', yEdgeMid)
      call mpas_pool_get_array(meshPool, 'zEdgeMid', zEdgeMid)

      xEdgeMid(:) = 0.0
      yEdgeMid(:) = 0.0
      zEdgeMid(:) = 0.0
      xEdgeMidPlanar(:) = 0.0
      yEdgeMidPlanar(:) = 0.0
      zEdgeMidPlanar(:) = 0.0
      radius = 6371220.0 

      do iEdge = 1, nEdges

         vertex1 = verticesOnEdge(1,iEdge)
         vertex2 = verticesOnEdge(2,iEdge)
         xVertex1 = xVertex(vertex1)
         yVertex1 = yVertex(vertex1)
         zVertex1 = zVertex(vertex1)
         xVertex2 = xVertex(vertex2)
         yVertex2 = yVertex(vertex2)
         zVertex2 = zVertex(vertex2)
         xEdgeMidPlanar(iEdge) = 0.5*(xVertex1 + xVertex2)
         yEdgeMidPlanar(iEdge) = 0.5*(yVertex1 + yVertex2)
         zEdgeMidPlanar(iEdge) = 0.5*(zVertex1 + zVertex2)
         length = sqrt(xEdgeMidPlanar(iEdge)**2.0 + yEdgeMidPlanar(iEdge)**2.0 + zEdgeMidPlanar(iEdge)**2.0)
         xEdgeMid(iEdge) = (xEdgeMidPlanar(iEdge)/length)*radius
         yEdgeMid(iEdge) = (yEdgeMidPlanar(iEdge)/length)*radius
         zEdgeMid(iEdge) = (zEdgeMidPlanar(iEdge)/length)*radius
         if (planarMesh) then
           xEdgeMid(iEdge) = xEdgeMidPlanar(iEdge)
           yEdgeMid(iEdge) = yEdgeMidPlanar(iEdge)
           zEdgeMid(iEdge) = zEdgeMidPlanar(iEdge)   
         end if 

      end do

   end subroutine xyzEdgeMid

   subroutine compute_weightsOnEdge_baricentric(meshPool)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Computing the weights for the Coriolis term in the equation of u using eq.
      ! (37) in Peixoto. Assumptions made: 
      ! 1. t_e = (x_e - x_v)/||x_e - x_v||
      ! 2. we assume that n_e goes from the center of cellsOnEdge(1, iEdge) to
      ! cellsOnEdge(2, iEdge)  
      !         
      !
      ! Reference: Pedro S. Peixoto, "Accuracy analysis of mimetic finite
      !            volume operators on geodesic grids and a consistent alternative" 
      !            J. of Comp. Phys., 310, pp. 127--160
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (mpas_pool_type), intent(inout) :: meshPool

      integer :: iEdge, jEdge, i, j, ii, jj, v2, sharedCell, v2Edge, jEdgeIsNextEdge, nextEdge, nextEdgeOld, iVertex, n, indexEdge
      integer, pointer :: nEdges, vertexDegree

      integer, dimension(2) :: iEdgeCells, jEdgeCells, jEdgeVertices, nextEdgeCells

      integer, dimension(:), pointer :: nEdgesOnCell, nEdgesOnEdge, nEdgesArray
      integer, dimension(:,:), pointer :: verticesOnEdge, edgesOnCell, cellsOnEdge, edgesOnEdge, edgesOnVertex, edgeSignOnCell

      real (kind=RKIND) :: xTangent, yTangent, zTangent, normTangent, zCross, xCross, yCross, dotProd, ne_x, ne_y, ne_z
      real (kind=RKIND), dimension(:), pointer :: xEdgeMid, yEdgeMid, zEdgeMid, xEdge, yEdge, zEdge
      real (kind=RKIND), dimension(:), pointer :: xVertex, yVertex, zVertex
      real (kind=RKIND), dimension(:), pointer :: dvEdge, dcEdge, areaCell, xCell, yCell, zCell

      real (kind=RKIND), dimension(:,:), pointer :: weightsOnEdge, weightsOnEdge_baricentric

      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_dimension(meshPool, 'vertexDegree', vertexDegree)

      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgesOnEdge', edgesOnEdge)
      call mpas_pool_get_array(meshPool, 'nEdgesOnEdge', nEdgesOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnVertex', edgesOnVertex)
      call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)
      call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)
      call mpas_pool_get_array(meshPool, 'xCell', xCell)
      call mpas_pool_get_array(meshPool, 'yCell', yCell)
      call mpas_pool_get_array(meshPool, 'zCell', zCell)

      call mpas_pool_get_array(meshPool, 'xEdgeMid', xEdgeMid)
      call mpas_pool_get_array(meshPool, 'yEdgeMid', yEdgeMid)
      call mpas_pool_get_array(meshPool, 'zEdgeMid', zEdgeMid)
      call mpas_pool_get_array(meshPool, 'xEdge', xEdge)
      call mpas_pool_get_array(meshPool, 'yEdge', yEdge)
      call mpas_pool_get_array(meshPool, 'zEdge', zEdge)
      call mpas_pool_get_array(meshPool, 'xVertex', xVertex)
      call mpas_pool_get_array(meshPool, 'yVertex', yVertex)
      call mpas_pool_get_array(meshPool, 'zVertex', zVertex)

      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'weightsOnEdge_baricentric', weightsOnEdge_baricentric)
      call mpas_pool_get_array(meshPool, 'weightsOnEdge', weightsOnEdge)

      weightsOnEdge_baricentric(:,:) = 0.0
      !print*, 'nEdgesArray ', nEdgesArray
      !print*, 'nEdges ', nEdges  
      !do iEdge = 1, nEdges
      do iEdge = 1, nEdgesArray(3)
         iEdgeCells(1) = cellsOnEdge(1,iEdge)
         iEdgeCells(2) = cellsOnEdge(2,iEdge)
         !--- begin t_e Darren ---
         !do j = 1, nEdgesOnCell(iEdgeCells(1))
         !   if (iEdge == edgesOnCell(j,iEdgeCells(1))) then
         !      indexEdge = j
         !   end if
         !end do
         ! ---- BEGIN: compute n
         if (iEdgeCells(1) > iEdgeCells(2)) then
            n = - 1.0
         elseif (iEdgeCells(2) > iEdgeCells(1)) then
            n = 1.0
         end if
         ! ---- END: compute n
         ne_x = (xCell(iEdgeCells(1)) - xCell(iEdgeCells(2)))
         ne_y = (yCell(iEdgeCells(1)) - yCell(iEdgeCells(2)))
         ne_z = (zCell(iEdgeCells(1)) - zCell(iEdgeCells(2)))
         xTangent = yEdgeMid(iEdge) * ne_z - zEdgeMid(iEdge) * ne_y
         yTangent = zEdgeMid(iEdge) * ne_x - xEdgeMid(iEdge) * ne_z
         zTangent = xEdgeMid(iEdge) * ne_y - yEdgeMid(iEdge) * ne_x
         normTangent = sqrt(xTangent**2.0 + yTangent**2.0 + zTangent**2.0)
         xTangent = xTangent/normTangent
         yTangent = yTangent/normTangent
         zTangent = zTangent/normTangent
         !print*, xTangent, yTangent, zTangent
         !--- end t_e Darren ---
         do i = 1, nEdgesOnEdge(iEdge)
            jEdge = edgesOnEdge(i,iEdge)
            jEdgeCells(1) = cellsOnEdge(1,jEdge)
            jEdgeCells(2) = cellsOnEdge(2,jEdge)
            do ii = 1,2
               do jj = 1,2
                  if (iEdgeCells(ii)  == jEdgeCells(jj)) then
                     sharedCell = iEdgeCells(ii) !this is to find what is the cell in common (in what cell jEdge is)
                  end if
               end do
            end do
            ! ---- BEGIN: compute n
            if (sharedCell < jEdgeCells(1) .or. sharedCell < jEdgeCells(2)) then
               n = 1.0 !this is n_{e',i} in Peixoto
            else if (sharedCell > jEdgeCells(1) .or. sharedCell > jEdgeCells(2)) then
               n = - 1.0
            end if
            ! --- END: compute n 
            do j = 1, nEdgesOnCell(sharedCell)
               if (jEdge == edgesOnCell(j,sharedCell)) then
                  indexEdge = j
               end if
            end do
            dotProd = edgeSignOnCell(indexEdge, sharedCell)*(xEdgeMid(jEdge) - xCell(sharedCell)) * xTangent + &
                      edgeSignOnCell(indexEdge, sharedCell)*(yEdgeMid(jEdge) - yCell(sharedCell)) * yTangent + &
                      edgeSignOnCell(indexEdge, sharedCell)*(zEdgeMid(jEdge) - zCell(sharedCell)) * zTangent
            !print*, 'dotProd', dotProd
            !print*, 'areaCell(sharedCell) ', areaCell(sharedCell)  
            !print*, 'dvEdge(jEdge)/areaCell(sharedCell)',
            !dvEdge(jEdge)/areaCell(sharedCell) 
            weightsOnEdge_baricentric(i, iEdge) = 0.5 * (dvEdge(jEdge)/areaCell(sharedCell)) * dotProd
            !print*, weightsOnEdge_baricentric(i, iEdge)  
   
            weightsOnEdge(i, iEdge) = weightsOnEdge_baricentric(i, iEdge)
         end do
      end do

   end subroutine compute_weightsOnEdge_baricentric

   subroutine weightsHEDGE(meshPool)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! weightsHEDGE computes the weights needed in the HCm discretization
      ! for the computation of h_edge_baricentric 
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (mpas_pool_type), intent(inout) :: meshPool
      integer :: vertex1, vertex2, cell1, cell2, cell3, cellNew, iEdge, triangleCenter, i, k
      real (kind=RKIND) :: laVertex1, loVertex1, laVertex2, loVertex2, laEdge, loEdge, laEdgeMidVoronoi, loEdgeMidVoronoi, &
                           laCell1, loCell1, laCell2, loCell2, laCell3, loCell3, a, b, radius1
      real (kind=RKIND), dimension(:), pointer :: latVertex, lonVertex, latCell, lonCell
      real (kind=RKIND), dimension(:), pointer :: latEdge, lonEdge, latEdgeMid, lonEdgeMid
      integer, pointer :: nVertLevels, vertexDegree
      integer, dimension(:), pointer :: nEdgesArray
      integer, dimension(:,:), pointer :: cellsOnEdge, cellsOnVertex, verticesOnEdge
      real (kind=RKIND), dimension(:,:), pointer :: areaHEdge
      real (kind=RKIND), dimension(:), pointer :: areaTriangle
      integer, dimension(:), pointer :: cell3HEdge, triangleHEdge

      call mpas_pool_get_array(meshPool, 'latVertex', latVertex)
      call mpas_pool_get_array(meshPool, 'lonVertex', lonVertex)
      call mpas_pool_get_array(meshPool, 'latCell', latCell)
      call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
      call mpas_pool_get_array(meshPool, 'latEdge', latEdge)
      call mpas_pool_get_array(meshPool, 'lonEdge', lonEdge)
      call mpas_pool_get_array(meshPool, 'latEdgeMid', latEdgeMid)
      call mpas_pool_get_array(meshPool, 'lonEdgeMid', lonEdgeMid)
      call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'cellsOnVertex', cellsOnVertex)
      call mpas_pool_get_array(meshPool, 'areaHEdge', areaHEdge)
      call mpas_pool_get_array(meshPool, 'cell3HEdge', cell3HEdge)
      call mpas_pool_get_array(meshPool, 'triangleHEdge', triangleHEdge)
      call mpas_pool_get_array(meshPool, 'areaTriangle', areaTriangle)

      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      call mpas_pool_get_dimension(meshPool, 'vertexDegree', vertexDegree)

      areaHEdge(:,:) = 0.0
      cell3HEdge(:) = 0.0
      triangleHEdge(:) = 0.0

      radius1 = 6371220.0

      if (planarMesh) then 
         do iEdge = 1, nEdgesArray(5)
            cell1 = cellsOnEdge(1,iEdge)
            cell2 = cellsOnEdge(2,iEdge)
            a = planar_distance()
            b = planar_distance()
         end do 
      else
         do iEdge = 1, nEdgesArray(5)
            cell1 = cellsOnEdge(1,iEdge)
            cell2 = cellsOnEdge(2,iEdge) 
            vertex1 = verticesOnEdge(1,iEdge)
            vertex2 = verticesOnEdge(2,iEdge)
            laVertex1 = latVertex(vertex1)
            loVertex1 = lonVertex(vertex1)
            laVertex2 = latVertex(vertex2)
            loVertex2 = lonVertex(vertex2)
            laEdge = latEdge(iEdge)
            loEdge = lonEdge(iEdge)
            a = sphere_distance(laVertex1, loVertex1, laEdge, loEdge, radius1)
            b = sphere_distance(laVertex2, loVertex2, laEdge, loEdge, radius1)
            !print*, 'a ', a, 'b ', b, 'abs(a-b) ', abs(a-b)
            if (latCell(cell2)==0.0 .AND. lonCell(cell2)==0.0) then
               cell3HEdge(iEdge) = 0
               !print*, iEdge
            elseif (abs(a-b)<1.E-2) then
            !if (abs(a-b)<1.E-9) then
               areaHEdge(1,iEdge) = 0.5
               areaHEdge(2,iEdge) = 0.5
               areaHEdge(3,iEdge) = 0.0
               !print*, 'mid triangle = mid voronoi' 
            else
               if (a<b) then
                  triangleCenter = vertex2
               else
                  triangleCenter = vertex1
               end if
               do i = 1, vertexDegree
                  cellNew = cellsOnVertex(i, triangleCenter)
                  if ((cell1.NE.cellNew) .AND. (cell2.NE.cellNew)) then
                     cell3 = cellNew
                     cell3HEdge(iEdge) = cell3
                  end if
               end do
               if (latCell(cell3)==0.0 .AND. lonCell(cell3)==0.0) then 
                  cell3HEdge(iEdge) = 0
               else 
                  !print*, 'cell1, cell2, cell3', cell1, cell2, cell3 
                  triangleHEdge(iEdge) = triangleCenter
                  laEdgeMidVoronoi = latEdgeMid(iEdge)
                  loEdgeMidVoronoi = lonEdgeMid(iEdge)
                  !print*, 'print mid point voronoi and triangle',
                  !laEdgeMidVoronoi, laEdge, loEdgeMidVoronoi, loEdge
                  !print*, 'difference mid point voronoi e triangle',
                  !abs(laEdgeMidVoronoi-laEdge), abs(loEdgeMidVoronoi-loEdge) 
                  laCell1 = latCell(cell1)
                  loCell1 = lonCell(cell1)
                  laCell2 = latCell(cell2)
                  loCell2 = lonCell(cell2)
                  laCell3 = latCell(cell3)
                  loCell3 = lonCell(cell3)
                  areaHEdge(1,iEdge) = sphericalTriangleArea(laCell2, loCell2, laCell3, loCell3, laEdgeMidVoronoi, loEdgeMidVoronoi) 
                  areaHEdge(2,iEdge) = sphericalTriangleArea(laCell1, loCell1, laCell3, loCell3, laEdgeMidVoronoi, loEdgeMidVoronoi) 
                  areaHEdge(3,iEdge) = sphericalTriangleArea(laCell1, loCell1, laCell2, loCell2, laEdgeMidVoronoi, loEdgeMidVoronoi) 
               end if 
            end if
            !print*, iEdge, 'v1 ', vertex1, 'v2 ', vertex2 
            !print*, 'cell2 ', cell2
            !print*, '---------------'
         end do
      end if 
      print*, 'sono in weightsHEDGE '

   end subroutine weightsHEDGE

   subroutine weightsPV(meshPool)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! weightsPV computes the weights needed in the HCm discretization
      ! for the computation of h_vertex_baricentric 
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (mpas_pool_type), intent(inout) :: meshPool
      integer :: iVertex, cell1, cell2, cell3
      real (kind=RKIND) :: radius1
      real (kind=RKIND) :: laVertex1, loVertex1, laCell1, loCell1, laCell2, loCell2, laCell3, loCell3
      real (kind=RKIND), dimension(:), pointer :: latVertex, lonVertex
      real (kind=RKIND), dimension(:), pointer :: latCell, lonCell
      integer, dimension(:), pointer :: nVerticesArray
      integer, dimension(:,:), pointer :: cellsOnVertex
      real (kind=RKIND), dimension(:,:), pointer :: areaPV

      call mpas_pool_get_array(meshPool, 'latVertex', latVertex)
      call mpas_pool_get_array(meshPool, 'lonVertex', lonVertex)
      call mpas_pool_get_array(meshPool, 'latCell', latCell)
      call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
      call mpas_pool_get_dimension(meshPool, 'nVerticesArray', nVerticesArray)
      call mpas_pool_get_array(meshPool, 'cellsOnVertex', cellsOnVertex)
      call mpas_pool_get_array(meshPool, 'areaPV', areaPV)

      areaPV(:,:) = 0.0

      do iVertex = 1, nVerticesArray(3)
         cell1 = cellsOnVertex(1,iVertex)
         cell2 = cellsOnVertex(2,iVertex)
         cell3 = cellsOnVertex(3,iVertex)
         laCell1 = latCell(cell1)
         loCell1 = lonCell(cell1)
         laCell2 = latCell(cell2)
         loCell2 = lonCell(cell2)
         laCell3 = latCell(cell3)
         loCell3 = lonCell(cell3)
         laVertex1 = latVertex(iVertex)
         loVertex1 = lonVertex(iVertex)
         areaPV(1, iVertex) = sphericalTriangleArea(laCell2, loCell2, laCell3, loCell3, laVertex1, loVertex1)
         areaPV(2, iVertex) = sphericalTriangleArea(laCell1, loCell1, laCell3, loCell3, laVertex1, loVertex1)
         areaPV(3, iVertex) = sphericalTriangleArea(laCell1, loCell1, laCell2, loCell2, laVertex1, loVertex1)
         !areaPV(iVertex, cell1) = sphericalTriangleArea(laCell2, loCell2, laCell3, loCell3, laVertex1, loVertex1)
         !areaPV(iVertex, cell2) = sphericalTriangleArea(laCell1, loCell1, laCell3, loCell3, laVertex1, loVertex1)
         !areaPV(iVertex, cell3) = sphericalTriangleArea(laCell1, loCell1, laCell2, loCell2, laVertex1, loVertex1)
         !area3 = areaTriangle(iVertex) - area1 - area2
      end do
      print*, 'sono in weightsPV '

   end subroutine weightsPV

   real function sphericalTriangleArea(laEdge, loEdge, laVertex, loVertex, laCell, loCell)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! sphericalTriangleArea uses the spherical analog of Heron's formula to
      ! compute the area of a triangle on the surface of a sphere
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      real (kind=RKIND), intent(in) :: laEdge, loEdge, laVertex, loVertex, laCell, loCell
      real (kind=RKIND) :: tanqe, s, a, b, c, radius, one
      radius = 6371220.0
      one = 1.0_RKIND

      a = sphere_distance(laCell, loCell, laVertex, loVertex, one)
      b = sphere_distance(laCell, loCell, laEdge, loEdge, one)
      c = sphere_distance(laEdge, loEdge, laVertex, loVertex, one)
      s = 0.5_RKIND*(a+b+c)
      
      tanqe = sqrt(tan(0.5_RKIND*s)*tan(0.5_RKIND*(s-a))*tan(0.5_RKIND*(s-b))*tan(0.5_RKIND*(s-c)))
      sphericalTriangleArea = 4.*atan(tanqe)*radius**2

      !print*, 'printing a, b, c, s e tanqe', a, b, c, s, tanqe

   end function sphericalTriangleArea

   real function sphere_distance(lat1, lon1, lat2, lon2, radius)
   
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Compute the great-circle distance between (lat1, lon1) and (lat2, lon2) on
      ! a sphere with given radius.
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      real (kind=RKIND), intent(in) :: lat1, lon1, lat2, lon2, radius

      real (kind=RKIND) :: arg1

      arg1 = sqrt( sin(0.5_RKIND*(lat2-lat1))**2 +  &
                   cos(lat1)*cos(lat2)*sin(0.5_RKIND*(lon2-lon1))**2 )
      sphere_distance = 2.*radius*asin(arg1)

   end function sphere_distance

   real function planarTriangleArea(x1, x2, x3, y1, y2, y3, z1, z2, z3)

     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     ! planarTriangleArea uses Heron's formula to compute the area of a triangle in a plane
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     implicit none

     real (kind=RKIND), intent(in) :: x1, x2, x3, y1, y2, y3, z1, z2, z3 
     real (kind=RKIND) :: s, a, b, c

     !Get side lengths
     a = planar_distance(x1, x2, y1, y2, z1, z2);
     b = planar_distance(x2, x3, y2, y3, z2, z3);
     c = planar_distance(x1, x3, y1, y3, z1, z3);

     !Semi-perimeter of TRI(ABC)
     s = (a + b + c) * 0.5_RKIND;

     planarTriangleArea = sqrt(s * (s - a) * (s - b) * (s - c));
   
   end function planarTriangleArea

   real function planar_distance(x1, x2, y1, y2, z1, z2)

     implicit none

     real (kind=RKIND), intent(in) :: x1, x2, y1, y2, z1, z2
     real (kind=RKIND) :: a, b, c

     a = x1 - x2
     b = y1 - y2
     c = z1 - z2
     planar_distance = sqrt(a**2+b**2+c**2)

   end function planar_distance

end module ocn_diagnostics_routines

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker

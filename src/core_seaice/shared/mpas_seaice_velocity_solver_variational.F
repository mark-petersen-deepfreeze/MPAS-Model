!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_velocity_solver_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

module seaice_velocity_solver_variational

#include "gpu_macros.inc"

  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timer
  use mpas_log, only: mpas_log_write

  implicit none

  private
  save

  public :: &
       seaice_init_velocity_solver_variational, &
       seaice_strain_tensor_variational, &
       seaice_strain_tensor_variational_c_grid, &
       seaice_average_strains_on_vertex, &
       seaice_stress_tensor_variational, &
       seaice_stress_tensor_variational_c_grid, &
       seaice_stress_divergence_variational, &
       seaice_stress_divergence_variational_c_grid, &
       seaice_final_divergence_shear_variational

contains

!-----------------------------------------------------------------------
! initialization
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_init_velocity_solver_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 24 October 2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_init_velocity_solver_variational(&
       mesh, &
       velocity_variational, &
       boundary, &
       rotateCartesianGrid, &
       includeMetricTerms, &
       variationalBasisType, &
       variationalDenominatorType, &
       integrationType, &
       integrationOrder, &
       useCGrid)!{{{

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    type(MPAS_pool_type), pointer :: &
         velocity_variational, & !< Input/Output:
         boundary                !< Input/Output:

    logical, intent(in) :: &
         rotateCartesianGrid, & !< Input:
         includeMetricTerms, &  !< Input:
         useCGrid               !< Input:

    character(len=*), intent(in) :: &
         variationalBasisType, &       !< Input:
         variationalDenominatorType, & !< Input:
         integrationType               !< Input:

    integer, intent(in) :: &
         integrationOrder !< Input:

    call init_velocity_solver_variational_primary_mesh(&
         mesh, &
         velocity_variational, &
         boundary, &
         rotateCartesianGrid, &
         includeMetricTerms, &
         variationalBasisType, &
         variationalDenominatorType, &
         integrationType, &
         integrationOrder, &
         useCGrid)

  end subroutine seaice_init_velocity_solver_variational

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  init_velocity_solver_variational_primary_mesh
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 24 October 2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine init_velocity_solver_variational_primary_mesh(&
       mesh, &
       velocity_variational, &
       boundary, &
       rotateCartesianGrid, &
       includeMetricTerms, &
       variationalBasisType, &
       variationalDenominatorType, &
       integrationType, &
       integrationOrder, &
       useCGrid)!{{{

    use seaice_mesh, only: &
         seaice_cell_vertices_at_vertex, &
         seaice_cell_edges_at_edge, &
         seaice_triangle_edges_at_edge

    use seaice_velocity_solver_variational_shared, only: &
         seaice_calc_local_coords, &
         seaice_calc_variational_metric_terms

    use seaice_velocity_solver_wachspress, only: &
         seaice_init_velocity_solver_wachspress

    use seaice_velocity_solver_pwl, only: &
         seaice_init_velocity_solver_pwl, &
         seaice_init_velocity_solver_pwl_c_grid

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    type(MPAS_pool_type), pointer :: &
         velocity_variational, & !< Input/Output:
         boundary                !< Input/Output:

    logical, intent(in) :: &
         rotateCartesianGrid, & !< Input:
         includeMetricTerms, &  !< Input:
         useCGrid               !< Input:

    character(len=*), intent(in) :: &
         variationalBasisType, &       !< Input:
         variationalDenominatorType, & !< Input:
         integrationType               !< Input:

    integer, intent(in) :: &
         integrationOrder !< Input:

    integer, dimension(:,:), pointer :: &
         cellVerticesAtVertex, &
         cellEdgesAtEdge, &
         triangleEdgesAtEdge

    real(kind=RKIND), dimension(:), pointer :: &
         tanLatVertexRotatedOverRadius, &
         tanLatEdgeRotatedOverRadius, &
         variationalDenominator, &
         variationalDenominatorCGrid, &
         areaCellCGrid, &
         areaTriCGrid

    real(kind=RKIND), dimension(:,:,:), pointer :: &
         basisGradientU, &
         basisGradientV, &
         basisGradientUTri, &
         basisGradientVTri, &
         basisIntegralsMetric, &
         basisIntegralsU, &
         basisIntegralsV, &
         basisIntegralsMetricTri, &
         basisIntegralsUTri, &
         basisIntegralsVTri

    integer, pointer :: &
         nCells, &
         nVertices, &
         nEdges, &
         vertexDegree, &
         maxEdges

    integer, dimension(:), pointer :: &
         nEdgesOnCell, &
         interiorVertex

    integer, dimension(:,:), pointer :: &
         verticesOnCell, &
         cellsOnVertex, &
         edgesOnCell, &
         cellsOnEdge, &
         edgesOnVertex, &
         verticesOnEdge

    real(kind=RKIND), dimension(:), pointer :: &
         xVertex, &
         yVertex, &
         zVertex, &
         xEdge, &
         yEdge, &
         zEdge, &
         xCell, &
         yCell, &
         zCell, &
         latCell, &
         lonCell, &
         latVertex, &
         lonVertex, &
         latEdge, &
         lonEdge, &
         areaCell, &
         areaTriangle, &
         dvEdge

    logical, pointer :: &
         on_a_sphere

    real(kind=RKIND), pointer :: &
         sphere_radius

    integer, dimension(:), allocatable :: & !needed to use the existing refactored functions
         nEdgesOnVertex

    real(kind=RKIND), dimension(:,:), allocatable :: &
         xLocal, &
         yLocal, &
         xLocalTri, &
         yLocalTri

    call MPAS_pool_get_config(mesh, "on_a_sphere", on_a_sphere)
    call MPAS_pool_get_config(mesh, "sphere_radius", sphere_radius)

    call MPAS_pool_get_dimension(mesh, "nCells", nCells)
    call MPAS_pool_get_dimension(mesh, "nVertices", nVertices)
    call MPAS_pool_get_dimension(mesh, "nEdges", nEdges)
    call MPAS_pool_get_dimension(mesh, "vertexDegree", vertexDegree)
    call MPAS_pool_get_dimension(mesh, "maxEdges", maxEdges)

    call MPAS_pool_get_array(mesh, "nEdgesOnCell", nEdgesOnCell)
    call MPAS_pool_get_array(mesh, "verticesOnCell", verticesOnCell)
    call MPAS_pool_get_array(mesh, "verticesOnEdge", verticesOnEdge)
    call MPAS_pool_get_array(mesh, "cellsOnVertex", cellsOnVertex)
    call MPAS_pool_get_array(mesh, "edgesOnCell", edgesOnCell)
    call MPAS_pool_get_array(mesh, "edgesOnVertex", edgesOnVertex)
    call MPAS_pool_get_array(mesh, "cellsOnEdge", cellsOnEdge)
    call MPAS_pool_get_array(mesh, "xVertex", xVertex)
    call MPAS_pool_get_array(mesh, "yVertex", yVertex)
    call MPAS_pool_get_array(mesh, "zVertex", zVertex)
    call MPAS_pool_get_array(mesh, "xEdge", xEdge)
    call MPAS_pool_get_array(mesh, "yEdge", yEdge)
    call MPAS_pool_get_array(mesh, "zEdge", zEdge)
    call MPAS_pool_get_array(mesh, "xCell", xCell)
    call MPAS_pool_get_array(mesh, "yCell", yCell)
    call MPAS_pool_get_array(mesh, "zCell", zCell)
    call MPAS_pool_get_array(mesh, "latCell", latCell)
    call MPAS_pool_get_array(mesh, "lonCell", lonCell)
    call MPAS_pool_get_array(mesh, "latVertex", latVertex)
    call MPAS_pool_get_array(mesh, "lonVertex", lonVertex)
    call MPAS_pool_get_array(mesh, "latEdge", latEdge)
    call MPAS_pool_get_array(mesh, "lonEdge", lonEdge)
    call MPAS_pool_get_array(mesh, "areaCell", areaCell)
    call MPAS_pool_get_array(mesh, "areaTriangle", areaTriangle)
    call MPAS_pool_get_array(mesh, "dvEdge", dvEdge)

    call MPAS_pool_get_array(boundary, "interiorVertex", interiorVertex)

    call MPAS_pool_get_array(velocity_variational, "cellVerticesAtVertex", cellVerticesAtVertex)
    call MPAS_pool_get_array(velocity_variational, "cellEdgesAtEdge", cellEdgesAtEdge)
    call MPAS_pool_get_array(velocity_variational, "triangleEdgesAtEdge", triangleEdgesAtEdge)
    call MPAS_pool_get_array(velocity_variational, "tanLatVertexRotatedOverRadius", tanLatVertexRotatedOverRadius)
    call MPAS_pool_get_array(velocity_variational, "tanLatEdgeRotatedOverRadius", tanLatEdgeRotatedOverRadius)
    call MPAS_pool_get_array(velocity_variational, "basisGradientU", basisGradientU)
    call MPAS_pool_get_array(velocity_variational, "basisGradientV", basisGradientV)
    call MPAS_pool_get_array(velocity_variational, "basisGradientUTri", basisGradientUTri)
    call MPAS_pool_get_array(velocity_variational, "basisGradientVTri", basisGradientVTri)
    call MPAS_pool_get_array(velocity_variational, "basisIntegralsU", basisIntegralsU)
    call MPAS_pool_get_array(velocity_variational, "basisIntegralsV", basisIntegralsV)
    call MPAS_pool_get_array(velocity_variational, "basisIntegralsMetric", basisIntegralsMetric)
    call MPAS_pool_get_array(velocity_variational, "basisIntegralsUTri", basisIntegralsUTri)
    call MPAS_pool_get_array(velocity_variational, "basisIntegralsVTri", basisIntegralsVTri)
    call MPAS_pool_get_array(velocity_variational, "basisIntegralsMetricTri", basisIntegralsMetricTri)
    call MPAS_pool_get_array(velocity_variational, "variationalDenominator", variationalDenominator)
    call MPAS_pool_get_array(velocity_variational, "variationalDenominatorCGrid", variationalDenominatorCGrid)
    call MPAS_pool_get_array(velocity_variational, "areaCellCGrid", areaCellCGrid)
    call MPAS_pool_get_array(velocity_variational, "areaTriCGrid", areaTriCGrid)

    call mpas_timer_start("variational calc_metric_terms")
    if ( .not. useCGrid ) then
       call seaice_calc_variational_metric_terms(&
            tanLatVertexRotatedOverRadius, &
            nVertices, &
            xVertex, &
            yVertex, &
            zVertex, &
            sphere_radius, &
            rotateCartesianGrid, &
            includeMetricTerms)
    else
           call seaice_calc_variational_metric_terms(&
            tanLatEdgeRotatedOverRadius, &
            nEdges, &
            xEdge, &
            yEdge, &
            zEdge, &
            sphere_radius, &
            rotateCartesianGrid, &
            includeMetricTerms)
    end if
    call mpas_timer_stop("variational calc_metric_terms")

    call mpas_timer_start("variational vertices_at_vertex")
    if ( .not. useCGrid ) then
       call seaice_cell_vertices_at_vertex(&
            cellVerticesAtVertex, &
            nVertices, &
            vertexDegree, &
            nEdgesOnCell, &
            verticesOnCell, &
            cellsOnVertex)
    else
       call seaice_cell_edges_at_edge(&
            cellEdgesAtEdge, &
            nEdges, &
            nEdgesOnCell, &
            edgesOnCell, &
            cellsOnEdge)
       call seaice_triangle_edges_at_edge(&
            triangleEdgesAtEdge, &
            nEdges, &
            vertexDegree, &
            edgesOnVertex, &
            verticesOnEdge)
    end if
    call mpas_timer_stop("variational vertices_at_vertex")

    call mpas_timer_start("variational calc_local_coords")
    allocate(xLocal(maxEdges,nCells))
    allocate(yLocal(maxEdges,nCells))

    allocate(xLocalTri(vertexDegree,nVertices))
    allocate(yLocalTri(vertexDegree,nVertices))

    allocate(nEdgesOnVertex(nVertices))

    nEdgesOnVertex(:) = vertexDegree 

    if ( .not. useCGrid ) then
       call seaice_calc_local_coords(&
            xLocal, &
            yLocal, &
            nCells, &
            nEdgesOnCell, &
            verticesOnCell, &
            xVertex, &
            yVertex, &
            zVertex, &
            xCell, &
            yCell, &
            zCell, &
            rotateCartesianGrid, &
            on_a_sphere)
    else
       call seaice_calc_local_coords(&
            xLocal, &
            yLocal, &
            nCells, &
            nEdgesOnCell, &
            edgesOnCell, &
            xEdge, &
            yEdge, &
            zEdge, &
            xCell, &
            yCell, &
            zCell, &
            rotateCartesianGrid, &
            on_a_sphere)

       call seaice_calc_local_coords(&
            xLocalTri, &
            yLocalTri, &
            nVertices, &
            nEdgesOnVertex, &
            edgesOnVertex, &
            xEdge, &
            yEdge, &
            zEdge, &
            xVertex, &
            yVertex, &
            zVertex, &
            rotateCartesianGrid, &
            on_a_sphere)
    end if

    call mpas_timer_stop("variational calc_local_coords")

    if (useCGrid) then
       call mpas_timer_start("variational denominator_c_grid")
       call variational_denominator_c_grid(&
            on_a_sphere, &
            sphere_radius, &
            cellsOnEdge, &
            edgesOnCell, &
            verticesOnEdge, &
            edgesOnVertex, &
            nEdges, &
            nCells, &
            nVertices, &
            nEdgesOnCell, &
            vertexDegree, &
            xCell, &
            yCell, &
            zCell, &
            xVertex, &
            yVertex, &
            zVertex, &
            xEdge, &
            yEdge, &
            zEdge, &
            latCell, &
            lonCell, &
            latVertex, &
            lonVertex, &
            latEdge, &
            lonEdge, &
            variationalDenominatorCGrid, &
            areaCellCGrid, &
            areaTriCGrid)
       call mpas_timer_stop("variational denominator_c_grid")
    end if

    if (trim(variationalBasisType) == "wachspress") then

       call seaice_init_velocity_solver_wachspress(&
            nCells, &
            maxEdges, &
            nEdgesOnCell, &
            xLocal, &
            yLocal, &
            rotateCartesianGrid, &
            includeMetricTerms, &
            on_a_sphere, &
            integrationType, &
            integrationOrder, &
            sphere_radius, &
            basisGradientU, &
            basisGradientV, &
            basisIntegralsU, &
            basisIntegralsV, &
            basisIntegralsMetric)
       if (useCGrid) then
          call seaice_init_velocity_solver_wachspress(&
              nVertices, &
              vertexDegree, &
              nEdgesOnVertex, &
              xLocalTri, &
              yLocalTri, &
              rotateCartesianGrid, &
              includeMetricTerms, &
              on_a_sphere, &
              integrationType, &
              integrationOrder, &
              sphere_radius, &
              basisGradientUTri, &
              basisGradientVTri, &
              basisIntegralsUTri, &
              basisIntegralsVTri, &
              basisIntegralsMetricTri)
       end if

    else if (trim(variationalBasisType) == "pwl") then

       if ( .not. useCGrid ) then
          call seaice_init_velocity_solver_pwl(&
               nCells, &
               maxEdges, &
               nEdgesOnCell, &
               verticesOnCell, &  !NOTE: this is actually not used
               edgesOnCell, &     !NOTE: this can be avoided if c in the function is computed
               dvEdge, &          !NOTE: this can be avoided if c in the function is computed 
               areaCell, &
               xLocal, &
               yLocal, &
               basisGradientU, &
               basisGradientV, &
               basisIntegralsMetric, &
               basisIntegralsU, &
               basisIntegralsV)
       else
          call seaice_init_velocity_solver_pwl_c_grid(&
               nCells, &
               maxEdges, &
               nEdgesOnCell, &
               areaCellCGrid, &
               xLocal, &
               yLocal, &
               basisGradientU, &
               basisGradientV, &
               basisIntegralsMetric, &
               basisIntegralsU, &
               basisIntegralsV)

          call seaice_init_velocity_solver_pwl_c_grid(&
               nVertices, &
               vertexDegree, &
               nEdgesOnVertex, &
               areaTriCGrid, &
               xLocalTri, &
               yLocalTri, &
               basisGradientUTri, &
               basisGradientVTri, &
               basisIntegralsMetricTri, &
               basisIntegralsUTri, &
               basisIntegralsVTri)

       end if

    else if (trim(variationalBasisType) == "none") then

       continue

    else

       call MPAS_log_write("Unknown variational basis type: "//trim(variationalBasisType), MPAS_LOG_CRIT)

    endif

    if ( .not. useCGrid ) then
       call mpas_timer_start("variational denominator")
       call variational_denominator(&
            nVertices, &
            vertexDegree, &
            nEdgesOnCell, &
            interiorVertex, &
            areaTriangle, &
            cellsOnVertex, &
            cellVerticesAtVertex, &
            basisIntegralsMetric, &
            variationalDenominatorType, &
            variationalDenominator)
       call mpas_timer_stop("variational denominator")
    end if

    ! clean up
    deallocate(xLocal)
    deallocate(yLocal)
    deallocate(xLocalTri)
    deallocate(yLocalTri)
    deallocate(nEdgesOnVertex)

    !call homogenize_variational_basis_field()

  end subroutine init_velocity_solver_variational_primary_mesh

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  variational_denominator
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 30th January 2021
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine variational_denominator(&
       nVertices, &
       vertexDegree, &
       nEdgesOnCell, &
       interiorVertex, &
       areaTriangle, &
       cellsOnVertex, &
       cellVerticesAtVertex, &
       basisIntegralsMetric, &
       variationalDenominatorType, &
       variationalDenominator)

    integer, intent(in) :: &
         nVertices, & !< Input:
         vertexDegree !< Input:

    integer, dimension(:), intent(in) :: &
         nEdgesOnCell, & !< Input:
         interiorVertex  !< Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         areaTriangle !< Input:

    integer, dimension(:,:), intent(in) :: &
         cellsOnVertex !< Input:

    integer, dimension(:,:), intent(in) :: &
         cellVerticesAtVertex !< Input:

    real(kind=RKIND), dimension(:,:,:), intent(in) :: &
         basisIntegralsMetric !< Input:

    character(len=*), intent(in) :: &
         variationalDenominatorType !< Input:

    real(kind=RKIND), dimension(:), intent(out) :: &
         variationalDenominator

    integer :: &
         iVertex, &
         iSurroundingCell, &
         iStressVertex, &
         iCell, &
         iVelocityVertex

    if (trim(variationalDenominatorType) == "alternate") then

       do iVertex = 1, nVertices

          variationalDenominator(iVertex) = 0.0_RKIND

          ! loop over surrounding cells
          do iSurroundingCell = 1, vertexDegree

             ! get the cell number of this cell
             iCell = cellsOnVertex(iSurroundingCell, iVertex)

             ! get the vertexOnCell number of the iVertex velocity point from cell iCell
             iVelocityVertex = cellVerticesAtVertex(iSurroundingCell,iVertex)

             ! loop over the vertices of the surrounding cell
             do iStressVertex = 1, nEdgesOnCell(iCell)

                variationalDenominator(iVertex) = variationalDenominator(iVertex) + &
                     basisIntegralsMetric(iStressVertex,iVelocityVertex,iCell)

             enddo ! iStressVertex

          enddo ! iSurroundingCell

          ! inverse
          variationalDenominator(iVertex) = variationalDenominator(iVertex)

       enddo ! iVertex

    else if (trim(variationalDenominatorType) == "original") then

       do iVertex = 1, nVertices
          variationalDenominator(iVertex) = areaTriangle(iVertex)
       enddo ! iVertex

    else

       call MPAS_log_write("Unknown variational denominator type: "//trim(variationalDenominatorType), MPAS_LOG_CRIT)

    endif

  end subroutine variational_denominator

!-----------------------------------------------------------------------

  subroutine variational_denominator_c_grid(&
       on_a_sphere, &
       sphere_radius, &
       cellsOnEdge, &
       edgesOnCell, &
       verticesOnEdge, &
       edgesOnVertex, &
       nEdges, &
       nCells, &
       nVertices, &
       nEdgesOnCell, &
       vertexDegree, &
       xCell, &
       yCell, &
       zCell, &
       xVertex, &
       yVertex, &
       zVertex, &
       xEdge, &
       yEdge, &
       zEdge, &
       latCell, &
       lonCell, &
       latVertex, &
       lonVertex, &
       latEdge, &
       lonEdge, &
       variationalDenominatorCGrid, &
       areaCellCGrid, &
       areaTriCGrid)

    use seaice_mesh, only: &
         spherical_triangle_area, &
         planar_triangle_area

    use seaice_velocity_solver_variational_shared, only: &
         seaice_wrapped_index

    integer, intent(in) :: &
         nEdges, &     !< Input:
         nCells, &     !< Input:
         nVertices, &  !< Input:
         vertexDegree  !< Input:

    real(kind=RKIND), intent(in) :: &
         sphere_radius !<Input:

    logical, intent(in) :: &
         on_a_sphere !< Input:

    integer, dimension(:), intent(in) :: &
         nEdgesOnCell     !<Input:

    integer, dimension(:,:), intent(in) :: &
         cellsOnEdge, &     !<Input:
         edgesOnCell, &     !<Input:
         verticesOnEdge, &  !<Input:
         edgesOnVertex      !<Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         xCell,         & !< Input:
         yCell,         & !< Input:
         zCell,         & !< Input:
         xVertex,       & !< Input:
         yVertex,       & !< Input:
         zVertex,       & !< Input:
         xEdge,         & !< Input:
         yEdge,         & !< Input:
         zEdge,         & !< Input:
         latCell,       & !< Input:
         lonCell,       & !< Input:
         latVertex,     & !< Input:
         lonVertex,     & !< Input:
         latEdge,       & !< Input:
         lonEdge          !< Input:

    real(kind=RKIND), dimension(:), intent(out) :: &
         variationalDenominatorCGrid, & !< Output:
         areaCellCGrid, & !< Output:
         areaTriCGrid     !< Output:

    integer :: &
         iEdge, &
         jEdge, &
         iCell, &
         iVertex, &
         iEdgeOnVertex, &
         jEdgeOnVertex, &
         iCellOnEdge, &
         cellOnEdge, &
         vertOnEdge1, &
         vertOnEdge2

    real(kind=RKIND) :: &
         laCell, &
         loCell, &
         laVertex, &
         loVertex, &
         laVert1, &
         loVert1, &
         laVert2, &
         loVert2, &
         laEdge1, &
         loEdge1, &
         laEdge2, &
         loEdge2, &
         xc, &
         yc, &
         zc, &
         xv1, &
         yv1, &
         zv1, &
         xv2, &
         yv2, &
         zv2, &
         xe1, &
         ye1, &
         ze1, &
         xe2, &
         ye2, &
         ze2, &
         xe3, &
         ye3, &
         ze3
     
    real(kind=RKIND) :: &
         triangleArea 

    variationalDenominatorCGrid(:) = 0.0_RKIND
    areaCellCGrid(:) = 0.0_RKIND
    areaTriCGrid(:) = 0.0_RKIND

    if(on_a_sphere) then

       do iEdge = 1, nEdges

          vertOnEdge1 = verticesOnEdge(1, iEdge)
          vertOnEdge2 = verticesOnEdge(2, iEdge)
          laVert1 = latVertex(vertOnEdge1)
          loVert1 = lonVertex(vertOnEdge1)
          laVert2 = latVertex(vertOnEdge2)
          loVert2 = lonVertex(vertOnEdge2)

          do iCellOnEdge = 1, 2

             cellOnEdge = cellsOnEdge(iCellOnEdge, iEdge)
             laCell = latCell(cellOnEdge)             
             loCell = lonCell(cellOnEdge)

             triangleArea = spherical_triangle_area(laCell, loCell, laVert1, loVert1, laVert2, loVert2, sphere_radius)                 
             variationalDenominatorCGrid(iEdge) = variationalDenominatorCGrid(iEdge) + triangleArea 

          end do
          
       end do

       do iCell = 1, nCells

          do iEdge = 1, nEdgesOnCell(iCell)

             jEdge = seaice_wrapped_index(iEdge + 1, nEdgesOnCell(iCell))

             laEdge1 = latEdge(edgesOnCell(iEdge,iCell))
             loEdge1 = lonEdge(edgesOnCell(iEdge,iCell))
             laEdge2 = latEdge(edgesOnCell(jEdge,iCell))
             loEdge2 = lonEdge(edgesOnCell(jEdge,iCell))
             laCell = latCell(iCell)
             loCell = lonCell(iCell)

             areaCellCGrid(iCell) = areaCellCGrid(iCell) + spherical_triangle_area(laCell, loCell, laEdge1, loEdge1, laEdge2, loEdge2, sphere_radius)

          end do

       end do 

       do iVertex = 1, nVertices

          do iEdgeOnVertex = 1, vertexDegree

             jEdgeOnVertex = seaice_wrapped_index(iEdgeOnVertex + 1, vertexDegree)

             laEdge1 = latEdge(edgesOnVertex(iEdgeOnVertex,iVertex))
             loEdge1 = lonEdge(edgesOnVertex(iEdgeOnVertex,iVertex))
             laEdge2 = latEdge(edgesOnVertex(jEdgeOnVertex,iVertex))
             loEdge2 = lonEdge(edgesOnVertex(jEdgeOnVertex,iVertex))
             laVertex = latVertex(iVertex)
             loVertex = lonVertex(iVertex)

             areaTriCGrid(iVertex) = areaTriCGrid(iVertex) + spherical_triangle_area(laEdge1, loEdge1, laEdge2, loEdge2, laVertex, loVertex, sphere_radius)       

          end do

       end do

    else

       do iEdge = 1, nEdges

          vertOnEdge1 = verticesOnEdge(1, iEdge)
          vertOnEdge2 = verticesOnEdge(2, iEdge)
          xv1 = xVertex(vertOnEdge1)
          yv1 = yVertex(vertOnEdge1)
          zv1 = zVertex(vertOnEdge1)
          xv2 = xVertex(vertOnEdge2)
          yv2 = yVertex(vertOnEdge2)
          zv2 = zVertex(vertOnEdge2)

          do iCellOnEdge = 1, 2

             cellOnEdge = cellsOnEdge(iCellOnEdge, iEdge)
             xc = xCell(cellOnEdge)
             yc = yCell(cellOnEdge)
             zc = zCell(cellOnEdge)

             triangleArea = planar_triangle_area(xc, yc, zc, xv1, yv1, zv1, xv2, yv2, zv2)
             variationalDenominatorCGrid(iEdge) = variationalDenominatorCGrid(iEdge) + triangleArea

          end do

       end do

       do iCell = 1, nCells

          do iEdge = 1, nEdgesOnCell(iCell)

             jEdge = seaice_wrapped_index(iEdge + 1, nEdgesOnCell(iCell))

             xe1 = xEdge(edgesOnCell(iEdge,iCell))
             ye1 = yEdge(edgesOnCell(iEdge,iCell)) 
             ze1 = zEdge(edgesOnCell(iEdge,iCell))
             xe2 = xEdge(edgesOnCell(jEdge,iCell))
             ye2 = yEdge(edgesOnCell(jEdge,iCell))
             ze2 = zEdge(edgesOnCell(jEdge,iCell))
             xc = xCell(iCell)
             yc = yCell(iCell)
             zc = zCell(iCell)

             areaCellCGrid(iCell) = areaCellCGrid(iCell) + planar_triangle_area(xc, yc, zc, xe1, ye1, ze1, xe2, ye2, ze2) 

          end do

       end do

       do iVertex = 1, nVertices

          do iEdgeOnVertex = 1, vertexDegree

             jEdgeOnVertex = seaice_wrapped_index(iEdgeOnVertex + 1, vertexDegree)

             xe1 = xEdge(edgesOnVertex(iEdgeOnVertex,iVertex))
             ye1 = yEdge(edgesOnVertex(iEdgeOnVertex,iVertex))
             ze1 = zEdge(edgesOnVertex(iEdgeOnVertex,iVertex))
             xe2 = xEdge(edgesOnVertex(jEdgeOnVertex,iVertex))
             ye2 = yEdge(edgesOnVertex(jEdgeOnVertex,iVertex))
             ze2 = zEdge(edgesOnVertex(jEdgeOnVertex,iVertex))
             xc = xVertex(iVertex)
             yc = yVertex(iVertex)
             zc = zVertex(iVertex)

             areaTriCGrid(iVertex) = areaTriCGrid(iVertex) + planar_triangle_area(xc, yc, zc, xe1, ye1, ze1, xe2, ye2, ze2)

          end do

       end do

    end if

end subroutine variational_denominator_c_grid

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  homogenize_variational_basis_field
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 9th January 2021
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine homogenize_variational_basis_field()

    use seaice_mesh_pool, only: &
         basisGradientU, &
         basisGradientV, &
         basisIntegralsMetric, &
         basisIntegralsU, &
         basisIntegralsV, &
         nCells

    integer :: &
         iCell

    integer, parameter :: &
         iCellHomogenize = 1111

    !call homogenize_cell(basisGradientU, iCellHomogenize)
    !call homogenize_cell(basisGradientV, iCellHomogenize)
    !call homogenize_cell(basisIntegralsMetric, iCellHomogenize)
    !call homogenize_cell(basisIntegralsU, iCellHomogenize)
    !call homogenize_cell(basisIntegralsV, iCellHomogenize)

    do iCell = 1, nCells

       basisGradientU(:,:,iCell)       = basisGradientU(:,:,iCellHomogenize)
       basisGradientV(:,:,iCell)       = basisGradientV(:,:,iCellHomogenize)
       basisIntegralsMetric(:,:,iCell) = basisIntegralsMetric(:,:,iCellHomogenize)
       basisIntegralsU(:,:,iCell)      = basisIntegralsU(:,:,iCellHomogenize)
       basisIntegralsV(:,:,iCell)      = basisIntegralsV(:,:,iCellHomogenize)

    enddo ! iCell

  end subroutine homogenize_variational_basis_field

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  homogenize_variational_basis_field
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 9th January 2021
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine homogenize_cell(&
       field, &
       iCell)

    real(kind=RKIND), dimension(:,:,:), intent(inout) :: &
         field

    integer, intent(in) :: &
         iCell

    integer :: &
         iVertexOnCell1, &
         iVertexOnCell2, &
         nCanonicalValues, &
         iCV

    real(kind=RKIND), dimension(100) :: &
         canonicalValues

    logical :: &
         lFound

    nCanonicalValues = 0

    do iVertexOnCell1 = 1, 6
       do iVertexOnCell2 = 1, 6

          lFound = .false.
          do iCV = 1, nCanonicalValues

             if (abs(abs(canonicalValues(iCV) - abs(field(iVertexOnCell1,iVertexOnCell2,iCell)))) < 1e-12) then
                lFound = .true.
                field(iVertexOnCell1,iVertexOnCell2,iCell) = canonicalValues(iCV) * sign(1.0_RKIND, field(iVertexOnCell1,iVertexOnCell2,iCell))
                exit
             endif

          enddo ! iCV

          if (.not. lFound) then
             nCanonicalValues = nCanonicalValues + 1
             canonicalValues(nCanonicalValues) = abs(field(iVertexOnCell1,iVertexOnCell2,iCell))
          endif

       enddo ! iVertexOnCell2
    enddo ! iVertexOnCell1

    !do iVertexOnCell1 = 1, 6
    !   do iVertexOnCell2 = 1, 6
    !      write(*,*) iVertexOnCell1, iVertexOnCell2, field(iVertexOnCell1,iVertexOnCell2,iCell)
    !   enddo ! iVertexOnCell2
    !enddo ! iVertexOnCell1

  end subroutine homogenize_cell

!-----------------------------------------------------------------------
! time step
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_strain_tensor_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_strain_tensor_variational(&
       mesh, &
       strain11, &
       strain22, &
       strain12, &
       uVelocity, &
       vVelocity, &
       basisGradientU, &
       basisGradientV, &
       tanLatVertexRotatedOverRadius, &
       solveStress)!{{{

    use seaice_mesh_pool, only: &
         nCells, &
         verticesOnCell, &
         nEdgesOnCell

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:,:), intent(out) :: &
         strain11, & !< Output:
         strain22, & !< Output:
         strain12    !< Output:

    real(kind=RKIND), dimension(:), intent(in) :: &
         uVelocity, & !< Input:
         vVelocity, & !< Input:
         tanLatVertexRotatedOverRadius !< Input:

    real(kind=RKIND), dimension(:,:,:), contiguous, intent(in) :: &
         basisGradientU, & !< Input:
         basisGradientV    !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress !< Input:

    integer :: &
         iCell, &
         iGradientVertex, &
         iBasisVertex, &
         iVertex, &
         jVertex

    real(kind=RKIND) :: &
         strain11Tmp, &
         strain22Tmp, &
         strain12Tmp

    ! loop over cells
#ifdef MPAS_OPENMP_OFFLOAD
!$omp target teams distribute parallel do
#elif MPAS_OPENACC
!$acc parallel loop gang worker
#else
!$omp parallel do default(shared) private(iGradientVertex, iBasisVertex, iVertex, jVertex, &
!$omp&                                    strain11Tmp, strain22Tmp, strain12Tmp)
#endif

    do iCell = 1, nCells
   
       if (solveStress(iCell) == 1) then

          ! loop over velocity points surrounding cell - location of stress and derivative
          do iGradientVertex = 1, nEdgesOnCell(iCell)

             strain11Tmp = 0.0_RKIND
             strain22Tmp = 0.0_RKIND
             strain12Tmp = 0.0_RKIND

             ! loop over basis functions
             do iBasisVertex = 1, nEdgesOnCell(iCell)

                iVertex = verticesOnCell(iBasisVertex,iCell)

                strain11Tmp = strain11Tmp + uVelocity(iVertex) * basisGradientU(iBasisVertex,iGradientVertex,iCell)
                strain22Tmp = strain22Tmp + vVelocity(iVertex) * basisGradientV(iBasisVertex,iGradientVertex,iCell)
                strain12Tmp = strain12Tmp + 0.5_RKIND * (&
                            uVelocity(iVertex) * basisGradientV(iBasisVertex,iGradientVertex,iCell) + &
                            vVelocity(iVertex) * basisGradientU(iBasisVertex,iGradientVertex,iCell))

             enddo ! iVertexOnCell

             ! metric terms
             jVertex = verticesOnCell(iGradientVertex,iCell)

             strain11(iGradientVertex,iCell) = strain11Tmp - vVelocity(jVertex) * tanLatVertexRotatedOverRadius(jVertex)
             strain12(iGradientVertex,iCell) = strain12Tmp + uVelocity(jVertex) * tanLatVertexRotatedOverRadius(jVertex) * 0.5_RKIND
             strain22(iGradientVertex,iCell) = strain22Tmp

          enddo ! jVertexOnCell

       endif ! solveStress

    enddo ! iCell

  end subroutine seaice_strain_tensor_variational!}}}

!-----------------------------------------------------------------------

  subroutine seaice_strain_tensor_variational_c_grid(&
       mesh, &
       strain11, &
       strain22, &
       strain12, &
       strain11Tri, &
       strain22Tri, &
       strain12Tri, &
       uVelocityCGrid, &
       vVelocityCGrid, &
       basisGradientU, &
       basisGradientV, &
       basisGradientUTri, &
       basisGradientVTri, &
       tanLatEdgeRotatedOverRadius, &
       solveStress, &
       solveStressTri, &
       solveVelocityCGrid)!{{{

    use seaice_mesh_pool, only: &
         nCells, &
         nEdges, &
         nVertices, & 
         vertexDegree, &
         edgesOnCell, &
         edgesOnVertex, &
         nEdgesOnCell, &
         verticesOnEdge

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:,:), intent(out) :: &
         strain11,    & !< Output:
         strain22,    & !< Output:
         strain12,    &!< Output:
         strain11Tri, & !< Output:
         strain22Tri, & !< Output:
         strain12Tri    !< Output:

    real(kind=RKIND), dimension(:), intent(in) :: &
         tanLatEdgeRotatedOverRadius !< Input:

    real(kind=RKIND), dimension(:), intent(inout) :: &
         uVelocityCGrid, & !< Input:
         vVelocityCGrid    !< Input:

    real(kind=RKIND), dimension(:,:,:), contiguous, intent(in) :: &
         basisGradientU, & !< Input:
         basisGradientV, & !< Input:
         basisGradientUTri, & !< Input:
         basisGradientVTri    !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress, &  !< Input:
         solveStressTri, &  !< Input:
         solveVelocityCGrid !< Input:

    integer :: &
         iCell, &
         iGradientEdge, &
         iBasisEdge, &
         iEdge, &
         jEdge, &
         vertexOnEdge1, & 
         vertexOnEdge2, &
         iVertex

    real(kind=RKIND) :: &
         strain11Tmp, &
         strain22Tmp, &
         strain12Tmp

    do iCell = 1, nCells
   
       if (solveStress(iCell) == 1) then

          ! loop over velocity points surrounding cell - location of stress and derivative
          do iGradientEdge = 1, nEdgesOnCell(iCell)

             strain11Tmp = 0.0_RKIND
             strain22Tmp = 0.0_RKIND
             strain12Tmp = 0.0_RKIND

             ! loop over basis functions
             do iBasisEdge = 1, nEdgesOnCell(iCell)

                iEdge = edgesOnCell(iBasisEdge,iCell)

                strain11Tmp = strain11Tmp + uVelocityCGrid(iEdge) * basisGradientU(iBasisEdge,iGradientEdge,iCell)
                strain22Tmp = strain22Tmp + vVelocityCGrid(iEdge) * basisGradientV(iBasisEdge,iGradientEdge,iCell)
                strain12Tmp = strain12Tmp + 0.5_RKIND * (&
                              uVelocityCGrid(iEdge) * basisGradientV(iBasisEdge,iGradientEdge,iCell) + &
                              vVelocityCGrid(iEdge) * basisGradientU(iBasisEdge,iGradientEdge,iCell))

             end do ! iEdgeOnCell

             ! metric terms
             jEdge = edgesOnCell(iGradientEdge,iCell)

                   strain11(iGradientEdge,iCell) = strain11Tmp - vVelocityCGrid(jEdge) * tanLatEdgeRotatedOverRadius(jEdge)
                   strain12(iGradientEdge,iCell) = strain12Tmp + uVelocityCGrid(jEdge) * tanLatEdgeRotatedOverRadius(jEdge) * 0.5_RKIND
                   strain22(iGradientEdge,iCell) = strain22Tmp

          end do ! jEdgeOnCell

       end if !solveStress

    end do ! iCell

    do iVertex = 1, nVertices

       if (solveStressTri(iVertex) == 1) then

          ! loop over velocity points surrounding triangle - location of stress and derivative
          do iGradientEdge = 1, vertexDegree

             strain11Tmp = 0.0_RKIND
             strain22Tmp = 0.0_RKIND
             strain12Tmp = 0.0_RKIND

             ! loop over basis functions
             do iBasisEdge = 1, vertexDegree

                iEdge = edgesOnVertex(iBasisEdge,iVertex)


                strain11Tmp = strain11Tmp + uVelocityCGrid(iEdge) * basisGradientUTri(iBasisEdge,iGradientEdge,iVertex)
                strain22Tmp = strain22Tmp + vVelocityCGrid(iEdge) * basisGradientVTri(iBasisEdge,iGradientEdge,iVertex)
                strain12Tmp = strain12Tmp + 0.5_RKIND * (&
                              uVelocityCGrid(iEdge) * basisGradientVTri(iBasisEdge,iGradientEdge,iVertex) + &
                              vVelocityCGrid(iEdge) * basisGradientUTri(iBasisEdge,iGradientEdge,iVertex))

             end do ! iEdgeOnVertex

             ! metric terms
             jEdge = edgesOnVertex(iGradientEdge,iVertex)

             strain11Tri(iGradientEdge,iVertex) = strain11Tmp - vVelocityCGrid(jEdge) * tanLatEdgeRotatedOverRadius(jEdge)
             strain12Tri(iGradientEdge,iVertex) = strain12Tmp + uVelocityCGrid(jEdge) * tanLatEdgeRotatedOverRadius(jEdge) * 0.5_RKIND
             strain22Tri(iGradientEdge,iVertex) = strain22Tmp

          end do ! jEdgeOnVertex

       end if !solveStressTri

    end do ! iVertex

  end subroutine seaice_strain_tensor_variational_c_grid !}}}

!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_average_strains_on_vertex
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_average_strains_on_vertex(&
       areaCell, &
       strain11, &
       strain22, &
       strain12)

    use seaice_mesh_pool, only: &
         nCells, &
         nVerticesSolve, &
         cellVerticesAtVertex, &
         cellsOnVertex, &
         vertexDegree

    real(kind=RKIND), dimension(:), intent(in) :: &
         areaCell !< Input

    real(kind=RKIND), dimension(:,:), intent(inout) :: &
         strain11, & !< Input/Output:
         strain22, & !< Input/Output:
         strain12    !< Input/Output:

    real(kind=RKIND) :: &
         strain11avg, &
         strain22avg, &
         strain12avg, &
         denominator

    integer :: &
         iVertex, &
         iVertexDegree, &
         iCell, &
         iVertexOnCell

    do iVertex = 1, nVerticesSolve

       strain11avg = 0.0_RKIND
       strain22avg = 0.0_RKIND
       strain12avg = 0.0_RKIND
       denominator = 0.0_RKIND

       do iVertexDegree = 1, vertexDegree

          iCell = cellsOnVertex(iVertexDegree,iVertex)

          if (iCell <= nCells) then

             iVertexOnCell = cellVerticesAtVertex(iVertexDegree,iVertex)

             strain11avg = strain11avg + strain11(iVertexOnCell,iCell) * areaCell(iCell)
             strain22avg = strain22avg + strain22(iVertexOnCell,iCell) * areaCell(iCell)
             strain12avg = strain12avg + strain12(iVertexOnCell,iCell) * areaCell(iCell)
             denominator = denominator + areaCell(iCell)

          endif

       enddo ! iVertexDegree

       strain11avg = strain11avg / denominator
       strain22avg = strain22avg / denominator
       strain12avg = strain12avg / denominator

       do iVertexDegree = 1, vertexDegree

          iCell = cellsOnVertex(iVertexDegree,iVertex)

          if (iCell <= nCells) then

             iVertexOnCell = cellVerticesAtVertex(iVertexDegree,iVertex)

             strain11(iVertexOnCell,iCell) = strain11avg
             strain22(iVertexOnCell,iCell) = strain22avg
             strain12(iVertexOnCell,iCell) = strain12avg

          endif

       enddo ! iCellOnVertex

    enddo ! iVertex

  end subroutine seaice_average_strains_on_vertex

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_tensor_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_stress_tensor_variational(&
       mesh, &
       stress11, &
       stress22, &
       stress12, &
       strain11, &
       strain22, &
       strain12, &
       icePressure, &
       replacementPressure, &
       solveStress, &
       dtElastic)!{{{

    use seaice_velocity_solver_constitutive_relation, only: &
         constitutiveRelationType, &
         EVP_CONSTITUTIVE_RELATION, &
         REVISED_EVP_CONSTITUTIVE_RELATION, &
         LINEAR_CONSTITUTIVE_RELATION, &
         seaice_evp_constitutive_relation, &
         seaice_evp_constitutive_relation_revised, &
         seaice_linear_constitutive_relation, &
         eccentricitySquared, puny, dampingTimescale

    use seaice_mesh_pool, only: &
         nCells, &
         nEdgesOnCell

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:,:), contiguous, intent(inout) :: &
         stress11, & !< Input/Output:
         stress22, & !< Input/Output:
         stress12    !< Input/Output:

    real(kind=RKIND), dimension(:,:), contiguous, intent(out) :: &
         replacementPressure !< Output:

    real(kind=RKIND), dimension(:,:), contiguous, intent(in) :: &
         strain11, & !< Input:
         strain22, & !< Input:
         strain12    !< Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         icePressure !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress !< Input:

    real(kind=RKIND), intent(in) :: &
         dtElastic !< Input:

    integer :: &
         iCell, &
         iVertexOnCell

    real(kind=RKIND), dimension(:), pointer :: &
         areaCell

    real(kind=RKIND) :: &
         strainDivergence,    &
         strainTension,       &
         strainShearing,      &
         stress1,             &
         stress2,             &
         Delta,               &
         pressureCoefficient, &
         denominator

    ! init variables
    call MPAS_pool_get_array(mesh, "areaCell", areaCell)

    if (constitutiveRelationType == EVP_CONSTITUTIVE_RELATION) then

       denominator = 1.0_RKIND + (0.5_RKIND * dtElastic) / dampingTimescale

#ifdef MPAS_OPENMP_OFFLOAD
!$omp target teams distribute parallel do
#elif MPAS_OPENACC
!$acc parallel loop gang worker
#else
!$omp parallel do default(shared) private(iVertexOnCell)
#endif
       do iCell = 1, nCells

          replacementPressure(:,iCell) = 0.0_RKIND

          if (solveStress(iCell) == 1) then

#if defined(MPAS_OPENMP_OFFLOAD) || defined(MPAS_OPENACC)
             ! inline call to seaice_evp_constitutive_relation for GPUs
             do iVertexOnCell = 1, nEdgesOnCell(iCell)

                ! convert from stress11 to stress1 etc
                strainDivergence = strain11(iVertexOnCell,iCell) + strain22(iVertexOnCell,iCell)
                strainTension    = strain11(iVertexOnCell,iCell) - strain22(iVertexOnCell,iCell)
                strainShearing   = strain12(iVertexOnCell,iCell) * 2.0_RKIND

                stress1 = stress11(iVertexOnCell,iCell) + stress22(iVertexOnCell,iCell)
                stress2 = stress11(iVertexOnCell,iCell) - stress22(iVertexOnCell,iCell)

                ! perform the constituitive relation
                Delta = sqrt(strainDivergence*strainDivergence + &
                             (strainTension*strainTension + strainShearing*strainShearing) / eccentricitySquared)

                pressureCoefficient                      = icePressure(iCell) / max(Delta,puny)
                replacementPressure(iVertexOnCell,iCell) = pressureCoefficient * Delta

                pressureCoefficient = (pressureCoefficient * dtElastic) / (2.0_RKIND * dampingTimescale)

                stress1  = (stress1  +  pressureCoefficient                        * (strainDivergence - Delta)) / denominator
                stress2  = (stress2  + (pressureCoefficient / eccentricitySquared) *  strainTension            ) / denominator
                stress12(iVertexOnCell,iCell) = (stress12(iVertexOnCell,iCell) &
                                     + (pressureCoefficient / eccentricitysquared) * strainShearing * 0.5_RKIND) / denominator

                ! convert back
                stress11(iVertexOnCell,iCell) = 0.5_RKIND * (stress1 + stress2)
                stress22(iVertexOnCell,iCell) = 0.5_RKIND * (stress1 - stress2)

#else
             !$omp simd
             do iVertexOnCell = 1, nEdgesOnCell(iCell)

                call seaice_evp_constitutive_relation(&
                     stress11(iVertexOnCell,iCell), &
                     stress22(iVertexOnCell,iCell), &
                     stress12(iVertexOnCell,iCell), &
                     strain11(iVertexOnCell,iCell), &
                     strain22(iVertexOnCell,iCell), &
                     strain12(iVertexOnCell,iCell), &
                     icePressure(iCell), &
                     replacementPressure(iVertexOnCell,iCell), &
                     areaCell(iCell), &
                     dtElastic)
#endif
             enddo ! iVertexOnCell

          endif ! solveStress

       enddo ! iCell

    else if (constitutiveRelationType == REVISED_EVP_CONSTITUTIVE_RELATION) then

#ifdef MPAS_OPENMP
!$omp parallel do default(shared) private(iVertexOnCell)
#endif
       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             !$omp simd
             do iVertexOnCell = 1, nEdgesOnCell(iCell)

                call seaice_evp_constitutive_relation_revised(&
                     stress11(iVertexOnCell,iCell), &
                     stress22(iVertexOnCell,iCell), &
                     stress12(iVertexOnCell,iCell), &
                     strain11(iVertexOnCell,iCell), &
                     strain22(iVertexOnCell,iCell), &
                     strain12(iVertexOnCell,iCell), &
                     icePressure(iCell), &
                     replacementPressure(iVertexOnCell,iCell), &
                     areaCell(iCell))

             enddo ! iVertexOnCell

          endif ! solveStress

       enddo ! iCell

    else if (constitutiveRelationType == LINEAR_CONSTITUTIVE_RELATION) then

#ifdef MPAS_OPENMP
!$omp parallel do default(shared) private(iCell, iVertexOnCell)
#endif
       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             !$omp simd
             do iVertexOnCell = 1, nEdgesOnCell(iCell)

                call seaice_linear_constitutive_relation(&
                     stress11(iVertexOnCell,iCell), &
                     stress22(iVertexOnCell,iCell), &
                     stress12(iVertexOnCell,iCell), &
                     strain11(iVertexOnCell,iCell), &
                     strain22(iVertexOnCell,iCell), &
                     strain12(iVertexOnCell,iCell))

             enddo ! iVertexOnCell

          endif ! solveStress

       enddo ! iCell

    endif ! constitutiveRelationType

  end subroutine seaice_stress_tensor_variational!}}}

!-----------------------------------------------------------------------

  subroutine seaice_stress_tensor_variational_c_grid(&
       mesh, &
       velocityVariational, &
       stress11, &
       stress22, &
       stress12, &
       stress11Tri, &
       stress22Tri, &
       stress12Tri, &
       strain11, &
       strain22, &
       strain12, &
       strain11Tri, &
       strain22Tri, &
       strain12Tri, &
       icePressure, &
       replacementPressure, &
       replacementPressureTri, &
       solveStress, &
       solveStressTri, &
       dtElastic)!{{{

    use seaice_velocity_solver_constitutive_relation, only: &
         constitutiveRelationType, &
         EVP_CONSTITUTIVE_RELATION, &
         REVISED_EVP_CONSTITUTIVE_RELATION, &
         LINEAR_CONSTITUTIVE_RELATION, &
         seaice_evp_constitutive_relation, &
         seaice_evp_constitutive_relation_revised, &
         seaice_linear_constitutive_relation, &
         eccentricitySquared, puny, dampingTimescale

    use seaice_mesh_pool, only: &
         nCells, &
         nVertices, &
         nEdgesOnCell, &
         cellsOnVertex, &
         vertexDegree, &
         kiteAreasOnVertex, &
         areaTriangle

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh, & !< Input:
         velocityVariational !< Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         icePressure !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress, & !< Input:
         solveStressTri !< Input:

    real(kind=RKIND), dimension(:,:), contiguous, intent(inout) :: &
         stress11,    & !< Input/Output:
         stress22,    & !< Input/Output:
         stress12,    & !< Input/Output:
         stress11Tri, & !< Input/Output:
         stress22Tri, & !< Input/Output:
         stress12Tri    !< Input/Output:

    real(kind=RKIND), dimension(:,:), contiguous, intent(out) :: &
         replacementPressure, &    !< Output:
         replacementPressureTri    !< Output:

    real(kind=RKIND), dimension(:,:), contiguous, intent(in) :: &
         strain11,    & !< Input:
         strain22,    & !< Input:
         strain12,    & !< Input:
         strain11Tri, & !< Input:
         strain22Tri, & !< Input:
         strain12Tri    !< Input:

    real(kind=RKIND), intent(in) :: &
         dtElastic !< Input:

    real(kind=RKIND), dimension(:), pointer :: &
         areaCellCGrid, &
         areaTriCGrid

    integer :: &
         iCell, &
         iVertex, &
         iEdgesOnCell, &
         iEdgesOnVertex, &
         iCellsOnVertex

    real(kind=RKIND), dimension(:), allocatable :: &
         icePressureCellCGrid, &  !TODO: give as input!!!!
         icePressureTriCGrid

    real(kind=RKIND) :: &
         strainDivergence,    &
         strainTension,       &
         strainShearing,      &
         stress1,             &
         stress2,             &
         Delta,               &
         pressureCoefficient, &
         denominator

    ! TODO: I removed the OpenMP parts!

    allocate(icePressureCellCGrid(nCells))
    allocate(icePressureTriCGrid(nVertices))

    ! TODO: this has to be handled properly in the future
    do iCell = 1, nCells
       icePressureCellCGrid(iCell) = icePressure(iCell)
    end do

    icePressureTriCGrid(:) = 0.0_RKIND
    do iVertex = 1, nVertices
       do iCellsOnVertex = 1, vertexDegree
          iCell = cellsOnVertex(iCellsOnVertex,iVertex)
          icePressureTriCGrid(iVertex) = icePressureTriCGrid(iVertex) + kiteAreasOnVertex(iCellsOnVertex, iVertex) * icePressure(iCell) / areaTriangle(iVertex)
       end do
    end do

    ! init variables
    call MPAS_pool_get_array(velocityVariational, "areaCellCGrid", areaCellCGrid)
    call MPAS_pool_get_array(velocityVariational, "areaTriCGrid", areaTriCGrid)

    if (constitutiveRelationType == EVP_CONSTITUTIVE_RELATION) then

       denominator = 1.0_RKIND + (0.5_RKIND * dtElastic) / dampingTimescale

       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             do iEdgesOnCell = 1, nEdgesOnCell(iCell)

                call seaice_evp_constitutive_relation(&
                     stress11(iEdgesOnCell,iCell), &
                     stress22(iEdgesOnCell,iCell), &
                     stress12(iEdgesOnCell,iCell), &
                     strain11(iEdgesOnCell,iCell), &
                     strain22(iEdgesOnCell,iCell), &
                     strain12(iEdgesOnCell,iCell), &
                     icePressureCellCGrid(iCell), &
                     replacementPressure(iEdgesOnCell,iCell), &
                     areaCellCGrid(iCell), &
                     dtElastic)

              end do ! iEdgesOnCell
 
          end if 
     
       end do ! iCell

       do iVertex = 1, nVertices

          if (solveStressTri(iVertex) == 1) then

             do iEdgesOnVertex = 1, vertexDegree

                call seaice_evp_constitutive_relation(&
                     stress11Tri(iEdgesOnVertex,iVertex), &
                     stress22Tri(iEdgesOnVertex,iVertex), &
                     stress12Tri(iEdgesOnVertex,iVertex), &
                     strain11Tri(iEdgesOnVertex,iVertex), &
                     strain22Tri(iEdgesOnVertex,iVertex), &
                     strain12Tri(iEdgesOnVertex,iVertex), &
                     icePressureTriCGrid(iVertex), &
                     replacementPressureTri(iEdgesOnVertex,iVertex), &
                     areaTriCGrid(iVertex), &
                     dtElastic)

              end do ! iEdgesOnVertex

           end if

       end do ! iVertex

    else if (constitutiveRelationType == REVISED_EVP_CONSTITUTIVE_RELATION) then

       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             do iEdgesOnCell = 1, nEdgesOnCell(iCell)

                call seaice_evp_constitutive_relation_revised(&
                     stress11(iEdgesOnCell,iCell), &
                     stress22(iEdgesOnCell,iCell), &
                     stress12(iEdgesOnCell,iCell), &
                     strain11(iEdgesOnCell,iCell), &
                     strain22(iEdgesOnCell,iCell), &
                     strain12(iEdgesOnCell,iCell), &
                     icePressureCellCGrid(iCell), &
                     replacementPressure(iEdgesOnCell,iCell), &
                     areaCellCGrid(iCell))

              end do ! iEdgesOnCell

          end if

       end do ! iCell

       do iVertex = 1, nVertices

          if (solveStressTri(iVertex) == 1) then

             do iEdgesOnVertex = 1, vertexDegree

                call seaice_evp_constitutive_relation_revised(&
                     stress11Tri(iEdgesOnVertex,iVertex), &
                     stress22Tri(iEdgesOnVertex,iVertex), &
                     stress12Tri(iEdgesOnVertex,iVertex), &
                     strain11Tri(iEdgesOnVertex,iVertex), &
                     strain22Tri(iEdgesOnVertex,iVertex), &
                     strain12Tri(iEdgesOnVertex,iVertex), &
                     icePressureTriCGrid(iVertex), &
                     replacementPressureTri(iEdgesOnVertex,iVertex), &
                     areaTriCGrid(iVertex))

              end do ! iEdgesOnVertex

          end if  

       end do ! iVertex

    else if (constitutiveRelationType == LINEAR_CONSTITUTIVE_RELATION) then

       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             do iEdgesOnCell = 1, nEdgesOnCell(iCell)

                call seaice_linear_constitutive_relation(&
                     stress11(iEdgesOnCell,iCell), &
                     stress22(iEdgesOnCell,iCell), &
                     stress12(iEdgesOnCell,iCell), &
                     strain11(iEdgesOnCell,iCell), &
                     strain22(iEdgesOnCell,iCell), &
                     strain12(iEdgesOnCell,iCell))

             end do ! iEdgesOnCell

          end if  

       end do ! iCell

       do iVertex = 1, nVertices

          if (solveStressTri(iVertex) == 1) then

             do iEdgesOnVertex = 1, vertexDegree

                call seaice_linear_constitutive_relation(&
                     stress11Tri(iEdgesOnVertex,iVertex), &
                     stress22Tri(iEdgesOnVertex,iVertex), &
                     stress12Tri(iEdgesOnVertex,iVertex), &
                     strain11Tri(iEdgesOnVertex,iVertex), &
                     strain22Tri(iEdgesOnVertex,iVertex), &
                     strain12Tri(iEdgesOnVertex,iVertex))

             end do ! iEdgesOnVertex

           end if

       end do ! iVertex

    end if ! constitutiveRelationType

  deallocate(icePressureCellCGrid)
  deallocate(icePressureTriCGrid)

  end subroutine seaice_stress_tensor_variational_c_grid!}}}

!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_tensor_variational_linear
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 3rd October 2019
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_stress_tensor_variational_linear(&
       mesh, &
       stress11, &
       stress22, &
       stress12, &
       strain11, &
       strain22, &
       strain12, &
       solveStress)!{{{

    use seaice_velocity_solver_constitutive_relation, only: &
         seaice_linear_constitutive_relation

    use seaice_mesh_pool, only: &
         nCells, &
         nEdgesOnCell

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:,:), contiguous, intent(out) :: &
         stress11, & !< Input/Output:
         stress22, & !< Input/Output:
         stress12    !< Input/Output:

    real(kind=RKIND), dimension(:,:), contiguous, intent(in) :: &
         strain11, & !< Input:
         strain22, & !< Input:
         strain12    !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress !< Input:

    integer :: &
         iCell, &
         iVertexOnCell

#ifdef MPAS_OPENMP
!$omp parallel do default(shared) private(iCell, iVertexOnCell)
#endif
    do iCell = 1, nCells

       if (solveStress(iCell) == 1) then

          !$omp simd
          do iVertexOnCell = 1, nEdgesOnCell(iCell)

             call seaice_linear_constitutive_relation(&
                  stress11(iVertexOnCell,iCell), &
                  stress22(iVertexOnCell,iCell), &
                  stress12(iVertexOnCell,iCell), &
                  strain11(iVertexOnCell,iCell), &
                  strain22(iVertexOnCell,iCell), &
                  strain12(iVertexOnCell,iCell))

          enddo ! iVertexOnCell

       endif ! solveStress

    enddo ! iCell

  end subroutine seaice_stress_tensor_variational_linear!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_divergence_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_stress_divergence_variational(&
       mesh, &
       stressDivergenceU, &
       stressDivergenceV, &
       stress11, &
       stress22, &
       stress12, &
       basisIntegralsU, &
       basisIntegralsV, &
       basisIntegralsMetric, &
       variationalDenominator, &
       tanLatVertexRotatedOverRadius, &
       cellVerticesAtVertex, &
       solveVelocity)!{{{

    use seaice_mesh_pool, only: &
         nVerticesSolve, &
         cellsOnVertex, &
         nEdgesOnCell, &
         areaTriangle, &
         vertexDegree

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:), intent(out) :: &
         stressDivergenceU, & !< Output:
         stressDivergenceV    !< Output:

    real(kind=RKIND), dimension(:,:), contiguous, intent(in) :: &
         stress11, & !< Input:
         stress22, & !< Input:
         stress12    !< Input:

    real(kind=RKIND), dimension(:,:,:), contiguous, intent(in) :: &
         basisIntegralsU, &   !< Input:
         basisIntegralsV, &   !< Input:
         basisIntegralsMetric !< Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         tanLatVertexRotatedOverRadius, & !< Input:
         variationalDenominator           !< Input:

    integer, dimension(:,:), intent(in) :: &
         cellVerticesAtVertex !< Input:

    integer, dimension(:), intent(in) :: &
         solveVelocity !< Input:

    real(kind=RKIND) :: &
         stressDivergenceUVertex, &
         stressDivergenceVVertex, &
         stressDivergenceUCell, &
         stressDivergenceVCell

    integer :: &
         iVertex, &
         iSurroundingCell, &
         iCell, &
         iStressVertex, &
         iVelocityVertex

    ! loop over velocity positions
#ifdef MPAS_OPENMP_OFFLOAD
!$omp target teams distribute parallel do
#elif MPAS_OPENACC
!$acc parallel loop gang worker
#else
!$omp parallel do default(shared) private(stressDivergenceUVertex, stressDivergenceVVertex, &
!$omp&   iSurroundingCell, iCell, iVelocityVertex, stressDivergenceUCell, stressDivergenceVCell, iStressVertex)
#endif
    do iVertex = 1, nVerticesSolve

       if (solveVelocity(iVertex) == 1) then

          stressDivergenceUVertex = 0.0_RKIND
          stressDivergenceVVertex = 0.0_RKIND

          ! loop over surrounding cells
          do iSurroundingCell = 1, vertexDegree

             ! get the cell number of this cell
             iCell = cellsOnVertex(iSurroundingCell, iVertex)

             ! get the vertexOnCell number of the iVertex velocity point from cell iCell
             iVelocityVertex = cellVerticesAtVertex(iSurroundingCell,iVertex)

             stressDivergenceUCell = 0.0_RKIND
             stressDivergenceVCell = 0.0_RKIND

             ! loop over the vertices of the surrounding cell
             do iStressVertex = 1, nEdgesOnCell(iCell)

                ! normal & metric terms
                stressDivergenceUCell = stressDivergenceUCell - &
                     stress11(iStressVertex,iCell) * basisIntegralsU(iStressVertex,iVelocityVertex,iCell) - &
                     stress12(iStressVertex,iCell) * basisIntegralsV(iStressVertex,iVelocityVertex,iCell) - &
                     stress12(iStressVertex,iCell) * basisIntegralsMetric(iStressVertex,iVelocityVertex,iCell) * &
                     tanLatVertexRotatedOverRadius(iVertex)

                stressDivergenceVCell = stressDivergenceVCell - &
                     stress22(iStressVertex,iCell) * basisIntegralsV(iStressVertex,iVelocityVertex,iCell) - &
                     stress12(iStressVertex,iCell) * basisIntegralsU(iStressVertex,iVelocityVertex,iCell) + &
                     stress11(iStressVertex,iCell) * basisIntegralsMetric(iStressVertex,iVelocityVertex,iCell) * &
                     tanLatVertexRotatedOverRadius(iVertex)

             enddo ! iStressVertex

             stressDivergenceUVertex = stressDivergenceUVertex + stressDivergenceUCell
             stressDivergenceVVertex = stressDivergenceVVertex + stressDivergenceVCell

          enddo ! iSurroundingCell

          stressDivergenceU(iVertex) = stressDivergenceUVertex / variationalDenominator(iVertex)
          stressDivergenceV(iVertex) = stressDivergenceVVertex / variationalDenominator(iVertex)

       endif ! solveVelocity

    enddo ! iVertex

  end subroutine seaice_stress_divergence_variational!}}}

!-----------------------------------------------------------------------

  subroutine seaice_stress_divergence_variational_c_grid(&
       mesh, &
       boundary, &
       stressDivergenceUCGrid, &
       stressDivergenceVCGrid, &
       stress11, &
       stress22, &
       stress12, &
       stress11Tri, &
       stress22Tri, &
       stress12Tri, &
       basisIntegralsU, &
       basisIntegralsV, &
       basisIntegralsMetric, &
       basisIntegralsUTri, &
       basisIntegralsVTri, &
       basisIntegralsMetricTri, &
       variationalDenominatorCGrid, &
       tanLatEdgeRotatedOverRadius, &
       cellEdgesAtEdge, &
       triangleEdgesAtEdge, &
       solveVelocityCGrid)!{{{

    use seaice_mesh_pool, only: &
         nEdges, &
         nEdgesOnCell, &
         cellsOnEdge,  &
         verticesOnEdge, &
         vertexDegree

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh, &   !< Input:
         boundary  !< Input:

    real(kind=RKIND), dimension(:), intent(out) :: &
         stressDivergenceUCGrid,    & !< Output:
         stressDivergenceVCGrid       !< Output:

    real(kind=RKIND), dimension(:,:), contiguous, intent(in) :: &
         stress11,        & !< Input:
         stress22,        & !< Input:
         stress12,        & !< Input:
         stress11Tri,     & !< Input:
         stress22Tri,     & !< Input:
         stress12Tri        !< Input:

    real(kind=RKIND), dimension(:,:,:), contiguous, intent(in) :: &
         basisIntegralsU,        &   !< Input:
         basisIntegralsV,        &   !< Input:
         basisIntegralsMetric,   &   !< Input:
         basisIntegralsUTri,     &   !< Input:
         basisIntegralsVTri,     &   !< Input:
         basisIntegralsMetricTri     !< Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         tanLatEdgeRotatedOverRadius, & !< Input:
         variationalDenominatorCGrid    !< Input:

    integer, dimension(:,:), intent(in) :: &
         cellEdgesAtEdge,     &   !< Input:
         triangleEdgesAtEdge      !< Input:

    integer, dimension(:), intent(in) :: &
         solveVelocityCGrid !< Input:

    real(kind=RKIND) :: &
         stressDivergenceUEdge, &
         stressDivergenceVEdge, &
         stressDivergenceUCell, &
         stressDivergenceVCell

    integer :: &
         iEdge, &
         iCell, &
         iVertex, &
         iCellOnEdge, &
         iTriangleOnEdge, &
         iVelocityEdge, &
         iStressEdge

!!!!!!!!!! TODO: I REMOVED THE STUFF ON OpenMP

    do iEdge = 1, nEdges

       if (solveVelocityCGrid(iEdge)==1) then

          stressDivergenceUCGrid(iEdge) = 0.0_RKIND
          stressDivergenceVCGrid(iEdge) = 0.0_RKIND

          stressDivergenceUEdge = 0.0_RKIND
          stressDivergenceVEdge = 0.0_RKIND

          ! loop over surrounding cells
          do iCellOnEdge = 1, 2

             ! get the cell number of this cell
             iCell = cellsOnEdge(iCellOnEdge, iEdge)

             ! get the local index of iEdge on iCellOnEdge 
             iVelocityEdge = cellEdgesAtEdge(iCellOnEdge,iEdge)

             stressDivergenceUCell = 0.0_RKIND
             stressDivergenceVCell = 0.0_RKIND

             ! loop over the edges of the surrounding cell
             do iStressEdge = 1, nEdgesOnCell(iCell)

                ! normal & metric terms
                stressDivergenceUCell = stressDivergenceUCell - &
                     stress11(iStressEdge,iCell) * basisIntegralsU(iStressEdge,iVelocityEdge,iCell) - &
                     stress12(iStressEdge,iCell) * basisIntegralsV(iStressEdge,iVelocityEdge,iCell) - &
                     stress12(iStressEdge,iCell) * basisIntegralsMetric(iStressEdge,iVelocityEdge,iCell) * &
                     tanLatEdgeRotatedOverRadius(iEdge)

                stressDivergenceVCell = stressDivergenceVCell - &
                     stress22(iStressEdge,iCell) * basisIntegralsV(iStressEdge,iVelocityEdge,iCell) - &
                     stress12(iStressEdge,iCell) * basisIntegralsU(iStressEdge,iVelocityEdge,iCell) + &
                     stress11(iStressEdge,iCell) * basisIntegralsMetric(iStressEdge,iVelocityEdge,iCell) * &
                     tanLatEdgeRotatedOverRadius(iEdge)

             enddo ! iStressEdge

             stressDivergenceUEdge = stressDivergenceUEdge + stressDivergenceUCell
             stressDivergenceVEdge = stressDivergenceVEdge + stressDivergenceVCell

          enddo ! iCellOnEdge

          stressDivergenceUCGrid(iEdge) = stressDivergenceUCGrid(iEdge) + stressDivergenceUEdge / variationalDenominatorCGrid(iEdge)
          stressDivergenceVCGrid(iEdge) = stressDivergenceVCGrid(iEdge) + stressDivergenceVEdge / variationalDenominatorCGrid(iEdge)

          stressDivergenceUEdge = 0.0_RKIND
          stressDivergenceVEdge = 0.0_RKIND

          ! loop over surrounding triangles
          do iTriangleOnEdge = 1, 2

             ! get the vertex number of this triangle
             iVertex = verticesOnEdge(iTriangleOnEdge, iEdge)

             ! get the local index of iEdge on iTriangleOnEdge
             iVelocityEdge = triangleEdgesAtEdge(iTriangleOnEdge,iEdge)

             stressDivergenceUCell = 0.0_RKIND
             stressDivergenceVCell = 0.0_RKIND

             ! loop over the edges of the surrounding triangle
             do iStressEdge = 1, vertexDegree

                ! normal & metric terms
                stressDivergenceUCell = stressDivergenceUCell - & 
                     stress11Tri(iStressEdge,iVertex) * basisIntegralsUTri(iStressEdge,iVelocityEdge,iVertex) - &
                     stress12Tri(iStressEdge,iVertex) * basisIntegralsVTri(iStressEdge,iVelocityEdge,iVertex) - &
                     stress12Tri(iStressEdge,iVertex) * basisIntegralsMetricTri(iStressEdge,iVelocityEdge,iVertex) * &
                     tanLatEdgeRotatedOverRadius(iEdge)

                stressDivergenceVCell = stressDivergenceVCell - &
                     stress22Tri(iStressEdge,iVertex) * basisIntegralsVTri(iStressEdge,iVelocityEdge,iVertex) - &
                     stress12Tri(iStressEdge,iVertex) * basisIntegralsUTri(iStressEdge,iVelocityEdge,iVertex) + &
                     stress11Tri(iStressEdge,iVertex) * basisIntegralsMetricTri(iStressEdge,iVelocityEdge,iVertex) * &
                     tanLatEdgeRotatedOverRadius(iVertex)

             enddo ! iStressEdge

             stressDivergenceUEdge = stressDivergenceUEdge + stressDivergenceUCell
             stressDivergenceVEdge = stressDivergenceVEdge + stressDivergenceVCell

          enddo ! iTriangleOnEdge

          stressDivergenceUCGrid(iEdge) = stressDivergenceUCGrid(iEdge) + stressDivergenceUEdge / variationalDenominatorCGrid(iEdge)
          stressDivergenceVCGrid(iEdge) = stressDivergenceVCGrid(iEdge) + stressDivergenceVEdge / variationalDenominatorCGrid(iEdge)

       endif ! solveVelocityCGrid

    enddo ! iEdge

  end subroutine seaice_stress_divergence_variational_c_grid!}}}

!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  final_divergence_shear_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date July 9th 2015
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_final_divergence_shear_variational(block)

    use seaice_velocity_solver_constitutive_relation, only: &
         eccentricitySquared

    use seaice_mesh_pool, only: &
         nCells, &
         solveStress, &
         nEdgesOnCell

    type(block_type), intent(inout) :: &
         block

    type(MPAS_pool_type), pointer :: &
         velocityVariationalPool, &
         velocitySolverPool, &
         ridgingPool

    real(kind=RKIND), dimension(:,:), pointer :: &
         strain11, &
         strain22, &
         strain12

    real(kind=RKIND), dimension(:), pointer :: &
         divergence, &
         shear, &
         ridgeConvergence, &
         ridgeShear

    real(kind=RKIND), dimension(:), allocatable :: &
         DeltaAverage

    real(kind=RKIND) :: &
         strainDivergenceSum, &
         strainTensionSum, &
         strainShearingSum, &
         strainDivergence, &
         strainTension, &
         strainShearing, &
         Delta

    logical, pointer :: &
         config_use_column_package

    integer :: &
         iCell, &
         iVertexOnCell

    call MPAS_pool_get_subpool(block % structs, "velocity_variational", velocityVariationalPool)
    call MPAS_pool_get_subpool(block % structs, "velocity_solver", velocitySolverPool)

    call MPAS_pool_get_array(velocityVariationalPool, "strain11", strain11)
    call MPAS_pool_get_array(velocityVariationalPool, "strain22", strain22)
    call MPAS_pool_get_array(velocityVariationalPool, "strain12", strain12)

    call MPAS_pool_get_array(velocitySolverPool, "divergence", divergence)
    call MPAS_pool_get_array(velocitySolverPool, "shear", shear)

    allocate(DeltaAverage(nCells))

    do iCell = 1, nCells

       if (solveStress(iCell) == 1) then

          strainDivergenceSum = 0.0_RKIND
          strainTensionSum    = 0.0_RKIND
          strainShearingSum   = 0.0_RKIND
          DeltaAverage(iCell) = 0.0_RKIND

          do iVertexOnCell = 1, nEdgesOnCell(iCell)

             strainDivergence = strain11(iVertexOnCell,iCell) + strain22(iVertexOnCell,iCell)
             strainTension    = strain11(iVertexOnCell,iCell) - strain22(iVertexOnCell,iCell)
             strainShearing   = strain12(iVertexOnCell,iCell) * 2.0_RKIND

             Delta = sqrt(strainDivergence**2 + (strainTension**2 + strainShearing**2) / eccentricitySquared)

             strainDivergenceSum = strainDivergenceSum + strainDivergence
             strainTensionSum    = strainTensionSum    + strainTension
             strainShearingSum   = strainShearingSum   + strainShearing
             DeltaAverage(iCell) = DeltaAverage(iCell) + Delta

          enddo ! iVertexOnCell

          divergence(iCell)   = strainDivergenceSum                              / real(nEdgesOnCell(iCell),RKIND)
          shear(iCell)        = sqrt(strainTensionSum**2 + strainShearingSum**2) / real(nEdgesOnCell(iCell),RKIND)
          DeltaAverage(iCell) = DeltaAverage(iCell)                              / real(nEdgesOnCell(iCell),RKIND)

       else

          divergence(iCell)   = 0.0_RKIND
          shear(iCell)        = 0.0_RKIND

       endif

    enddo ! iCell

    ! ridging parameters
    call MPAS_pool_get_config(block % configs, "config_use_column_package", config_use_column_package)

    if (config_use_column_package) then

       call MPAS_pool_get_subpool(block % structs, "ridging", ridgingPool)

       call MPAS_pool_get_array(ridgingPool, "ridgeConvergence", ridgeConvergence)
       call MPAS_pool_get_array(ridgingPool, "ridgeShear", ridgeShear)

       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             ridgeConvergence(iCell) = -min(divergence(iCell),0.0_RKIND)
             ridgeShear(iCell)       = 0.5_RKIND * (DeltaAverage(iCell) - abs(divergence(iCell)))

          else

             ridgeConvergence(iCell) = 0.0_RKIND
             ridgeShear(iCell)       = 0.0_RKIND

          endif

       enddo ! iCell

    endif ! config_use_column_package

    ! units - for comparison to CICE
    divergence = divergence * 100.0_RKIND * 86400.0_RKIND
    shear      = shear      * 100.0_RKIND * 86400.0_RKIND

    ! cleanup
    deallocate(DeltaAverage)

  end subroutine seaice_final_divergence_shear_variational

!-----------------------------------------------------------------------

end module seaice_velocity_solver_variational

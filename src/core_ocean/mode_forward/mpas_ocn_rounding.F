!-----------------------------------------------------------------------
!
!  ocn_rounding
!
!> \brief MPAS ocean rounding procedure
!> \author June Wu
!> \date   July 2018
!> \details
!>  This routine handles the rounding of prognostic variables of the ocean.
!>  It rounds the prognostic variables to the specified number of siginificant
!>  digits after each time step of the integration. The number of siginificant
!>  digits can be specified by config_reduced_precision_digits in
!>  namelist.ocean.
!
!-----------------------------------------------------------------------

module ocn_rounding

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timekeeping
   use mpas_dmpar
   use mpas_vector_reconstruction
   use mpas_spline_interpolation
   use mpas_timer
   use mpas_log
   use ocn_constants

   implicit none
   private
   save

   public :: rounding

   contains

   subroutine rounding(domain, config_reduced_precision_digits)
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! Round prognostic variables of the ocean after each time step
   !
   ! Input: domain - current model state in time level 2 (e.g., time_levs(2)state%h(:,:))
   !                 plus mesh meta-data
   ! Input: config_reduced_precision_digits - the number of siginificatn digits
   !
   ! Output: domain - upon exit, time level 2 (e.g., time_levs(2)%state%h(:,:)) contains
   !                  model state after rounding
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (domain_type), intent(inout) :: domain
      integer, intent(in) :: config_reduced_precision_digits
      real(kind=RKIND), external :: round
      real(kind=RKIND) :: layerThickness_val, normalVelocity_val, salinity_val, temperature_val

      integer :: iCell, iEdge, k

      type (dm_info) :: dminfo
      type (block_type), pointer :: block

      type (mpas_pool_type), pointer :: diagnosticsPool, statePool, meshPool, tracersPool

      ! Dimensions
      integer :: nCells, nEdges
      integer, pointer :: nCellsPtr, nEdgesPtr, nVertLevels, num_tracersGroup, startIndex, endIndex
      integer, pointer :: indexTemperature, indexSalinity
      integer, dimension(:), pointer :: nCellsArray, nEdgesArray

      ! State Array Pointers
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocity
      real (kind=RKIND), dimension(:,:), pointer :: layerThickness
      real (kind=RKIND), dimension(:,:,:), pointer :: tracersGroup

      ! Mesh Array Pointers
      integer, dimension(:), pointer :: maxLevelCell, maxLevelEdgeTop, nEdgesOnEdge, nEdgesOnCell
      integer, dimension(:,:), pointer :: cellsOnEdge, edgeMask, edgesOnEdge
      integer, dimension(:,:), pointer :: edgesOnCell, edgeSignOnCell

      ! Obtain State Array
      block => domain % blocklist
      do while (associated(block))
        call mpas_pool_get_subpool(block % structs, 'state', statePool)
        call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
        call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)

        call mpas_pool_get_dimension(block % dimensions, 'nCells', nCellsPtr)
        call mpas_pool_get_dimension(block % dimensions, 'nEdges', nEdgesPtr)
        call mpas_pool_get_dimension(block % dimensions, 'nCellsArray', nCellsArray)
        call mpas_pool_get_dimension(block % dimensions, 'nEdgesArray', nEdgesArray)
        call mpas_pool_get_dimension(block % dimensions, 'nVertLevels', nVertLevels)

        call mpas_pool_get_subpool(block % structs, 'state', statePool)
        call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
        call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
        call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)

        call mpas_pool_get_dimension(tracersPool, 'index_salinity', indexSalinity)
        call mpas_pool_get_dimension(tracersPool, 'index_temperature', indexTemperature)

        call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity, 2)
        call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 2)
        call mpas_pool_get_array(tracersPool, 'activeTracers', tracersGroup, 2)
        call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

        nCells = nCellsPtr
        nEdges = nEdgesPtr

        !$omp do schedule(runtime) private(k)
        do iCell = 1, nCells
           do k = 1, maxLevelCell(iCell)
              layerThickness(k,iCell) = round(layerThickness(k, iCell), config_reduced_precision_digits)
              temperature_val = tracersGroup(indexTemperature, k, iCell)
              tracersGroup(indexTemperature, k, iCell) = round(temperature_val, config_reduced_precision_digits)
              salinity_val = tracersGroup(indexSalinity, k, iCell)
              tracersGroup(indexSalinity, k, iCell) = round(salinity_val, config_reduced_precision_digits)
           end do
        end do
        !$omp end do

        !$omp do schedule(runtime) private(k)
        do iEdge = 1, nEdges
           do k = 1, nVertLevels
              normalVelocity_val = normalVelocity(k, iEdge)
              normalVelocity(k, iEdge) = round(normalVelocity_val, config_reduced_precision_digits)
           end do
        end do
        !$omp end do

        block => block % next
     end do

   end subroutine rounding

end module ocn_rounding

function round(var, digits)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Round function that rounds a  real number to certain number of
! siginificant digits.
!
! Input: var - the real number to be rounded
! Input: digits - the number of siginificant digits
!
! Output: round
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     use mpas_constants
     use ocn_constants

     implicit none

     real(RKIND), intent(in) :: var
     integer, intent(in) :: digits
     real(RKIND) :: mult, div, round, absvar
     integer :: nshift, n

     if (var == 0.0) then
        round = 0.0
     else
        absvar = abs(var)
        n = floor(log10(absvar)) + 1
        nshift = min(15, digits - n)
        if (nshift >= 0) then
           mult = 10.0 ** nshift
           round = sign( anint(absvar * mult) / mult, var)
        else
           div = 10.0 ** nshift
           round = sign( anint(absvar / div) * div, var)
        endif
     end if
end function round
! vim: foldmethod=marker

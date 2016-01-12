! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_init_zapiola_rise
!
!> \brief MPAS ocean initialize case -- zapiola_rise
!> \author Mark Petersen
!> \date   December 2015
!> \details
!>  This module contains the routines for initializing the 
!>  zapiola_rise initial condition
!
!-----------------------------------------------------------------------

module ocn_init_zapiola_rise

   use mpas_kind_types
   use mpas_io_units
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants

   use ocn_constants
   use ocn_init_vertical_grids
   use ocn_init_cell_markers

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

   public :: ocn_init_setup_zapiola_rise, &
             ocn_init_validate_zapiola_rise

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_init_setup_zapiola_rise
!
!> \brief   Setup for this initial condition
!> \author  Mark Petersen
!> \date    December 2015
!> \details 
!>  This routine sets up the initial conditions for this case.
!
!-----------------------------------------------------------------------

  subroutine ocn_init_setup_zapiola_rise(domain, iErr)!{{{

    !--------------------------------------------------------------------

    type (domain_type), intent(inout) :: domain
    integer, intent(out) :: iErr

    type (block_type), pointer :: block_ptr
    type (mpas_pool_type), pointer :: meshPool
    type (mpas_pool_type), pointer :: statePool
    type (mpas_pool_type), pointer :: tracersPool
    type (mpas_pool_type), pointer :: verticalMeshPool
    type (mpas_pool_type), pointer :: forcingPool
    type (mpas_pool_type), pointer :: tracersSurfaceRestoringFieldsPool, tracersInteriorRestoringFieldsPool

    ! local variables
    integer :: iCell, k, idx
    real (kind=RKIND) :: yMin, yMax, xMin, xMax, dcEdgeMin, dcEdgeMinGlobal
    real (kind=RKIND) :: yMinGlobal, yMaxGlobal, yMidGlobal, xMinGlobal, xMaxGlobal
    real (kind=RKIND) :: localVar1, localVar2
    real (kind=RKIND), dimension(:), allocatable :: interfaceLocations

    ! Define config variable pointers
    character (len=StrKIND), pointer :: config_init_configuration, config_vertical_grid
    logical, pointer :: config_zapiola_rise_example_flag1
    real (kind=RKIND), pointer :: config_zapiola_rise_example_flag2

    ! Define dimension pointers
    integer, pointer :: nCellsSolve, nEdgesSolve, nVertLevels, nVertLevelsP1
    integer, pointer :: index_temperature, index_salinity

    ! Define variable pointers
    logical, pointer :: on_a_sphere
    integer, dimension(:), pointer :: maxLevelCell
    real (kind=RKIND), dimension(:), pointer :: lonCell, latCell,refBottomDepth, refZMid, refLayerThickness, &
         vertCoordMovementWeights, bottomDepth, &
         fCell, fEdge, fVertex, dcEdge
    real (kind=RKIND), dimension(:), pointer :: windStressZonal, windStressMeridional
    real (kind=RKIND), dimension(:,:), pointer :: layerThickness, restingThickness
    real (kind=RKIND), dimension(:,:,:), pointer :: activeTracers
       real (kind=RKIND), dimension(:, :), pointer ::    activeTracersPistonVelocity, activeTracersSurfaceRestoringValue
       real (kind=RKIND), dimension(:, :, :), pointer :: activeTracersInteriorRestoringValue, activeTracersInteriorRestoringRate

   ! config flags
    real (kind=RKIND) :: config_zapiola_rise_basin_radius
    real (kind=RKIND) :: config_zapiola_rise_basin_center_lon
    real (kind=RKIND) :: config_zapiola_rise_basin_center_lat
    real (kind=RKIND) :: config_zapiola_rise_basin_max_depth
    real (kind=RKIND) :: config_zapiola_rise_center_lon
    real (kind=RKIND) :: config_zapiola_rise_center_lat
    real (kind=RKIND) :: config_zapiola_rise_radius
    real (kind=RKIND) :: config_zapiola_rise_height
    real (kind=RKIND) :: config_zapiola_rise_shelf_offset
    real (kind=RKIND) :: config_zapiola_rise_shelf_scale
    real (kind=RKIND) :: config_zapiola_rise_shelf_depth
    real (kind=RKIND) :: config_zapiola_rise_wind_center_lat
    real (kind=RKIND) :: config_zapiola_rise_ref_zonal_wind
    real (kind=RKIND) :: config_zapiola_rise_bottom_depth
    real (kind=RKIND) :: config_zapiola_rise_piston_velocity
    real (kind=RKIND) :: config_zapiola_rise_interior_restore_rate

    ! local variables
    real (kind=RKIND) :: radiusEarth, degToRad, radToDeg, distToCenterOfBasin,distToCenterOfZR,depthShelf,depthZR
    real (kind=RKIND) :: referenceTemperature,distToCenterWindLat

    iErr = 0

    call mpas_pool_get_config(ocnConfigs, 'config_init_configuration', config_init_configuration)

    if(config_init_configuration .ne. trim('zapiola_rise')) return

    ! Get config flag settings

    call mpas_pool_get_config(ocnConfigs, 'config_vertical_grid', config_vertical_grid)

    call mpas_pool_get_config(ocnConfigs, 'config_zapiola_rise_example_flag1', config_zapiola_rise_example_flag1)
    call mpas_pool_get_config(ocnConfigs, 'config_zapiola_rise_example_flag2', config_zapiola_rise_example_flag2)

    ! Determine vertical grid for configuration
    call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', meshPool)
    call mpas_pool_get_dimension(meshPool, 'nVertLevelsP1', nVertLevelsP1)
    call mpas_pool_get_config(meshPool, 'on_a_sphere', on_a_sphere)

    if ( .not.on_a_sphere ) call mpas_dmpar_global_abort( &
         'IERROR: The zapiola_rise configuration can only be applied to a spherical mesh. Exiting...')

    allocate(interfaceLocations(nVertLevelsP1))
    call ocn_generate_vertical_grid( config_vertical_grid, interfaceLocations )

    !--------------------------------------------------------------------
    ! Define config flags.  Move to Registry later
    !--------------------------------------------------------------------

    ! Basin geometry
    config_zapiola_rise_bottom_depth=6000.0
    config_zapiola_rise_basin_radius=3200e3
    config_zapiola_rise_basin_center_lon = -20.0
    config_zapiola_rise_basin_center_lat = -40.0
    config_zapiola_rise_basin_max_depth = 6000

    ! Zapiola Rise geometry

    config_zapiola_rise_center_lon = -45.0
    config_zapiola_rise_center_lat = -45.0
    config_zapiola_rise_radius    =  4.0 ! in degrees
    config_zapiola_rise_height    =  1500.0

    ! Shelf geometry
    config_zapiola_rise_shelf_offset = -0.15
    config_zapiola_rise_shelf_scale= 0.05
    config_zapiola_rise_shelf_depth = 150.0

    ! Wind stress parameter
    config_zapiola_rise_wind_center_lat = -64.0
    config_zapiola_rise_ref_zonal_wind=0.1

    ! restoring values
    config_zapiola_rise_piston_velocity = 5.0e-5
    config_zapiola_rise_interior_restore_rate = 5.0e-8

    radiusEarth = a

    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
       call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh', verticalMeshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'forcing', forcingPool)
       call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)

       call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
       call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)

       call mpas_pool_get_dimension(tracersPool, 'index_temperature', index_temperature)
       call mpas_pool_get_dimension(tracersPool, 'index_salinity', index_salinity)

       call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
       call mpas_pool_get_array(meshPool, 'latCell', latCell)
       call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)
       call mpas_pool_get_array(meshPool, 'vertCoordMovementWeights', vertCoordMovementWeights)
       call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
       call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

       call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
       call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)

       call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
       call mpas_pool_get_array(verticalMeshPool, 'refLayerThickness',refLayerThickness)
       call mpas_pool_get_array(verticalMeshPool, 'restingThickness', restingThickness)

       call mpas_pool_get_array(forcingPool, 'windStressZonal', windStressZonal)
       call mpas_pool_get_array(forcingPool, 'windStressMeridional', windStressMeridional)

       ! Set refBottomDepth and refZMid
       do k = 1, nVertLevels
          refBottomDepth(k) = config_zapiola_rise_bottom_depth * interfaceLocations(k+1)
          refLayerThickness(k) = config_zapiola_rise_bottom_depth*(interfaceLocations(k+1) - interfaceLocations(k))
          refZMid(k) = - 0.5_RKIND * config_zapiola_rise_bottom_depth * (interfaceLocations(k+1) + interfaceLocations(k))
       end do

       ! Set vertCoordMovementWeights
       vertCoordMovementWeights(:) = 1.0_RKIND
       degToRad = pii/180.0_RKIND
       radToDeg = 1/degToRad

       layerThickness(:,:) = 0.0_RKIND

       do iCell = 1, nCellsSolve

          !--------------------------------------------------------------------
          ! Compute depth
          !--------------------------------------------------------------------


          distToCenterOfBasin = radiusEarth*acos(sin(latCell(iCell))*sin(config_zapiola_rise_basin_center_lat*degToRad) +  &
               cos(latCell(iCell))*cos(config_zapiola_rise_basin_center_lat*degToRad)*  &
               (cos(lonCell(iCell))*cos(config_zapiola_rise_basin_center_lon*degToRad) +  &
               sin(lonCell(iCell))*sin(config_zapiola_rise_basin_center_lon*degToRad)) )
          distToCenterOfZR = radiusEarth*acos(sin(latCell(iCell))*sin(config_zapiola_rise_center_lat*degToRad) +  &
               cos(latCell(iCell))*cos(config_zapiola_rise_center_lat*degToRad)*  &
               (cos(lonCell(iCell))*cos(config_zapiola_rise_center_lon*degToRad) + &
               sin(lonCell(iCell))*sin(config_zapiola_rise_center_lon*degToRad)) )

          if (distToCenterOfBasin < config_zapiola_rise_basin_radius) then
             depthShelf = config_zapiola_rise_shelf_depth &
                  + 0.5*(1.001_RKIND*config_zapiola_rise_basin_max_depth - config_zapiola_rise_shelf_depth) *( & 
                  1.0_RKIND + tanh( ( (1.0_RKIND - distToCenterOfBasin**2/config_zapiola_rise_basin_radius**2) &
                           + config_zapiola_rise_shelf_offset)/config_zapiola_rise_shelf_scale) )


             depthZR = config_zapiola_rise_basin_max_depth - config_zapiola_rise_height &
                  *exp(- (distToCenterOfZR/(config_zapiola_rise_radius*degToRad*radiusEarth) )**2)

             bottomDepth(iCell)=min(min(depthShelf,depthZR),config_zapiola_rise_basin_max_depth)

             maxLevelCell(iCell) = nVertLevels
             do k = 1, nVertLevels
                if (refBottomDepth(k) >= bottomDepth(iCell)) then
                   maxLevelCell(iCell) = k
                   exit
                end if
             end do

             call ocn_alter_bottomDepth_for_pbcs(bottomDepth(iCell), refBottomDepth, maxLevelCell(iCell), iErr)

          else
             bottomDepth(iCell)= 0.0_RKIND
             maxLevelCell(iCell) = -1
          end if

          !--------------------------------------------------------------------
          ! Compute wind
          !--------------------------------------------------------------------

          distToCenterWindLat=radiusEarth*(latCell(iCell)-config_zapiola_rise_wind_center_lat*degToRad)

          windStressZonal(iCell)=config_zapiola_rise_ref_zonal_wind &
               *(cos(0.35*pii*distToCenterWindLat/config_zapiola_rise_basin_radius) &
               * (1.0_RKIND+4.0_RKIND*distToCenterWindLat/config_zapiola_rise_basin_radius) &
               * exp(-(distToCenterWindLat/config_zapiola_rise_basin_radius)**2) -0.5)

          windStressMeridional(iCell) = 0.0_RKIND

          !--------------------------------------------------------------------
          ! Compute temperature
          !--------------------------------------------------------------------

          ! Set temperature
          idx = index_temperature
          referenceTemperature=1.0_RKIND+tanh( (latCell(iCell)*radToDeg+40.0_RKIND)/10.0_RKIND )
          do k = 1, nVertLevels
             activeTracers(idx, k, iCell) = 1.0_RKIND+11.0_RKIND*referenceTemperature*exp(refZMid(k)/500.0_RKIND)
          end do

          ! Set salinity
          idx = index_salinity
          do k = 1, nVertLevels
             activeTracers(idx, k, iCell) = 35.0_RKIND
          end do

          ! Set layerThickness and restingThickness
          do k = 1, maxLevelCell(iCell) - 1
             layerThickness(k, iCell) = refLayerThickness(k)
          end do
          k = maxLevelCell(iCell)
          layerThickness(k, iCell) = bottomDepth(iCell) - refBottomDepth(k-1)
          restingThickness(:, iCell) = layerThickness(:, iCell)

       end do
    block_ptr => block_ptr % next
    end do

    deallocate(interfaceLocations)

    !--------------------------------------------------------------------
    ! Set restoring variables
    !--------------------------------------------------------------------
    block_ptr => domain % blocklist
    do while (associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)

       call ocn_mark_maxlevelcell(meshPool, iErr)
       block_ptr => block_ptr % next
    end do

    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh', meshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'state', statePool)
       call mpas_pool_get_subpool(block_ptr % structs, 'forcing', forcingPool)

       call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
       call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)

       call mpas_pool_get_subpool(forcingPool, 'tracersSurfaceRestoringFields', tracersSurfaceRestoringFieldsPool)
       call mpas_pool_get_subpool(forcingPool, 'tracersInteriorRestoringFields', tracersInteriorRestoringFieldsPool)
       call mpas_pool_get_array(tracersSurfaceRestoringFieldsPool, 'activeTracersSurfaceRestoringValue', activeTracersSurfaceRestoringValue, 1)
       call mpas_pool_get_array(tracersSurfaceRestoringFieldsPool, 'activeTracersPistonVelocity', activeTracersPistonVelocity, 1)
       call mpas_pool_get_array(tracersInteriorRestoringFieldsPool, 'activeTracersInteriorRestoringValue', activeTracersInteriorRestoringValue, 1)
       call mpas_pool_get_array(tracersInteriorRestoringFieldsPool, 'activeTracersInteriorRestoringRate', activeTracersInteriorRestoringRate, 1)

       ! set interior restoring values and rate
       if ( associated(activeTracersInteriorRestoringValue) .and. associated(activeTracers) ) then
          activeTracersInteriorRestoringValue(:, :, :) = activeTracers(:, :, :)
       end if

       if ( associated(activeTracersInteriorRestoringRate) ) then
          activeTracersInteriorRestoringRate(:, :, :) = config_zapiola_rise_interior_restore_rate
       end if

       ! set surface restoring values and rate
       if ( associated(activeTracersSurfaceRestoringValue) .and. associated(activeTracers) ) then
          activeTracersSurfaceRestoringValue(:, :) = activeTracers(:, 1, :)
       end if

       if ( associated(activeTracersPistonVelocity) ) then
          activeTracersPistonVelocity(:, :) = config_zapiola_rise_piston_velocity
       end if

       block_ptr => block_ptr % next
    end do

  end subroutine ocn_init_setup_zapiola_rise!}}}

!***********************************************************************
!
!  routine ocn_init_validate_zapiola_rise
!
!> \brief   Validation for this initial condition
!> \author  Mark Petersen
!> \date    December 2015
!> \details 
!>  This routine validates the configuration options for this case.
!
!-----------------------------------------------------------------------

   subroutine ocn_init_validate_zapiola_rise(configPool, packagePool, iocontext, iErr)!{{{

   !--------------------------------------------------------------------
      type (mpas_pool_type), intent(inout) :: configPool, packagePool
      type (mpas_io_context_type), intent(inout) :: iocontext

      integer, intent(out) :: iErr

      character (len=StrKIND), pointer :: config_init_configuration
      integer, pointer :: config_vert_levels, config_zapiola_rise_vert_levels

      iErr = 0

      call mpas_pool_get_config(configPool, 'config_init_configuration', config_init_configuration)

      if(config_init_configuration .ne. trim('zapiola_rise')) return

      call mpas_pool_get_config(configPool, 'config_vert_levels', config_vert_levels)
      call mpas_pool_get_config(configPool, 'config_zapiola_rise_vert_levels', config_zapiola_rise_vert_levels)

      if(config_vert_levels <= 0 .and. config_zapiola_rise_vert_levels > 0) then
         config_vert_levels = config_zapiola_rise_vert_levels
      else if (config_vert_levels <= 0) then
         write(stderrUnit,*) 'IERROR: Validation failed for zapiola_rise. Not given a usable value for vertical levels.'
         iErr = 1
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_init_validate_zapiola_rise!}}}


!***********************************************************************

end module ocn_init_zapiola_rise

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker

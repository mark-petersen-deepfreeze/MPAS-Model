! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_integration
!
!> \brief MPAS ocean time integration driver
!> \author Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date   September 2011
!> \details
!>  This module contains the main driver routine for calling
!>  the time integration scheme
!
!-----------------------------------------------------------------------

module ocn_time_integration

   use netcdf
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timekeeping
   use mpas_dmpar
   use mpas_vector_reconstruction
   use mpas_spline_interpolation
   use mpas_timer
   use mpas_log
   use mpas_kind_types
   use mpas_io_units
   use mpas_io
   use mpas_io_streams

   use ocn_constants
   use ocn_time_integration_rk4
   use ocn_time_integration_split
   use ocn_init_interpolation
   use ocn_interp_mat

   implicit none
   private
   save

   public :: ocn_timestep, &
             ocn_timestep_init

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

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

    logical :: rk4On, splitOn

    integer :: nLat_obs, nLon_obs, nVertLevels_obs, time, tick, month
    type (field1DReal) :: lat_obs, lon_obs, dep_obs
    type (field4DReal) :: temp_obs, psal_obs, cell_mask
    real (kind=RKIND), pointer :: daysSinceStartOfSim

    !integer :: nLat_obs, nLon_obs, nVertLevels_obs, nLat_interp, nLon_interp, nVertLevels_interp, &
    !           nLat_cell, nLon_cell, nLat_cell_obs, nLon_cell_obs!, nVertLevels_cell_obs, nVertLevels_cell !for bathymetry version only
    !type (field1DReal) :: lat_obs, lon_obs, iVertLevels_obs, & 
    !                      lat_interp, lon_interp, iVertLevels_interp, &
    !                      lat_cell, lon_cell, lat_cell_obs, lon_cell_obs
    !type (field4DReal) :: velocityZonal_obs_gridded, velocityZonal_interp_gridded, &
    !                      velocityMeridional_obs_gridded, velocityMeridional_interp_gridded, &
    !                      temperature_obs_gridded, temperature_interp_gridded, &
    !                      salinity_obs_gridded, salinity_interp_gridded

   contains

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_timestep
!
!> \brief MPAS ocean time integration driver
!> \author Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date   September 2011
!> \details
!>  This routine handles a single timestep for the ocean. It determines
!>  the time integrator that will be used for the run, and calls the
!>  appropriate one.
!
!-----------------------------------------------------------------------

   subroutine ocn_timestep(domain, itimestep, dt, timeStamp)!{{{
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! Advance model state forward in time by the specified time step
   !
   ! Input: domain - current model state in time level 1 (e.g., time_levs(1)state%h(:,:))
   !                 plus mesh meta-data
   ! Output: domain - upon exit, time level 2 (e.g., time_levs(2)%state%h(:,:)) contains
   !                  model state advanced forward in time by dt seconds
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (domain_type), intent(inout) :: domain
      real (kind=RKIND), intent(in) :: dt
      integer, intent(in) :: itimestep
      character(len=*), intent(in) :: timeStamp

      type (dm_info) :: dminfo
      type (block_type), pointer :: block

      type (mpas_pool_iterator_type) :: groupItr

      type (mpas_pool_type), pointer :: diagnosticsPool, statePool, meshPool, forcingPool, verticalMeshPool

      real (kind=RKIND) :: xfrac, yfrac, zfrac, lat_left_idx, lon_down_idx, lat_right_idx, lon_up_idx, dep_low_idx, dep_high_idx
      real (kind=RKIND) :: xfrac_M2O, yfrac_M2O, zfrac_M2O, bottom_left_M2O, top_left_M2O, bottom_right_M2O, top_right_M2O, dep_low_idx_M2O, dep_high_idx_M2O
      real (kind=RKIND) :: tolerance, distance_bottom_left, distance_bottom_right, distance_top_left, distance_top_right

      real (kind=RKIND), dimension(:,:), allocatable :: wgtMatrix, wgtMatrix_M2O
      real (kind=RKIND), dimension(:), pointer :: latCell, lonCell, cell_mask_interp
      real (kind=RKIND), dimension(:,:), pointer :: &
       velocityZonal_obs, velocityMeridional_obs, &
       velocityZonal_interp, velocityMeridional_interp
      real (kind=RKIND), dimension(:,:,:), pointer :: activeTracersAOTObservedData, activeTracersAOTInterpolatedData, activeTracers
      real (kind=RKIND), dimension(:,:,:,:), allocatable :: temp_mpas_interp_to_obsgrid, psal_mpas_interp_to_obsgrid

      type (mpas_pool_type), pointer :: tracersPool                             ! model data
      type (mpas_pool_type), pointer :: tracersAOTDataPool                      ! AOT observed data
      type (mpas_pool_type), pointer :: tracersAOTInterpDataPool                ! AOT interpolated data from previous time step

      character (len=StrKIND) :: modifiedGroupName, modifiedConfigName
      logical, pointer :: config_AOT_DA_on, config_use_activeTracers_AOT_data_assimilation 
      integer, pointer :: indexTemperature, indexSalinity, nEdges, nCells, nEdges_obs, nCells_obs, nVertLevels
      integer :: new_timestep, iErr, i, j, iCell, iVertLevels, iLat, iLon, iDep, daysinyear

      real (kind=RKIND), dimension(1,1) :: matrix

      character (len=StrKIND), pointer :: xtime
      character (len=StrKIND), pointer :: simulationStartTime
      type (MPAS_Time_type) :: xtime_timeType, simulationStartTime_timeType

      INTEGER, dimension(12) :: month_day
      INTEGER, PARAMETER :: mday(12)   &
                          = (/31,28,31,30,31,30,31,31,30,31,30,31/)
      INTEGER, PARAMETER :: mdayleap(12) &
                          = (/31,29,31,30,31,30,31,31,30,31,30,31/)
      INTEGER, DIMENSION(365) :: daym
      INTEGER, DIMENSION(366) :: daymleap


      iErr = 0
      block => domain % blocklist
      do while(associated(block))
        call mpas_pool_get_subpool(block % structs, 'state', statePool)
        call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
        call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
        call mpas_pool_get_array(meshPool, 'latCell', latCell)
        call mpas_pool_get_array(meshPool, 'lonCell', lonCell)

        call mpas_pool_get_array(diagnosticsPool, 'xtime', xtime)

        xtime = timeStamp

        matrix = 1.0

        call mpas_pool_get_array(diagnosticsPool, 'simulationStartTime', simulationStartTime)
        block => block % next
     end do

      call mpas_pool_get_config(ocnConfigs, 'config_AOT_DA_on', config_AOT_DA_on)
      !modifiedConfigName = 'config_use_' // trim(groupItr % memberName) // '_AOT_data_assimilation'
      !call mpas_pool_get_config(ocnConfigs, modifiedConfigName, config_use_tracerGroup_AOT_data_assimilation)
      call mpas_pool_get_config(ocnConfigs, 'config_use_activeTracers_AOT_data_assimilation', config_use_activeTracers_AOT_data_assimilation)

      if (itimestep .eq. 1 .and. config_AOT_DA_on) then
          call read_obs_grid_dims(domain, iErr)
          month = 1
          tick = 2 ! the dataset begins in 2002; adjust accordingly (easy reference: 2000 was a leap year)
      endif

     ! since the data is going to be coming in on a basis that is irregular, and
     ! the timing scheme is currently only written as characters, we will need
     ! to construct a timer here based on itimestep (there ought to be a nice
     ! way to generalize this, will use the ESMF timer to help us with this

     if (config_AOT_DA_on .and. itimestep .gt. 1) then 

         if (modulo(tick,4) == 0) then
            daysinyear = 366
            month_day = mdayleap
         else
            daysinyear = 365
            month_day = mday
         endif

         do i=1,11
            if (modulo(daysSinceStartOfSim,real(daysinyear, kind=RKIND)) .lt. month_day(1)) then
               month = 1
               call read_obs_temp_grid_fields_only(domain, tick, month, iErr)
               call read_obs_psal_grid_fields_only(domain, tick, month, iErr)
               exit
            elseif (modulo(daysSinceStartOfSim,real(daysinyear, kind=RKIND)) .gt. month_day(i) .and. modulo(daysSinceStartOfSim,real(daysinyear,kind=RKIND)) .le. month_day(i+1)) then 
               print*, 'in the if loop on second iteration?'
               month = i+1
               if (month == 12) tick = tick + 1
               call read_obs_temp_grid_fields_only(domain, tick, month, iErr)
               call read_obs_psal_grid_fields_only(domain, tick, month, iErr)
            endif
         enddo

     endif

     if (config_use_activeTracers_AOT_data_assimilation .and. config_AOT_DA_on .and. itimestep .gt. 1) then
         if (1==1) then !(modulo(daysSinceStartOfSim + 15.0, real(month_day(month), kind=RKIND)) == 0) then 
    
            ! note that the weight matrices have real entries, even for the indices, so
            ! the code will compile with the error Warning: Legacy Extension: REAL array
            ! index at (1), but this will not affect the run (except potentially in
            ! speed? will have to look in to this)
    
            block => domain % blocklist
            do while(associated(block))
                call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)
                call mpas_pool_get_subpool(block % structs, 'verticalMesh', verticalMeshPool)

                call mpas_pool_get_subpool(forcingPool, 'tracersAOTdata', tracersAOTDataPool)
                call mpas_pool_get_array(tracersAOTDataPool, "activeTracersAOTObservedData", & 
                                         activeTracersAOTObservedData)
                call mpas_pool_get_array(tracersAOTDataPool, "activeTracersAOTInterpolatedData", & 
                                         activeTracersAOTInterpolatedData)
                !call interp_ObsGrid_to_MPAS(meshPool, verticalMeshPool, lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, wgtMatrix, domain, iErr)

                ! interpolate MPAS to observed grid
                ! note: MAKE SURE the for loop is in the SAME ORDER as in the routine
                ! called
    
                call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
                if (rk4On) then
                   call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
                elseif (splitOn) then
                   call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 2)
                endif
    
                call mpas_pool_get_dimension(tracersPool, 'index_temperature', indexTemperature)
                call mpas_pool_get_dimension(tracersPool, 'index_salinity', indexSalinity)
                call interp_MPAS_to_ObsGrid(meshPool, verticalMeshPool, lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, wgtMatrix_M2O, domain, iErr)
    
                allocate ( temp_mpas_interp_to_obsgrid(1, nVertLevels_obs, nLat_obs, nLon_obs) )
                allocate ( psal_mpas_interp_to_obsgrid(1, nVertLevels_obs, nLat_obs, nLon_obs) )

                do iDep = 1,nVertLevels_obs
                   do iLat = 1, 2!nLat_obs
                      do iLon = 1,2!nLon_obs
                       !print*, minval(cell_mask % array), maxval(cell_mask % array) 
                       !print*, cell_mask % array (iLon, iLat, iDep, 1)
                       if (cell_mask % array (iLon, iLat, iDep, 1) == 0) then
                          temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = 1e34 ! fill_value 
                          psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = 1e34 ! fill_value 
                       else
                          xfrac_M2O        = wgtMatrix_M2O(iLat+iLon+iDep-2, 1)
                          yfrac_M2O        = wgtMatrix_M2O(iLat+iLon+iDep-2, 2)
                          zfrac_M2O        = wgtMatrix_M2O(iLat+iLon+iDep-2, 3)
                          bottom_left_M2O  = wgtMatrix_M2O(iLat+iLon+iDep-2, 4)
                          top_left_M2O     = wgtMatrix_M2O(iLat+iLon+iDep-2, 5)
                          bottom_right_M2O = wgtMatrix_M2O(iLat+iLon+iDep-2, 6)
                          top_right_M2O    = wgtMatrix_M2O(iLat+iLon+iDep-2, 7)
                          dep_low_idx_M2O  = wgtMatrix_M2O(iLat+iLon+iDep-2, 8)
                          dep_high_idx_M2O = wgtMatrix_M2O(iLat+iLon+iDep-2, 9)
                          distance_bottom_left  = wgtMatrix_M2O(iLat+iLon+iDep-2, 10)
                          distance_bottom_right = wgtMatrix_M2O(iLat+iLon+iDep-2, 11)
                          distance_top_left     = wgtMatrix_M2O(iLat+iLon+iDep-2, 12)
                          distance_top_right    = wgtMatrix_M2O(iLat+iLon+iDep-2, 13)
   
                          print*, lat_obs % array(iLat), lon_obs % array (iLon)
                          print*, latCell(bottom_left_M2O), lonCell(bottom_left_M2O)
                          print*, latCell(top_left_M2O), lonCell(top_left_M2O)
                          print*, latCell(top_right_M2O), lonCell(top_right_M2O)
                          print*, latCell(bottom_right_M2O), lonCell(bottom_right_M2O)
    
                          tolerance = 240 !how wide is an MPAS grid cell
                          print*, distance_bottom_left, distance_top_left, distance_top_right, distance_bottom_right
    
                          !  we can use a cell mask for observed data because the
                          !  latitude and longitude are ordered sets, but the
                          !  ordering is random for the MPAS mesh, and is wildly
                          !  different between the base and culled meshes.  The
                          !  culled mesh is what is used to evolve the physics. So
                          !  this is why we have to do the interpolation the hard
                          !  way.
    
                            if (distance_bottom_left .ge. tolerance .and. distance_top_left .ge. tolerance .and. distance_bottom_right .ge. tolerance) then
    
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexTemperature, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   zfrac_M2O * activeTracers(indexTemperature, dep_high_idx_M2O, top_right_M2O)
    
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexSalinity, dep_low_idx_M2O, top_right_M2O) + &
                                                                                  zfrac_M2O * activeTracers(indexSalinity, dep_high_idx_M2O, top_right_M2O)
    
                                print*, 'top right good only'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_bottom_left .ge. tolerance .and. distance_top_right .ge. tolerance .and. distance_bottom_right .ge. tolerance) then
    
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexTemperature, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   zfrac_M2O * activeTracers(indexTemperature, dep_high_idx_M2O, top_left_M2O)
    
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexSalinity, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   zfrac_M2O * activeTracers(indexSalinity, dep_high_idx_M2O, top_left_M2O)
    
                                print*, 'top left good only'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_top_left .ge. tolerance .and. distance_top_right .ge. tolerance .and. distance_bottom_right .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexTemperature, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   zfrac_M2O * activeTracers(indexTemperature, dep_high_idx_M2O, bottom_left_M2O)
    
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexSalinity, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   zfrac_M2O * activeTracers(indexSalinity, dep_high_idx_M2O, bottom_left_M2O)
    
                                print*, 'bottom left good only'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_top_left .ge. tolerance .and. distance_top_right .ge. tolerance .and. distance_bottom_left .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexTemperature, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   zfrac_M2O * activeTracers(indexTemperature, dep_high_idx_M2O, bottom_right_M2O)
    
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-zfrac_M2O) * activeTracers(indexSalinity, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   zfrac_M2O * activeTracers(indexSalinity, dep_high_idx_M2O, bottom_right_M2O)
    
                                print*, 'bottom right good only'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_bottom_left .ge. tolerance .and. distance_bottom_right .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_left_M2O) + &
                                                                                   (xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_right_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_left_M2O) + &
                                                                                   (xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_right_M2O)
    
                                print*, 'average top two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_top_left .ge. tolerance .and. distance_top_right .ge. tolerance) then 
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_right_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (xfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (xfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_right_M2O)
    
                                print*, 'average bottom two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_top_left .ge. tolerance .and. distance_bottom_left .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_right_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_right_M2O)
    
                                print*, 'average right two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_top_right .ge. tolerance .and. distance_bottom_right .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_left_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_left_M2O)
    
                                print*, 'average left two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_top_right .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_left_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_left_M2O)
    
                                print*, 'top_right not viable, average left two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_bottom_right .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_left_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_left_M2O)
    
                                print*, 'bottom_right not viable, average left two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_top_left .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_right_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_right_M2O)
    
                                print*, 'top_left not viable, average right two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            elseif (distance_bottom_left .ge. tolerance) then
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_right_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_right_M2O)
    
                                print*, 'bottom_left not viable, average right two nearest'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
    
                            else
                                temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-xfrac_M2O)*(1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_left_M2O) + &
                                                                                   (xfrac_M2O)*(1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (xfrac_M2O)*(1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (xfrac_M2O)*(yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (xfrac_M2O)*(yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexTemperature, dep_high_idx_M2O, top_right_M2O)
        
                                psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon) = (1-xfrac_M2O)*(1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_left_M2O) + &
                                                                                   (1-xfrac_M2O)*(yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_left_M2O) + &
                                                                                   (xfrac_M2O)*(1-yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, bottom_right_M2O) + &
                                                                                   (xfrac_M2O)*(1-yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, bottom_right_M2O) + &
                                                                                   (xfrac_M2O)*(yfrac_M2O)*(1-zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_low_idx_M2O, top_right_M2O) + &
                                                                                   (xfrac_M2O)*(yfrac_M2O)*(zfrac_M2O)*&
                                                                                   activeTracers(indexSalinity, dep_high_idx_M2O, top_right_M2O)
    
                                print*, 'all viable, interpolate as usual'
                                print*, temp_mpas_interp_to_obsgrid(1, iDep, iLat, iLon), psal_mpas_interp_to_obsgrid(1, iDep, iLat, iLon)
                            endif
                       endif
    
                            print*, latCell(-1)
    
                      enddo
                   enddo
                enddo 

    
                call interp_ObsGrid_to_MPAS(meshPool, verticalMeshPool, lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, wgtMatrix, domain, iErr)
                ! interpolate observed data  and MPAS interpolated data back to MPAS grid
                do iVertLevels = 1,nVertLevels
                   do iCell = 1,nCells
                      xfrac         = wgtMatrix(iCell + iVertLevels - 1,1)
                      yfrac         = wgtMatrix(iCell + iVertLevels - 1,2)
                      zfrac         = wgtMatrix(iCell + iVertLevels - 1,3)
                      lat_left_idx  = wgtMatrix(iCell + iVertLevels - 1,4)
                      lat_right_idx = wgtMatrix(iCell + iVertLevels - 1,5)
                      lon_down_idx  = wgtMatrix(iCell + iVertLevels - 1,6)
                      lon_up_idx    = wgtMatrix(iCell + iVertLevels - 1,7)
                      dep_low_idx   = wgtMatrix(iCell + iVertLevels - 1,8)
                      dep_high_idx  = wgtMatrix(iCell + iVertLevels - 1,9)
                      
                      activeTracersAOTObservedData(1,iVertLevels, iCell) = (1-xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                           temp_obs % array (lat_left_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                       (1-xfrac)*(1-yfrac)*(zfrac)*&
                                                                       temp_obs % array (lat_left_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                       (1-xfrac)*(yfrac)*(1-zfrac)*&
                                                                       temp_obs % array (lat_left_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                       (1-xfrac)*(yfrac)*(zfrac)*&
                                                                       temp_obs % array (lat_left_idx, lon_up_idx, dep_high_idx, 1) + &
                                                                       (xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                       temp_obs % array (lat_right_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                       (xfrac)*(1-yfrac)*(zfrac)*&
                                                                       temp_obs % array (lat_right_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                       (xfrac)*(yfrac)*(1-zfrac)*&
                                                                       temp_obs % array (lat_right_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                       (xfrac)*(yfrac)*(zfrac)*&
                                                                       temp_obs % array (lat_right_idx, lon_up_idx, dep_high_idx, 1)
    
                      activeTracersAOTObservedData(2,iVertLevels, iCell) = (1-xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                           psal_obs % array (lat_left_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                       (1-xfrac)*(1-yfrac)*(zfrac)*&
                                                                       psal_obs % array (lat_left_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                       (1-xfrac)*(yfrac)*(1-zfrac)*&
                                                                       psal_obs % array (lat_left_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                       (1-xfrac)*(yfrac)*(zfrac)*&
                                                                       psal_obs % array (lat_left_idx, lon_up_idx, dep_high_idx, 1) + &
                                                                       (xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                       psal_obs % array (lat_right_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                       (xfrac)*(1-yfrac)*(zfrac)*&
                                                                       psal_obs % array (lat_right_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                       (xfrac)*(yfrac)*(1-zfrac)*&
                                                                       psal_obs % array (lat_right_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                       (xfrac)*(yfrac)*(zfrac)*&
                                                                       psal_obs % array (lat_right_idx, lon_up_idx, dep_high_idx, 1)
    
                      activeTracersAOTInterpolatedData(1,iVertLevels, iCell) = (1-xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                               temp_mpas_interp_to_obsgrid(lat_left_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                           (1-xfrac)*(1-yfrac)*(zfrac)*&
                                                                           temp_mpas_interp_to_obsgrid(lat_left_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                           (1-xfrac)*(yfrac)*(1-zfrac)*&
                                                                           temp_mpas_interp_to_obsgrid(lat_left_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                           (1-xfrac)*(yfrac)*(zfrac)*&
                                                                           temp_mpas_interp_to_obsgrid(lat_left_idx, lon_up_idx, dep_high_idx, 1) + &
                                                                           (xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                           temp_mpas_interp_to_obsgrid(lat_right_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                           (xfrac)*(1-yfrac)*(zfrac)*&
                                                                           temp_mpas_interp_to_obsgrid(lat_right_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                           (xfrac)*(yfrac)*(1-zfrac)*&
                                                                           temp_mpas_interp_to_obsgrid(lat_right_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                           (xfrac)*(yfrac)*(zfrac)*&
                                                                           temp_mpas_interp_to_obsgrid(lat_right_idx, lon_up_idx, dep_high_idx, 1)
    
                      activeTracersAOTInterpolatedData(2,iVertLevels, iCell) = (1-xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                           psal_mpas_interp_to_obsgrid(lat_left_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                       (1-xfrac)*(1-yfrac)*(zfrac)*&
                                                                       psal_mpas_interp_to_obsgrid(lat_left_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                       (1-xfrac)*(yfrac)*(1-zfrac)*&
                                                                       psal_mpas_interp_to_obsgrid(lat_left_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                       (1-xfrac)*(yfrac)*(zfrac)*&
                                                                       psal_mpas_interp_to_obsgrid(lat_left_idx, lon_up_idx, dep_high_idx, 1) + &
                                                                       (xfrac)*(1-yfrac)*(1-zfrac)*&
                                                                       psal_mpas_interp_to_obsgrid(lat_right_idx, lon_down_idx, dep_low_idx, 1) + &
                                                                       (xfrac)*(1-yfrac)*(zfrac)*&
                                                                       psal_mpas_interp_to_obsgrid(lat_right_idx, lon_down_idx, dep_high_idx, 1) + &
                                                                       (xfrac)*(yfrac)*(1-zfrac)*&
                                                                       psal_mpas_interp_to_obsgrid(lat_right_idx, lon_up_idx, dep_low_idx, 1) + &
                                                                       (xfrac)*(yfrac)*(zfrac)*&
                                                                       psal_mpas_interp_to_obsgrid(lat_right_idx, lon_up_idx, dep_high_idx, 1)
                   enddo
                enddo
            block => block % next
            enddo
                  
         endif
     endif

      if (rk4On) then
         call ocn_time_integrator_rk4(domain, itimestep, dt)
      elseif (splitOn) then
         call ocn_time_integrator_split(domain, itimestep, dt)
     endif

     block => domain % blocklist
     do while (associated(block))

        ! compute time since start of simulation, in days
        call mpas_pool_get_array(diagnosticsPool, 'daysSinceStartOfSim',daysSinceStartOfSim)
        call mpas_set_time(xtime_timeType, dateTimeString=xtime)
        call mpas_set_time(simulationStartTime_timeType, dateTimeString=simulationStartTime)
        call mpas_get_timeInterval(xtime_timeType - simulationStartTime_timeType,dt=daysSinceStartOfSim)

        !$omp single
        daysSinceStartOfSim = daysSinceStartOfSim*days_per_second
        !$omp end single

        block => block % next
     end do

   end subroutine ocn_timestep!}}}

   subroutine ocn_timestep_init(iocontext, err)!{{{

      !type (mpas_pool_type), intent(in) :: configPool, packagePool
      type (mpas_io_context_type), intent(inout), target :: iocontext

      type (mpas_io_context_type), pointer :: iocontext_ptr

      type (MPAS_IO_Handle_type) :: input_obs_File, input_interp_File

      integer, intent(out) :: err

      character (len=StrKIND), pointer :: config_time_integrator
      character(len=StrKIND), pointer :: config_input_obs_file ! should be "ISAS15_ARGO_20", with a prefix for the correct directory
      character(len=StrKIND), target :: config_input_obs_file_full 
      character(len=StrKIND), pointer :: config_input_obs_file_full_ptr 
      logical, pointer :: config_AOT_DA_on

      err = 0

      iocontext_ptr => iocontext

      call mpas_pool_get_config(ocnConfigs, 'config_AOT_DA_on', config_AOT_DA_on)
      call mpas_pool_get_config(ocnConfigs, 'config_time_integrator', config_time_integrator)


      if (config_AOT_DA_on) then 
          call mpas_pool_get_config(ocnConfigs, 'config_input_psal_obs_file', config_input_obs_file)

          config_input_obs_file_full = trim(config_input_obs_file) // "150115_fld_PSAL.nc"  ! this just has to be one of the files that has the total dimensions in it, which should be uniform across all files you are assimilating; if it's not you'll have to edit the implementation altogether
          config_input_obs_file_full_ptr => config_input_obs_file_full
 
          input_obs_File = MPAS_io_open(config_input_obs_file_full_ptr, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=err)
          if(err > 0) then
             call mpas_log_write( 'Validation failed. Could not open file:'// &
                  trim(config_input_obs_file_full_ptr) , MPAS_LOG_CRIT)
             return
          end if
     
          call MPAS_io_inq_dim(input_obs_File, 'latitude', nLat_obs, err)
          if(err /= 0) then
             call mpas_log_write( 'Validation failed. Could not find dim latitude in file:'// &
                trim(config_input_obs_file_full_ptr) , MPAS_LOG_CRIT)
             return
          end if
    
          call MPAS_io_inq_dim(input_obs_File, 'longitude', nLon_obs, err)
          if(err /= 0) then
             call mpas_log_write( 'Validation failed. Could not find dim longitude in file:'// &
                trim(config_input_obs_file_full_ptr) , MPAS_LOG_CRIT)
             return
          end if
    
          call MPAS_io_inq_dim(input_obs_File, 'depth', nVertLevels_obs, err)
          if(err /= 0) then
             call mpas_log_write( 'Validation failed. Could not find dim depth in file:'// &
                trim(config_input_obs_file_full_ptr) , MPAS_LOG_CRIT)
             return
          end if
      endif

      rk4On = .false.
      splitOn = .false.

      if (trim(config_time_integrator) == 'RK4') then
          rk4On = .true.
      elseif (trim(config_time_integrator) == 'split_explicit' &
          .or.trim(config_time_integrator) == 'unsplit_explicit') then
          splitOn = .true.
      else
          err = 1
          call mpas_log_write('Incorrect choice for config_time_integrator:' // trim(config_time_integrator) // &
             '   choices are: RK4, split_explicit, unsplit_explicit', MPAS_LOG_CRIT)
      endif

   end subroutine ocn_timestep_init!}}}

  subroutine check(status)
    integer, intent ( in) :: status

    if(status /= nf90_noerr) then
      print *, trim(nf90_strerror(status))
      stop "Stopped"
    end if
  end subroutine check

subroutine read_obs_temp_grid_fields_only(domain, i, j, iErr)

      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      integer, intent(in) :: i, j

      type (MPAS_Stream_type) :: tempObsDataStream
      character(len=ShortStrKIND) :: tmp_int1, tmp_int2, tail_char
      character(len=StrKIND), pointer :: config_input_obs_file  ! first one here should be "ISAS15_ARGO_20"
      character(len=DoubleStrKIND) :: config_input_obs_file_iterance 

      type (MPAS_IO_Handle_type) :: input_obs_File

      iErr = 0

      call mpas_pool_get_config(domain % configs, 'config_input_temp_obs_file', config_input_obs_file)

      if (trim(config_input_obs_file) == 'none') then
         call mpas_log_write( 'Validation failed. Invalid filename for config_input_obs_file', MPAS_LOG_CRIT)
         iErr = 1
         return
      end if

     if (i .lt. 10) then
        write(tmp_int1, "(I1, I1)") 0, i ! this should be the last two digits of the year, e.g. 15 for 2015
     else
        write(tmp_int1, "(I2)") i 
     endif 

     if (j .lt. 10) then
        write(tmp_int2, "(I1, I1)") 0, j ! this should be the two-digit month, e.g., january is 01
     else
        write(tmp_int2, "(I2)") j 
     endif 

      tail_char = "15_fld_TEMP.nc"
      config_input_obs_file_iterance = trim(config_input_obs_file) // trim(tmp_int1) // trim(tmp_int2) // trim(tail_char)

      call MPAS_createStream(tempObsDataStream, domain % iocontext, config_input_obs_file_iterance, &
           MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)

       temp_obs % fieldName = 'TEMP'
       temp_obs % dimSizes(4) = 1 ! each file has time dim 1 for ARGO data
       temp_obs % dimSizes(3) = nVertLevels_obs
       temp_obs % dimSizes(2) = nLat_obs
       temp_obs % dimSizes(1) = nLon_obs
       temp_obs % dimNames(4) = 'time'
       temp_obs % dimNames(3) = 'depth'
       temp_obs % dimNames(2) = 'latitude'
       temp_obs % dimNames(1) = 'longitude'
       temp_obs % isVarArray = .false.
       temp_obs % isPersistent = .true.
       temp_obs % isActive = .true.
       temp_obs % hasTimeDimension = .true.
       temp_obs % block => domain % blocklist
       allocate(temp_obs % attLists(1))
       allocate(temp_obs % array(nLon_obs, nLat_obs, nVertLevels_obs, 1))

       call MPAS_streamAddField(tempObsDataStream, temp_obs, iErr)

       call MPAS_readStream(tempObsDataStream, 1, iErr)
       call MPAS_closeStream(tempObsDataStream)

       temp_obs % array = 20.0 + 0.001 * temp_obs % array ! this is important because our data has an offset and scaling parameter

end subroutine read_obs_temp_grid_fields_only

subroutine read_obs_psal_grid_fields_only(domain, i, j, iErr)

      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      integer, intent(in) :: i, j

      type (MPAS_Stream_type) :: psalObsDataStream
      character(len=ShortStrKIND) :: tmp_int1, tmp_int2, tail_char
      character(len=StrKIND), pointer :: config_input_obs_file  ! first one here should be "ISAS15_ARGO_20"
      character(len=DoubleStrKIND) :: config_input_obs_file_iterance 

      type (MPAS_IO_Handle_type) :: input_obs_File

      iErr = 0

      call mpas_pool_get_config(domain % configs, 'config_input_psal_obs_file', config_input_obs_file)

      if (trim(config_input_obs_file) == 'none') then
         call mpas_log_write( 'Validation failed. Invalid filename for config_input_obs_file', MPAS_LOG_CRIT)
         iErr = 1
         return
      end if

     if (i .lt. 10) then
        write(tmp_int1, "(I1, I1)") 0, i ! this should be the last two digits of the year, e.g. 15 for 2015
     else
        write(tmp_int1, "(I2)") i 
     endif 

     if (j .lt. 10) then
        write(tmp_int2, "(I1, I1)") 0, j ! this should be the two-digit month, e.g., january is 01
     else
        write(tmp_int2, "(I2)") j 
     endif 

      tail_char = "15_fld_PSAL.nc"
      config_input_obs_file_iterance = trim(config_input_obs_file) // trim(tmp_int1) // trim(tmp_int2) // trim(tail_char)

      call MPAS_createStream(psalObsDataStream, domain % iocontext, config_input_obs_file_iterance, &
           MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)

       psal_obs % fieldName = 'PSAL'
       psal_obs % dimSizes(4) = 1 ! each file has time dim 1 for ARGO data
       psal_obs % dimSizes(3) = nVertLevels_obs
       psal_obs % dimSizes(2) = nLat_obs
       psal_obs % dimSizes(1) = nLon_obs
       psal_obs % dimNames(4) = 'time'
       psal_obs % dimNames(3) = 'depth'
       psal_obs % dimNames(2) = 'latitude'
       psal_obs % dimNames(1) = 'longitude'
       psal_obs % isVarArray = .false.
       psal_obs % isPersistent = .true.
       psal_obs % isActive = .true.
       psal_obs % hasTimeDimension = .true.
       psal_obs % block => domain % blocklist
       allocate(psal_obs % attLists(1))
       allocate(psal_obs % array(nLon_obs, nLat_obs, nVertLevels_obs, 1))

       call MPAS_streamAddField(psalObsDataStream, psal_obs, iErr)

       call MPAS_readStream(psalObsDataStream, 1, iErr)
       call MPAS_closeStream(psalObsDataStream)

       psal_obs % array = 30.0 + 0.001 * psal_obs % array ! this is important because our data has an offset and scaling parameter

end subroutine read_obs_psal_grid_fields_only

subroutine read_obs_grid_dims(domain, iErr)

!! this subroutine should only be called ONCE (or however many times
!! the dims of obs files changes, which would necessitate editting 
!! the rest of the code)

      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      integer :: iLon

      type (block_type), pointer :: block_ptr

      type (MPAS_Stream_type) :: obsDataStream
      character(len=StrKIND), pointer :: config_input_obs_file_full_ptr ! should be "ISAS15_ARGO_20", with a prefix for the correct directory
      !character(len=91), pointer :: testme / '/lustre/scratch4/turquoise/ecarlson10/climate_data/all_years/ISAS15_DM_20150115_fld_PSAL.nc' /

      type (MPAS_IO_Handle_type) :: input_obs_File!, input_interp_File, cell_mask_File, cell_mask_obs_File

      iErr = 0

      call mpas_pool_get_config(domain % configs, 'config_input_cell_mask_obs_file', config_input_obs_file_full_ptr)


      if (trim(config_input_obs_file_full_ptr) == 'none') then
         call mpas_log_write( 'Validation failed. Invalid filename for config_input_obs_file', MPAS_LOG_CRIT)
         iErr = 1
         return
      end if


      !call MPAS_createStream(obsDataStream, domain % iocontext, testme, &
      !     MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)
      call MPAS_createStream(obsDataStream, domain % iocontext, config_input_obs_file_full_ptr, &
           MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)

      !! in degrees !!
      lat_obs % fieldName        = 'latitude'
      lat_obs % dimSizes(1)      =  nLat_obs
      lat_obs % dimNames(1)      = 'latitude'
      lat_obs % isVarArray       = .false.
      lat_obs % isPersistent     = .true.
      lat_obs % isActive         = .true.
      lat_obs % hasTimeDimension = .false.
      lat_obs % block => domain % blocklist
      allocate(lat_obs % attLists(1))
      allocate(lat_obs % array(nLat_obs))

      !! in degrees !!
      lon_obs % fieldName        = 'longitude'
      lon_obs % dimSizes(1)      =  nLon_obs
      lon_obs % dimNames(1)      = 'longitude'
      lon_obs % isVarArray       = .false.
      lon_obs % isPersistent     = .true.
      lon_obs % isActive         = .true.
      lon_obs % hasTimeDimension = .false.
      lon_obs % block => domain % blocklist
      allocate(lon_obs % attLists(1))
      allocate(lon_obs % array(nLon_obs))

      !! in meters !!
      dep_obs % fieldName        = 'depth'
      dep_obs % dimSizes(1)      =  nVertLevels_obs
      dep_obs % dimNames(1)      = 'depth'
      dep_obs % isVarArray       = .false.
      dep_obs % isPersistent     = .true.
      dep_obs % isActive         = .true.
      dep_obs % hasTimeDimension = .false.
      dep_obs % block => domain % blocklist
      allocate(dep_obs % attLists(1))
      allocate(dep_obs % array(nVertLevels_obs))

      cell_mask % fieldName      = 'cell_mask'
      cell_mask % dimSizes(4)    = 1
      cell_mask % dimSizes(3)    = nVertLevels_obs
      cell_mask % dimSizes(2)    = nLat_obs
      cell_mask % dimSizes(1)    = nLon_obs
      cell_mask % dimNames(4)    = 'time'
      cell_mask % dimNames(3)    = 'depth'
      cell_mask % dimNames(2)    = 'latitude'
      cell_mask % dimNames(1)    = 'longitude'
      cell_mask % isVarArray       = .false.
      cell_mask % isPersistent     = .true.
      cell_mask % isActive         = .true.
      cell_mask % hasTimeDimension = .true.
      cell_mask % block            => domain % blocklist
      allocate(cell_mask % attLists(1))
      allocate(cell_mask % array(nLon_obs, nLat_obs, nVertLevels_obs, 1))
       

      call MPAS_streamAddField(obsDataStream, lat_obs, iErr)
      call MPAS_streamAddField(obsDataStream, lon_obs, iErr)
      call MPAS_streamAddField(obsDataStream, dep_obs, iErr)
      call MPAS_streamAddField(obsDataStream, cell_mask, iErr)

      call MPAS_readStream(obsDataStream, 1, iErr)
      call MPAS_closeStream(obsDataStream)

      lat_obs % array = lat_obs % array * 4*ATAN(1.d0)/180.
      lon_obs % array = lon_obs % array * 4*ATAN(1.d0)/180.

      do iLon = 1, nLon_obs
         if (lon_obs % array (iLon) .lt. 0) then
            lon_obs % array (iLon) = lon_obs % array (iLon) + 2*4*ATAN(1.d0)
         endif
      enddo

end subroutine read_obs_grid_dims

end module ocn_time_integration

! vim: foldmethod=marker

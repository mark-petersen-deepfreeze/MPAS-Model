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

    integer :: nLat_obs, nLon_obs, nVertLevels_obs, nLat_interp, nLon_interp, nVertLevels_interp, &
               nLat_cell, nLon_cell, nVertLevels_cell, nLat_cell_obs, nLon_cell_obs, nVertLevels_cell_obs
    type (field1DReal) :: lat_obs, lon_obs, iVertLevels_obs, & 
                          lat_interp, lon_interp, iVertLevels_interp, &
                          lat_cell, lon_cell, lat_cell_obs, lon_cell_obs
    type (field4DReal) :: velocityZonal_obs_gridded, velocityZonal_interp_gridded, &
                          velocityMeridional_obs_gridded, velocityMeridional_interp_gridded, &
                          temperature_obs_gridded, temperature_interp_gridded, &
                          salinity_obs_gridded, salinity_interp_gridded
   type (field3DReal) :: cell_mask, cell_mask_obs

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

      type (mpas_pool_type), pointer :: diagnosticsPool, statePool, meshPool, forcingPool

      real (kind=RKIND), dimension(:), pointer :: latCell, lonCell, cell_mask_interp
      real (kind=RKIND), dimension(:,:), pointer :: &
       velocityZonal_obs, velocityMeridional_obs, &
       velocityZonal_interp, velocityMeridional_interp
      real (kind=RKIND), dimension(:,:,:), pointer :: activeTracersAOTObservedData, activeTracersAOTInterpolatedData

      type (mpas_pool_type), pointer :: tracersAOTDataPool                      ! AOT observed data
      type (mpas_pool_type), pointer :: tracersAOTInterpDataPool                ! AOT interpolated data from previous time step

      character (len=StrKIND) :: modifiedGroupName, modifiedConfigName
      logical, pointer :: config_AOT_DA_on, config_use_activeTracers_AOT_data_assimilation 
      integer, pointer :: indexTemperature, indexSalinity, nEdges, nCells, nEdges_obs, nCells_obs, nVertLevels
      integer :: new_timestep, iErr

      real (kind=RKIND), dimension(1,1) :: matrix

      character (len=StrKIND), pointer :: xtime
      real (kind=RKIND), pointer :: daysSinceStartOfSim
      character (len=StrKIND), pointer :: simulationStartTime
      type (MPAS_Time_type) :: xtime_timeType, simulationStartTime_timeType

      iErr = 0
      block => domain % blocklist
      do while(associated(block))
        call mpas_pool_get_subpool(block % structs, 'state', statePool)
        call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
        call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)

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

     if (config_AOT_DA_on) then! .and. itimestep .ge. 2) then ! this part needed only for first time
          !write(1111,*) itimestep
          ! write(11111,*), itimestep
          !call mpas_pool_get_dimension(meshPool, 'nCells', nCells)

        !!!!!!! TESTING INTERPOLATOR !!!!!!!!!!!
       !block => domain % blocklist
       !do while(associated(block))
       !   call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)
       !   call mpas_pool_get_array(meshPool, 'latCell', latCell)
       !   call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
       !   call mpas_pool_get_dimension(meshPool, 'nCells', nCells)

       !   call mpas_pool_get_array(diagnosticsPool, 'cell_mask_interp', cell_mask_interp)
       !   !write(1000,*) shape(cell_mask_interp), nCells

       !   call ocn_init_interpolation_bilinear_horiz(lon_cell % array, lat_cell % array, &
       !                                              cell_mask % array, nLon_cell, nLat_cell, &
       !                                              lonCell, latCell, cell_mask_interp, nCells, cell_mask % array)
       !   block => block % next
       !end do
       if (itimestep .eq. 1) then !2) then ! this part needed only for first time step
          call ocn_read_cell_mask(domain, iErr)
          call ocn_read_cell_mask_obs(domain,iErr)
          call ocn_AOT_read_obs_data(domain,iErr) 
       endif

       block => domain % blocklist
       do while(associated(block))
          call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
          call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)
          call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
          call mpas_pool_get_subpool(block % structs, 'state', statePool)

          call mpas_pool_get_array(meshPool, 'latCell', latCell)
          call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
          call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
          
          call mpas_pool_get_array(forcingPool, 'velocityZonal_obs', velocityZonal_obs)
          call mpas_pool_get_array(forcingPool, 'velocityMeridional_obs', velocityMeridional_obs)

          !write(123451,*), minval(velocityZonal_obs_gridded % array), maxval(velocityZonal_obs_gridded % array)
          !write(12345,*), velocityZonal_obs_gridded % array (16,16,1,1)!itimestep)
          !write(123451,*), velocityZonal_obs_gridded % array (16,16,1,itimestep)
          !write(123451,*), velocityZonal_obs_gridded % array (16,16,2,1)!itimestep)
          !write(12346,*), velocityMeridional_obs_gridded % array (16,16,1,itimestep)
          !write(123461,*), velocityMeridional_obs_gridded % array (16,16,2,itimestep)
          !write(12347,*), temperature_obs_gridded % array (16,16,1,itimestep)
          !write(123471,*), temperature_obs_gridded % array (16,16,2,itimestep)
          !write(12348,*), salinity_obs_gridded % array (16,16,1,itimestep)
          !write(123481,*), salinity_obs_gridded % array (16,16,2,itimestep)

          !write(123496,*), shape(velocityZonal_obs_gridded % array)
          !write(123497,*), shape(velocityMeridional_obs_gridded % array)
          !write(123498,*), shape(temperature_obs_gridded % array)
          !write(123499,*), shape(salinity_obs_gridded % array)

          call ocn_init_interpolation_bilinear_horiz(lon_obs % array, lat_obs % array, &
                                                     velocityZonal_obs_gridded % array (:,:,:,itimestep) , nLon_obs, nLat_obs, &
                                                     lonCell, latCell, velocityZonal_obs, nCells, cell_mask_obs % array)
          
          call ocn_init_interpolation_bilinear_horiz(lon_obs % array, lat_obs % array, &
                                                     velocityMeridional_obs_gridded % array (:,:,:,itimestep) , nLon_obs, nLat_obs, &
                                                     lonCell, latCell, velocityMeridional_obs, nCells, cell_mask_obs % array)
 
          if (config_use_activeTracers_AOT_data_assimilation) then

               call mpas_pool_get_subpool(forcingPool, 'tracersAOTdata', tracersAOTDataPool)
               call mpas_pool_get_array(tracersAOTDataPool, "activeTracersAOTObservedData", &
                                        activeTracersAOTObservedData)

            call ocn_init_interpolation_bilinear_horiz(lon_obs % array, lat_obs % array, &
                                                       temperature_obs_gridded % array (:,:,:,itimestep) , nLon_obs, nLat_obs, &
                                                       lonCell, latCell, activeTracersAOTObservedData(1,:,:), nCells, cell_mask_obs % array)
            
            call ocn_init_interpolation_bilinear_horiz(lon_obs % array, lat_obs % array, &
                                                                  salinity_obs_gridded % array (:,:,:,itimestep) , nLon_obs, nLat_obs, &
                                                                  lonCell, latCell, activeTracersAOTObservedData(2,:,:), nCells, cell_mask_obs % array)
          endif

          block => block % next
       end do

          new_timestep = itimestep-1
         
          call ocn_AOT_read_interp_data(domain, 1, iErr)

          !write(123496,*), maxval(velocityZonal_interp_gridded % array), minval(velocityZonal_interp_gridded % array)
          !write(123496,*), velocityZonal_interp_gridded % array(16,16,1,1)
          !write(123497,*), velocityMeridional_interp_gridded % array(16,16,1,1) !maxval(velocityMeridional_interp_gridded % array), minval(velocityMeridional_interp_gridded % array)
          !write(123498,*), temperature_interp_gridded % array(16,16,1,1)!maxval(temperature_interp_gridded % array), minval(temperature_interp_gridded % array)
          !write(123499,*), salinity_interp_gridded % array(16,16,1,1) !maxval(salinity_interp_gridded % array), minval(salinity_interp_gridded % array)
       block => domain % blocklist
       do while(associated(block))
          call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
          call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)
          call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
          call mpas_pool_get_subpool(block % structs, 'state', statePool)

          call mpas_pool_get_array(meshPool, 'latCell', latCell)
          call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
          call mpas_pool_get_dimension(meshPool, 'nCells', nCells)

          call mpas_pool_get_array(forcingPool, 'velocityZonal_interp', velocityZonal_interp)
          call mpas_pool_get_array(forcingPool, 'velocityMeridional_interp', velocityMeridional_interp)

          !write(43214321,*), shape(velocityZonal_interp), shape(velocityZonal_obs)

            !write(43215,*), itimestep,  new_timestep, velocityZonal_interp_gridded % array (16,16,1,1)
            !write(432151,*), new_timestep, velocityZonal_interp_gridded % array (16,16,3,new_timestep)
            !write(43216,*), velocityMeridional_interp_gridded % array (16,16,1,1)
            !write(432161,*), velocityMeridional_interp_gridded % array (16,16,3,new_timestep)
            !write(43217,*), temperature_interp_gridded % array (16,16,1,1)
            !write(432171,*), temperature_interp_gridded % array (16,16,3,new_timestep)
            !write(43218,*), salinity_interp_gridded % array (16,16,1,1)
            !write(432181,*), salinity_interp_gridded % array (16,16,3,new_timestep)

          call ocn_init_interpolation_bilinear_horiz(lon_interp % array, lat_interp % array, &
                                                     velocityZonal_interp_gridded % array (:,:,:,1) , nLon_obs, nLat_obs, &
                                                     lonCell, latCell, velocityZonal_interp, nCells, cell_mask % array)
       
          call ocn_init_interpolation_bilinear_horiz(lon_interp % array, lat_interp % array, &
                                                     velocityMeridional_interp_gridded % array (:,:,:,1), nLon_interp, nLat_interp, &
                                                     lonCell, latCell, velocityMeridional_interp, nCells, cell_mask % array)

          if (config_use_activeTracers_AOT_data_assimilation) then
               call mpas_pool_get_subpool(forcingPool, 'tracersAOTInterpData', tracersAOTInterpDataPool)
               call mpas_pool_get_array(tracersAOTInterpDataPool, "activeTracersAOTInterpolatedData", &
                                        activeTracersAOTInterpolatedData)
              call ocn_init_interpolation_bilinear_horiz(lon_interp % array, lat_interp % array, &
                                                         temperature_interp_gridded % array (:,:,:,1), nLon_interp, nLat_interp, &
                                                         lonCell, latCell, activeTracersAOTInterpolatedData(1,:,:), nCells, cell_mask % array)
              
              call ocn_init_interpolation_bilinear_horiz(lon_interp % array, lat_interp % array, &
                                                                    salinity_interp_gridded % array (:,:,:,1), nLon_interp, nLat_interp, &
                                                                    lonCell, latCell, activeTracersAOTInterpolatedData(2,:,:), nCells, cell_mask % array)
          endif

          block => block % next
       end do
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

      type (MPAS_IO_Handle_type) :: input_obs_File, input_interp_File, cell_mask_File, cell_mask_obs_File

      integer, intent(out) :: err

      character (len=StrKIND), pointer :: config_time_integrator, config_input_interp_file, config_input_obs_file , config_cell_mask_file, config_cell_mask_obs_file
      logical, pointer :: config_AOT_DA_on

      err = 0

      iocontext_ptr => iocontext

      call mpas_pool_get_config(ocnConfigs, 'config_AOT_DA_on', config_AOT_DA_on)
      call mpas_pool_get_config(ocnConfigs, 'config_time_integrator', config_time_integrator)

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


      if (config_AOT_DA_on ) then

         call mpas_pool_get_config(ocnConfigs, 'config_input_obs_file', config_input_obs_file)
         call mpas_pool_get_config(ocnConfigs, 'config_input_interp_file', config_input_interp_file)
         call mpas_pool_get_config(ocnConfigs, 'config_cell_mask_file', config_cell_mask_file)
         call mpas_pool_get_config(ocnConfigs, 'config_cell_mask_obs_file', config_cell_mask_obs_file)
   
         if (trim(config_input_obs_file) == 'none' .or. trim(config_input_interp_file) == 'none') then
            call mpas_log_write( 'Validation failed. Invalid filename for config_input_obs_file or config_input_interp_file', MPAS_LOG_CRIT)
            err = 1
            return
         end if
   
         input_obs_File = MPAS_io_open(config_input_obs_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=err)
         if(err > 0) then
            call mpas_log_write( 'Validation failed. Could not open file:'// &
               trim(config_input_obs_file) , MPAS_LOG_CRIT)
            return
         end if
   
         input_interp_File = MPAS_io_open(config_input_interp_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=err)
         if(err > 0) then
            call mpas_log_write( 'Validation failed. Could not open file:'// &
               trim(config_input_interp_file) , MPAS_LOG_CRIT)
            return
         end if
   
         cell_mask_File = MPAS_io_open(config_cell_mask_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=err)
         if(err > 0) then
            call mpas_log_write( 'Validation failed. Could not open file:'// &
               trim(config_cell_mask_file) , MPAS_LOG_CRIT)
            return
         end if

         cell_mask_obs_File = MPAS_io_open(config_cell_mask_obs_file, MPAS_IO_READ, MPAS_IO_NETCDF, iocontext_ptr, ierr=err)
         if(err > 0) then
            call mpas_log_write( 'Validation failed. Could not open file:'// &
               trim(config_cell_mask_obs_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_inq_dim(input_obs_File, 'lat', nLat_obs, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lat in file:'// &
               trim(config_input_obs_file) , MPAS_LOG_CRIT)
            return
         end if
   
         call MPAS_io_inq_dim(input_obs_File, 'lon', nLon_obs, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lon in file:'// &
               trim(config_input_obs_file) , MPAS_LOG_CRIT)
            return
         end if
   
         call MPAS_io_inq_dim(input_obs_File, 'nVertLevels', nVertLevels_obs, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim nVertLevels in file:'// &
               trim(config_input_obs_file) , MPAS_LOG_CRIT)
            return
         end if
   
         call MPAS_io_inq_dim(input_interp_File, 'lat', nLat_interp, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lat in file:'// &
               trim(config_input_interp_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_inq_dim(input_interp_File, 'lon', nLon_interp, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lon in file:'// &
               trim(config_input_interp_file) , MPAS_LOG_CRIT)
            return
         end if
   
         call MPAS_io_inq_dim(input_interp_File, 'nVertLevels', nVertLevels_interp, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim nVertLevels in file:'// &
               trim(config_input_interp_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_inq_dim(cell_mask_File, 'lat', nLat_cell, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lat in file:'// &
               trim(config_cell_mask_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_inq_dim(cell_mask_File, 'lon', nLon_cell, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lon in file:'// &
               trim(config_cell_mask_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_inq_dim(cell_mask_File, 'nVertLevels', nVertLevels_cell, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim nVertLevels in file:'// &
               trim(config_cell_mask_file) , MPAS_LOG_CRIT)
            return
         end if
   
         call MPAS_io_inq_dim(cell_mask_obs_File, 'lat', nLat_cell_obs, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lat in file:'// &
               trim(config_cell_mask_obs_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_inq_dim(cell_mask_obs_File, 'lon', nLon_cell_obs, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim lon in file:'// &
               trim(config_cell_mask_obs_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_inq_dim(cell_mask_obs_File, 'nVertLevels', nVertLevels_cell_obs, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not find dim nVertLevels in file:'// &
               trim(config_cell_mask_obs_file) , MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_close(input_obs_File, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not close file:'// &
               trim(config_input_obs_file), MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_close(input_interp_File, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not close file:'// &
               trim(config_input_interp_file), MPAS_LOG_CRIT)
            return
         end if

         call MPAS_io_close(cell_mask_File, err)
         if(err /= 0) then
            call mpas_log_write( 'Validation failed. Could not close file:'// &
               trim(config_cell_mask_file), MPAS_LOG_CRIT)
            return
         end if

    end if

   end subroutine ocn_timestep_init!}}}

subroutine ocn_read_cell_mask(domain, iErr)

      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr

      type (block_type), pointer :: block_ptr

      type (MPAS_Stream_type) :: cellMaskStream 
      integer :: i
      character(len=StrKIND), pointer :: config_cell_mask_file
      !character(LEN=184), pointer :: cell_mask / '/lustre/scratch4/turquoise/ecarlson10/mpas_ocean_runs/mpas_o_soma_tests/ocean_32_km/soma/32km/default/isopycnal_3layer_compil+interp/ocean/soma/32km/3layer/forward/output/cell_mask.nc' /

      iErr = 0     

      call mpas_pool_get_config(domain % configs, 'config_cell_mask_file', config_cell_mask_file)
 
      call MPAS_createStream(cellMaskStream, domain % iocontext, config_cell_mask_file, &
           MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)

       cell_mask % fieldName = 'cell_mask'
       cell_mask % dimSizes(3) = nVertLevels_cell
       cell_mask % dimSizes(2) = nLat_cell
       cell_mask % dimSizes(1) = nLon_cell
       cell_mask % dimNames(3) = 'nVertLevels'
       cell_mask % dimNames(2) = 'lat'
       cell_mask % dimNames(1) = 'lon'
       cell_mask % isVarArray = .false.
       cell_mask % isPersistent = .true.
       cell_mask % isActive = .true.
       cell_mask % hasTimeDimension = .false.
       cell_mask % block => domain % blocklist
       allocate(cell_mask % attLists(1))
       allocate(cell_mask % array(nLon_cell, nLat_cell, nVertLevels_cell))

       lat_cell % fieldName = 'lat'
       lat_cell % dimSizes(1) = nLat_cell 
       lat_cell % dimNames(1) = 'lat'
       lat_cell % isVarArray = .false.
       lat_cell % isPersistent = .true.
       lat_cell % isActive = .true.
       lat_cell % hasTimeDimension = .false.
       lat_cell % block => domain % blocklist
       allocate(lat_cell % attLists(1))
       allocate(lat_cell % array(nLat_cell))  

       lon_cell % fieldName = 'lon'
       lon_cell % dimSizes(1) = nLon_cell
       lon_cell % dimNames(1) = 'lon'
       lon_cell % isVarArray = .false.
       lon_cell % isPersistent = .true.
       lon_cell % isActive = .true.
       lon_cell % hasTimeDimension = .false.
       lon_cell % block => domain % blocklist
       allocate(lon_cell % attLists(1))
       allocate(lon_cell % array(nLon_cell))  

       call MPAS_streamAddField(cellMaskStream, cell_mask, iErr)
       call MPAS_streamAddField(cellMaskStream, lat_cell, iErr)
       call MPAS_streamAddField(cellMaskStream, lon_cell, iErr)

       call MPAS_readStream(cellMaskStream, 1, iErr)
       call MPAS_closeStream(cellMaskStream)

       lat_cell % array = lat_cell % array * 4*ATAN(1.d0)/180.
       lon_cell % array = lon_cell % array * 4*ATAN(1.d0)/180.

       !write(5497,*), maxval(cell_mask % array), minval(cell_mask % array)
       !write(5498,*), lat_cell % array
       !write(5499,*), lon_cell % array

end subroutine ocn_read_cell_mask

subroutine ocn_read_cell_mask_obs(domain, iErr)

      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr

      type (block_type), pointer :: block_ptr

      type (MPAS_Stream_type) :: cellMaskObsStream 
      integer :: i
      character(len=StrKIND), pointer :: config_cell_mask_obs_file
      !character(LEN=184), pointer :: cell_mask / '/lustre/scratch4/turquoise/ecarlson10/mpas_ocean_runs/mpas_o_soma_tests/ocean_32_km/soma/32km/default/isopycnal_3layer_compil+interp/ocean/soma/32km/3layer/forward/output/cell_mask.nc' /

      iErr = 0     

      call mpas_pool_get_config(domain % configs, 'config_cell_mask_obs_file', config_cell_mask_obs_file)
 
      call MPAS_createStream(cellMaskObsStream, domain % iocontext, config_cell_mask_obs_file, &
           MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)

       cell_mask_obs % fieldName = 'cell_mask'
       cell_mask_obs % dimSizes(3) = nVertLevels_cell_obs
       cell_mask_obs % dimSizes(2) = nLat_cell_obs
       cell_mask_obs % dimSizes(1) = nLon_cell_obs
       cell_mask_obs % dimNames(3) = 'nVertLevels'
       cell_mask_obs % dimNames(2) = 'lat'
       cell_mask_obs % dimNames(1) = 'lon'
       cell_mask_obs % isVarArray = .false.
       cell_mask_obs % isPersistent = .true.
       cell_mask_obs % isActive = .true.
       cell_mask_obs % hasTimeDimension = .false.
       cell_mask_obs % block => domain % blocklist
       allocate(cell_mask_obs % attLists(1))
       allocate(cell_mask_obs % array(nLon_cell_obs, nLat_cell_obs, nVertLevels_cell_obs))

       lat_cell_obs % fieldName = 'lat'
       lat_cell_obs % dimSizes(1) = nLat_cell_obs
       lat_cell_obs % dimNames(1) = 'lat'
       lat_cell_obs % isVarArray = .false.
       lat_cell_obs % isPersistent = .true.
       lat_cell_obs % isActive = .true.
       lat_cell_obs % hasTimeDimension = .false.
       lat_cell_obs % block => domain % blocklist
       allocate(lat_cell_obs % attLists(1))
       allocate(lat_cell_obs % array(nLat_cell_obs))  

       lon_cell_obs % fieldName = 'lon'
       lon_cell_obs % dimSizes(1) = nLon_cell_obs
       lon_cell_obs % dimNames(1) = 'lon'
       lon_cell_obs % isVarArray = .false.
       lon_cell_obs % isPersistent = .true.
       lon_cell_obs % isActive = .true.
       lon_cell_obs % hasTimeDimension = .false.
       lon_cell_obs % block => domain % blocklist
       allocate(lon_cell_obs % attLists(1))
       allocate(lon_cell_obs % array(nLon_cell_obs))  

       call MPAS_streamAddField(cellMaskObsStream, cell_mask_obs, iErr)
       call MPAS_streamAddField(cellMaskObsStream, lat_cell_obs, iErr)
       call MPAS_streamAddField(cellMaskObsStream, lon_cell_obs, iErr)

       call MPAS_readStream(cellMaskObsStream, 1, iErr)
       call MPAS_closeStream(cellMaskObsStream)

       lat_cell_obs % array = lat_cell_obs % array * 4*ATAN(1.d0)/180.
       lon_cell_obs % array = lon_cell_obs % array * 4*ATAN(1.d0)/180.
   
       !write(9000,*) minval(cell_mask_obs % array), maxval(cell_mask_obs % array), shape(cell_mask_obs % array)

end subroutine ocn_read_cell_mask_obs

subroutine ocn_AOT_read_obs_data(domain, iErr)
      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr

      type (block_type), pointer :: block_ptr

      type (MPAS_Stream_type) :: inputObsStream

      character (len=StrKIND), pointer :: config_input_obs_file 

      iErr = 0

      call mpas_pool_get_config(domain % configs,'config_input_obs_file', config_input_obs_file)

      call MPAS_createStream(inputObsStream, domain % iocontext, config_input_obs_file, &
           MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)

       lat_obs % fieldName = 'lat'
       lat_obs % dimSizes(1) = nLat_obs 
       lat_obs % dimNames(1) = 'lat'
       lat_obs % isVarArray = .false.
       lat_obs % isPersistent = .true.
       lat_obs % isActive = .true.
       lat_obs % hasTimeDimension = .false.
       lat_obs % block => domain % blocklist
       allocate(lat_obs % attLists(1))
       allocate(lat_obs % array(nLat_obs))  

       lon_obs % fieldName = 'lon'
       lon_obs % dimSizes(1) = nLon_obs
       lon_obs % dimNames(1) = 'lon'
       lon_obs % isVarArray = .false.
       lon_obs % isPersistent = .true.
       lon_obs % isActive = .true.
       lon_obs % hasTimeDimension = .false.
       lon_obs % block => domain % blocklist
       allocate(lon_obs % attLists(1))
       allocate(lon_obs % array(nLon_obs))  

       velocityZonal_obs_gridded % fieldName = 'velocityZonal'
       velocityZonal_obs_gridded % dimSizes(4) = 100!26101 
       velocityZonal_obs_gridded % dimSizes(3) = nVertLevels_obs
       velocityZonal_obs_gridded % dimSizes(2) = nLat_obs
       velocityZonal_obs_gridded % dimSizes(1) = nLon_obs
       velocityZonal_obs_gridded % dimNames(4) = 'Time'
       velocityZonal_obs_gridded % dimNames(3) = 'nVertLevels'
       velocityZonal_obs_gridded % dimNames(2) = 'lat'
       velocityZonal_obs_gridded % dimNames(1) = 'lon'
       velocityZonal_obs_gridded % isVarArray = .false.
       velocityZonal_obs_gridded % isPersistent = .true.
       velocityZonal_obs_gridded % isActive = .true.
       velocityZonal_obs_gridded % hasTimeDimension = .true.
       velocityZonal_obs_gridded % block => domain % blocklist
       allocate(velocityZonal_obs_gridded % attLists(1))
       allocate(velocityZonal_obs_gridded % array(nLon_obs, nLat_obs, nVertLevels_obs, 100))!26101))

       velocityMeridional_obs_gridded % fieldName = 'velocityMeridional'
       velocityMeridional_obs_gridded % dimSizes(4) = 100!26101
       velocityMeridional_obs_gridded % dimSizes(3) = nVertLevels_obs
       velocityMeridional_obs_gridded % dimSizes(2) = nLat_obs
       velocityMeridional_obs_gridded % dimSizes(1) = nLon_obs
       velocityMeridional_obs_gridded % dimNames(4) = 'Time'
       velocityMeridional_obs_gridded % dimNames(3) = 'nVertLevels'
       velocityMeridional_obs_gridded % dimNames(2) = 'lat'
       velocityMeridional_obs_gridded % dimNames(1) = 'lon'
       velocityMeridional_obs_gridded % isVarArray = .false.
       velocityMeridional_obs_gridded % isPersistent = .true.
       velocityMeridional_obs_gridded % isActive = .true.
       velocityMeridional_obs_gridded % hasTimeDimension = .true.
       velocityMeridional_obs_gridded % block => domain % blocklist
       allocate(velocityMeridional_obs_gridded % attLists(1))
       allocate(velocityMeridional_obs_gridded % array(nLon_obs, nLat_obs, nVertLevels_obs, 100))!26101))


       temperature_obs_gridded % fieldName = 'temperature'
       temperature_obs_gridded % dimSizes(4) = 100!26101
       temperature_obs_gridded % dimSizes(3) = nVertLevels_obs
       temperature_obs_gridded % dimSizes(2) = nLat_obs
       temperature_obs_gridded % dimSizes(1) = nLon_obs
       temperature_obs_gridded % dimNames(4) = 'Time'
       temperature_obs_gridded % dimNames(3) = 'nVertLevels'
       temperature_obs_gridded % dimNames(2) = 'lat'
       temperature_obs_gridded % dimNames(1) = 'lon'
       temperature_obs_gridded % isVarArray = .false.
       temperature_obs_gridded % isPersistent = .true.
       temperature_obs_gridded % isActive = .true.
       temperature_obs_gridded % hasTimeDimension = .true.
       temperature_obs_gridded % block => domain % blocklist
       allocate(temperature_obs_gridded % attLists(1))
       allocate(temperature_obs_gridded % array(nLon_obs, nLat_obs, nVertLevels_obs, 100))!26101))


       salinity_obs_gridded % fieldName = 'salinity'
       salinity_obs_gridded % dimSizes(4) = 100!26101
       salinity_obs_gridded % dimSizes(3) = nVertLevels_obs
       salinity_obs_gridded % dimSizes(2) = nLat_obs
       salinity_obs_gridded % dimSizes(1) = nLon_obs
       salinity_obs_gridded % dimNames(4) = 'Time'
       salinity_obs_gridded % dimNames(3) = 'nVertLevels'
       salinity_obs_gridded % dimNames(2) = 'lat'
       salinity_obs_gridded % dimNames(1) = 'lon'
       salinity_obs_gridded % isVarArray = .false.
       salinity_obs_gridded % isPersistent = .true.
       salinity_obs_gridded % isActive = .true.
       salinity_obs_gridded % hasTimeDimension = .true.
       salinity_obs_gridded % block => domain % blocklist
       allocate(salinity_obs_gridded % attLists(1))
       allocate(salinity_obs_gridded % array(nLon_obs, nLat_obs, nVertLevels_obs, 100))!26101))
       

       call MPAS_streamAddField(inputObsStream, lat_obs, iErr)
       call MPAS_streamAddField(inputObsStream, lon_obs, iErr)
       call MPAS_streamAddField(inputObsStream, velocityZonal_obs_gridded, iErr)
       call MPAS_streamAddField(inputObsStream, velocityMeridional_obs_gridded, iErr)
       call MPAS_streamAddField(inputObsStream, temperature_obs_gridded, iErr)
       call MPAS_streamAddField(inputObsStream, salinity_obs_gridded, iErr)

       call MPAS_readStream(inputObsStream, 1, iErr)
       call MPAS_closeStream(inputObsStream)

       lat_obs % array = lat_obs % array * 4*ATAN(1.d0)/180.
       lon_obs % array = lon_obs % array * 4*ATAN(1.d0)/180.

       !write(1110,*) maxval(lat_obs % array), minval(lon_obs % array)
       !write(1111,*) shape(velocityZonal_obs_gridded % array)!, velocityZonal_obs_gridded % array (1,16,16,1), velocityZonal_obs_gridded % array (1,16,16,2)
       !write(1112,*) velocityMeridional_obs_gridded % array (1,16,16,1), velocityMeridional_obs_gridded % array (1,16,16,2)
       !write(1113,*) temperature_obs_gridded % array (1,16,16,1), temperature_obs_gridded % array (1,16,16,2)
       !write(1114,*) salinity_obs_gridded % array (1,16,16,1), salinity_obs_gridded % array (1,16,16,2)

end subroutine ocn_AOT_read_obs_data

subroutine ocn_AOT_read_interp_data(domain, timestep, iErr)

      type (domain_type), intent(inout) :: domain
      integer, intent(in) :: timestep
      integer, intent(out) :: iErr

      type (block_type), pointer :: block_ptr

      type (MPAS_Stream_type) :: inputInterpStream

      character (len=StrKIND), pointer :: config_input_interp_file 

      iErr = 0

     call mpas_pool_get_config(domain % configs,'config_input_interp_file', config_input_interp_file)

     call MPAS_createStream(inputInterpStream, domain % iocontext, config_input_interp_file, &
            MPAS_IO_NETCDF, MPAS_IO_READ, ierr=iErr)

       lat_interp % fieldName = 'lat'
       lat_interp % dimSizes(1) = nLat_interp 
       lat_interp % dimNames(1) = 'lat'
       lat_interp % isVarArray = .false.
       lat_interp % isPersistent = .true.
       lat_interp % isActive = .true.
       lat_interp % hasTimeDimension = .false.
       lat_interp % block => domain % blocklist
       allocate(lat_interp % attLists(1))
       allocate(lat_interp % array(nLat_interp))

       lon_interp % fieldName = 'lon'
       lon_interp % dimSizes(1) = nLon_interp
       lon_interp % dimNames(1) = 'lon'
       lon_interp % isVarArray = .false.
       lon_interp % isPersistent = .true.
       lon_interp % isActive = .true.
       lon_interp % hasTimeDimension = .false.
       lon_interp % block => domain % blocklist
       allocate(lon_interp % attLists(1))
       allocate(lon_interp % array(nLon_interp))

       velocityZonal_interp_gridded % fieldName = 'velocityZonal'
       velocityZonal_interp_gridded % dimSizes(4) = timestep 
       velocityZonal_interp_gridded % dimSizes(3) = nVertLevels_interp
       velocityZonal_interp_gridded % dimSizes(2) = nLat_interp
       velocityZonal_interp_gridded % dimSizes(1) = nLon_interp
       velocityZonal_interp_gridded % dimNames(4) = 'Time'
       velocityZonal_interp_gridded % dimNames(3) = 'nVertLevels'
       velocityZonal_interp_gridded % dimNames(2) = 'lat'
       velocityZonal_interp_gridded % dimNames(1) = 'lon'
       velocityZonal_interp_gridded % isVarArray = .false.
       velocityZonal_interp_gridded % isPersistent = .true.
       velocityZonal_interp_gridded % isActive = .true.
       velocityZonal_interp_gridded % hasTimeDimension = .true.
       velocityZonal_interp_gridded % block => domain % blocklist
       allocate(velocityZonal_interp_gridded % attLists(1))
       allocate(velocityZonal_interp_gridded % array(nLon_interp, nLat_interp, nVertLevels_interp, timestep))

       velocityMeridional_interp_gridded % fieldName = 'velocityMeridional'
       velocityMeridional_interp_gridded % dimSizes(4) = timestep
       velocityMeridional_interp_gridded % dimSizes(3) = nVertLevels_interp
       velocityMeridional_interp_gridded % dimSizes(2) = nLat_interp
       velocityMeridional_interp_gridded % dimSizes(1) = nLon_interp
       velocityMeridional_interp_gridded % dimNames(4) = 'Time'
       velocityMeridional_interp_gridded % dimNames(3) = 'nVertLevels'
       velocityMeridional_interp_gridded % dimNames(2) = 'lat'
       velocityMeridional_interp_gridded % dimNames(1) = 'lon'
       velocityMeridional_interp_gridded % isVarArray = .false.
       velocityMeridional_interp_gridded % isPersistent = .true.
       velocityMeridional_interp_gridded % isActive = .true.
       velocityMeridional_interp_gridded % hasTimeDimension = .true.
       velocityMeridional_interp_gridded % block => domain % blocklist
       allocate(velocityMeridional_interp_gridded % attLists(1))
       allocate(velocityMeridional_interp_gridded % array(nLon_interp, nLat_interp, nVertLevels_interp, timestep))

       temperature_interp_gridded % fieldName = 'temperature'
       temperature_interp_gridded % dimSizes(4) = timestep
       temperature_interp_gridded % dimSizes(3) = nVertLevels_interp
       temperature_interp_gridded % dimSizes(2) = nLat_interp
       temperature_interp_gridded % dimSizes(1) = nLon_interp
       temperature_interp_gridded % dimNames(4) = 'Time'
       temperature_interp_gridded % dimNames(3) = 'nVertLevels'
       temperature_interp_gridded % dimNames(2) = 'lat'
       temperature_interp_gridded % dimNames(1) = 'lon'
       temperature_interp_gridded % isVarArray = .false.
       temperature_interp_gridded % isPersistent = .true.
       temperature_interp_gridded % isActive = .true.
       temperature_interp_gridded % hasTimeDimension = .true.
       temperature_interp_gridded % block => domain % blocklist
       allocate(temperature_interp_gridded % attLists(1))
       allocate(temperature_interp_gridded % array(nLon_interp, nLat_interp, nVertLevels_interp, timestep))

       salinity_interp_gridded % fieldName = 'salinity'
       salinity_interp_gridded % dimSizes(4) = timestep
       salinity_interp_gridded % dimSizes(3) = nVertLevels_interp
       salinity_interp_gridded % dimSizes(2) = nLat_interp
       salinity_interp_gridded % dimSizes(1) = nLon_interp
       salinity_interp_gridded % dimNames(4) = 'Time'
       salinity_interp_gridded % dimNames(3) = 'nVertLevels'
       salinity_interp_gridded % dimNames(2) = 'lat'
       salinity_interp_gridded % dimNames(1) = 'lon'
       salinity_interp_gridded % isVarArray = .false.
       salinity_interp_gridded % isPersistent = .true.
       salinity_interp_gridded % isActive = .true.
       salinity_interp_gridded % hasTimeDimension = .true.
       salinity_interp_gridded % block => domain % blocklist
       allocate(salinity_interp_gridded % attLists(1))
       allocate(salinity_interp_gridded % array(nLon_interp, nLat_interp, nVertLevels_interp, timestep))
       
       call MPAS_streamAddField(inputInterpStream, lat_interp, iErr)
       call MPAS_streamAddField(inputInterpStream, lon_interp, iErr)
       call MPAS_streamAddField(inputInterpStream, velocityZonal_interp_gridded, iErr)
       call MPAS_streamAddField(inputInterpStream, velocityMeridional_interp_gridded, iErr)
       call MPAS_streamAddField(inputInterpStream, temperature_interp_gridded, iErr)
       call MPAS_streamAddField(inputInterpStream, salinity_interp_gridded, iErr)

       call MPAS_readStream(inputInterpStream, 1, iErr)
       call MPAS_closeStream(inputInterpStream)

       lat_interp % array = lat_interp % array * 4*ATAN(1.d0)/180.
       lon_interp % array = lon_interp % array * 4*ATAN(1.d0)/180.

end subroutine ocn_AOT_read_interp_data

  subroutine check(status)
    integer, intent ( in) :: status

    if(status /= nf90_noerr) then
      print *, trim(nf90_strerror(status))
      stop "Stopped"
    end if
  end subroutine check

end module ocn_time_integration

! vim: foldmethod=marker

!!! Weight matrices for interpolation !!!
!!! NOTE: This routine needs to only be called once at the beginning of the run

module ocn_partition_ObsGrid

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timer
   use mpas_io_units
   use mpas_io
   use mpas_io_streams
   use mpas_log
   use mpas_dmpar

   use ocn_constants

   implicit none
   private
   save

   public :: partition_obsgrid

   real (kind=RKIND), dimension(:), pointer :: latCell, lonCell, refZMid
   integer :: nLat_obs, nLon_obs, nVertLevels_obs, time, tick, month
   type (field1DReal) :: lat_obs, lon_obs, dep_obs
   type (field4DReal) :: temp_obs, psal_obs, cell_mask
   real (kind=RKIND), pointer :: daysSinceStartOfSim

contains

subroutine partition_obsgrid(lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, domain, iErr)
      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      type (field1DReal), intent(inout) :: lat_obs, lon_obs, dep_obs
      integer, intent(inout) :: nLat_obs, nLon_obs, nVertLevels_obs

      type (mpas_pool_type), pointer :: diagnosticsPool, meshPool, statePool, forcingPool, verticalMeshPool
      real (kind=RKIND) :: xfrac, yfrac, zfrac
      integer :: iDep, iLat, iLon, procout, iCell_loc_1Global
      integer, dimension(1) :: iCell_loc_1, dep_loc_1, dep_loc_2
      integer :: bottom_left, top_left, top_right, bottom_right, dep_low_idx, dep_high_idx
      integer, pointer :: nCells, nVertLevels
      real (kind=RKIND) :: a, tolerance, distance
      real (kind=RKIND), allocatable, dimension(:) :: latCell_tmp, lonCell_tmp, refZMid_tmp, latCell_positive, latCell_negative, lonCell_left, lonCell_right, lat_obsGrid0, lon_obsGrid0, dep_obsGrid0
      real (kind=RKIND), dimension(:), pointer :: refZMid, latCell, lonCell
      integer :: i, j, iCell, iVertLevels, daysinyear, rank, local_rank
      real (kind=RKIND), allocatable, dimension(:,:,:) :: iCell_matrix, procout_matrix

      type (dm_info) :: dminfo
      type (block_type), pointer :: block

      INTEGER, dimension(12) :: month_day
      INTEGER, PARAMETER :: mday(12)   &
                          = (/31,28,31,30,31,30,31,31,30,31,30,31/)
      INTEGER, PARAMETER :: mdayleap(12) &
                          = (/31,29,31,30,31,30,31,31,30,31,30,31/)
      INTEGER, DIMENSION(365) :: daym
      INTEGER, DIMENSION(366) :: daymleap

      !call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      !call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      !call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
      !call mpas_pool_get_array(meshPool, 'latCell', latCell)
      !call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
      !call mpas_pool_get_dimension(meshPool, 'nCells', nCells)

      iErr=0

      tolerance = 480!240

      allocate (iCell_matrix(nLat_obs, nLon_obs, nVertLevels_obs))
      allocate (procout_matrix(nLat_obs, nLon_obs, nVertLevels_obs))

      block => domain % blocklist
      do while(associated(block))
        call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
        call mpas_pool_get_subpool(block % structs, 'state', statePool)
        call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
        call mpas_pool_get_array(meshPool, 'latCell', latCell)
        call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
        call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
        call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)

        call mpas_pool_get_array(diagnosticsPool, 'daysSinceStartOfSim',daysSinceStartOfSim)

        block => block % next
      end do

      latCell_positive = latCell
      latCell_negative = latCell
      lonCell_left = lonCell
      lonCell_right = lonCell

         do iCell = 1,nCells
             if (latCell(iCell) .lt. 0.0) then
                 latCell_positive(iCell) = latCell(iCell) + 2*4*ATAN(1.d0)
             elseif (latCell(iCell) .gt. 0.0) then
                 latCell_negative(iCell) = latCell(iCell) - 2*4*ATAN(1.d0)
             endif
         enddo

         do iCell = 1,nCells
             if (lonCell(iCell) .lt. 4*ATAN(1.d0)) then
                 lonCell_right(iCell) = lonCell_right(iCell) + 2*4*ATAN(1.d0)
             elseif (lonCell(iCell) .gt. 4*ATAN(1.d0)) then
                 lonCell_left(iCell) = lonCell_left(iCell) - 2*4*ATAN(1.d0)
             endif
         enddo

      call MPI_Comm_rank(MPI_COMM_WORLD, rank, iErr)

      !! Read observed data to all processors on first iteration, then delete
      !! forever later
      call read_obs_grid_dims(domain, iErr)

!      if (rank == 0) then
!         ! only define the lat_obsGrid0 on processor 0, do for all dims
!         lat_obsGrid0 = lat_obs % array
!         lon_obsGrid0 = lon_obs % array
!         dep_obsGrid0 = dep_obs % array
!
!         if (modulo(tick,4) == 0) then
!            daysinyear = 366
!            month_day = mdayleap
!         else
!            daysinyear = 365
!            month_day = mday
!         endif
!
!         do i=1,11
!            if (modulo(daysSinceStartOfSim,real(daysinyear, kind=RKIND)) .lt. month_day(1)) then
!               month = 1
!               call read_obs_temp_grid_fields_only(domain, tick, month, iErr)
!               call read_obs_psal_grid_fields_only(domain, tick, month, iErr)
!               exit
!            elseif (modulo(daysSinceStartOfSim,real(daysinyear, kind=RKIND)) .gt. month_day(i) .and. modulo(daysSinceStartOfSim,real(daysinyear,kind=RKIND)) .le. month_day(i+1)) then
!               print*, 'in the if loop on second iteration?'
!               month = i+1
!               if (month == 12) tick = tick + 1
!               call read_obs_temp_grid_fields_only(domain, tick, month, iErr)
!               call read_obs_psal_grid_fields_only(domain, tick, month, iErr)
!            endif
!         enddo
!
!      endif
! 
!            
!      do iDep = 1,nVertLevels_obs
!         do iLat = 1,nLat_obs
!            do iLon = 1,nLon_obs
!               if (iLat < 20) then
!                   latCell_tmp = latCell_negative
!               elseif (iLat > (nLat_obs - 20)) then
!                    latCell_tmp = latCell_positive
!               else
!                    latCell_tmp = latCell
!               endif
!         
!               if ( iLon .le. 5) then
!                   lonCell_tmp = lonCell_left
!               elseif ( iLon .ge. (nLon_obs - 5)) then
!                   lonCell_tmp = lonCell_right
!               else
!                   lonCell_tmp = lonCell
!               endif
!
!               iCell_loc_1 = minloc((latCell_tmp-lat_obs % array (iLat))**2 + (lonCell_tmp-lon_obs % array (iLon))**2)
!               call mpas_dmpar_minloc_int(domain % dminfo, iCell_loc_1(1), iCell_loc_1Global, procout)
!               iCell_matrix(iLat,iLon,iDep) = iCell_loc_1Global
!               procout_matrix(iLat,iLon,iDep) = procout
!               call MPI_Comm_rank(MPI_COMM_WORLD, local_rank, iErr)
!               
!               call MPI_Send(iCell_matrix(iLat,iLon,iDep), 1, MPI_REAL, 0, 0, MPI_COMM_WORLD, iErr)
!               call MPI_Send(procout_matrix(iLat,iLon,iDep), 1, MPI_REAL, 0, 0, MPI_COMM_WORLD, iErr)
!
!               if (rank == 0) then
!                  call MPI_Recv(iCell_matrix(iLat,iLon,iDep), 1, MPI_REAL, 0, 0, MPI_COMM_WORLD, iErr)
!                  call MPI_Recv(procout_matrix(iLat,iLon,iDep), 1, MPI_REAL, 0, 0, MPI_COMM_WORLD, iErr)
!               endif
!
!               !print*, iCell_loc_1Global, lat_obs % array (iLat), latCell_tmp(iCell_loc_1Global)
!               !print*, iCell_loc_1Global, lon_obs % array (iLon), lonCell_tmp(iCell_loc_1Global)
!
!            enddo
!         enddo
!      enddo

end subroutine partition_obsgrid

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

      !print*, config_input_obs_file_iterance

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
      !character(len=91), pointer :: testme /
      !'/lustre/scratch4/turquoise/ecarlson10/climate_data/all_years/ISAS15_DM_20150115_fld_PSAL.nc'
      !/

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
      dep_obs % array = -1.0 * dep_obs % array
      !print*, nLat_obs, nLon_obs, nVertLevels_obs

      do iLon = 1, nLon_obs
         if (lon_obs % array (iLon) .lt. 0) then
            lon_obs % array (iLon) = lon_obs % array (iLon) + 2*4*ATAN(1.d0)
         endif
      enddo

end subroutine read_obs_grid_dims

end module ocn_partition_ObsGrid

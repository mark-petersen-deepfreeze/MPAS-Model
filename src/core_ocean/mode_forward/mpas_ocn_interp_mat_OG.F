!!! Weight matrices for interpolation !!!
!!! NOTE: This routine needs to only be called once at the beginning of the run

module ocn_interp_mat

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timer

   use mpas_log
   use mpas_kind_types
   use mpas_io_units
   use mpas_io
   use mpas_io_streams

   use ocn_constants
   !use ocn_partition_obsgrid

   implicit none
   private
   save

   public :: interp_MPAS_to_ObsGrid, interp_ObsGrid_to_MPAS, call_twice

   real (kind=RKIND), dimension(:), pointer :: latCell, lonCell, refZMid
   integer :: nLat_obs, nLon_obs, nVertLevels_obs, time, tick, month
   type (field1DReal) :: lat_obs, lon_obs, dep_obs
   type (field4DReal) :: temp_obs, psal_obs, cell_mask

contains

! This script is written for ARGO data from this website:
! https://www.seanoe.org/data/00412/52367/
! Note that the MPAS grid is nested inside of the observed grid, not in the
! latitude and longitude but in the depth, so the time integration file accounts
! for the lack of data

subroutine call_twice(lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, domain, iErr)

      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      type (field1DReal), intent(inout) :: lat_obs, lon_obs, dep_obs
      integer, intent(inout) :: nLat_obs, nLon_obs, nVertLevels_obs

      iErr=0

      call read_obs_grid_dims(domain, iErr)

end subroutine call_twice

subroutine interp_MPAS_to_ObsGrid(meshPool, verticalMeshPool, lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, wgtMatrix, domain, iErr)
     
      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      type (field1DReal), intent(inout) :: lat_obs, lon_obs, dep_obs
      integer, intent(inout) :: nLat_obs, nLon_obs, nVertLevels_obs
      integer :: iCell

      !! internal variables !!
      real (kind=RKIND) :: xfrac, yfrac, zfrac
      integer :: iDep, iLat, iLon, procout, iCell_loc_1Global
      integer, dimension(1) :: iCell_loc_1, dep_loc_1, dep_loc_2
      integer :: bottom_left, top_left, top_right, bottom_right, dep_low_idx, dep_high_idx 
      integer, pointer :: nCells, nVertLevels
      real (kind=RKIND) :: a, tolerance, distance
      real (kind=RKIND), allocatable, dimension(:) :: latCell_tmp, lonCell_tmp, refZMid_tmp, latCell_positive, latCell_negative, lonCell_left, lonCell_right
      real (kind=RKIND), dimension(:), pointer :: refZMid, latCell, lonCell
      real (kind=RKIND), allocatable, dimension(:,:), intent(out) :: wgtMatrix
      type (mpas_pool_type), pointer, intent(in) :: meshPool, verticalMeshPool
      integer, dimension(:,:), pointer, contiguous :: cellsOnCell

      call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
      call mpas_pool_get_array(meshPool, 'latCell', latCell)
      call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
      call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      
      allocate( wgtMatrix(nLat_obs + nLon_obs + nVertLevels_obs-2, 10) )

      tolerance = 480!240

      !call call_twice(lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, domain, iErr)
      !print*, maxval(lat_obs % array)

      print*, latCell(-1)

      !call partition_obsgrid(lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, domain, iErr) 


      top_left = -1
      top_right = -1
      bottom_left = -1
      bottom_right = -1

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

      do iDep = 1,nVertLevels_obs
         do iLat = 1,nLat_obs
            do iLon = 1,nLon_obs

               if (iLat < 20) then
                   latCell_tmp = latCell_negative
               elseif (iLat > (nLat_obs - 20)) then
                    latCell_tmp = latCell_positive
               else
                    latCell_tmp = latCell
               endif

               if ( iLon .le. 5) then
                   lonCell_tmp = lonCell_left
               elseif ( iLon .ge. (nLon_obs - 5)) then
                   lonCell_tmp = lonCell_right
               else
                   lonCell_tmp = lonCell
               endif

               !print*, lat_obs % array (iLat), lon_obs % array (iLon)
               !print*, minval(latCell_negative), maxval(latCell_negative)
               !print*, minval(latCell_positive), maxval(latCell_positive)
               !print*, minval(lonCell_left), maxval(lonCell_left)
               !print*, minval(lonCell_right), maxval(lonCell_right)

               do while (bottom_right .eq. -1 .or. bottom_left .eq. -1 .or. top_left .eq. -1 .or. top_right .eq. -1)
                   iCell_loc_1 = minloc((latCell_tmp-lat_obs % array (iLat))**2 + (lonCell_tmp-lon_obs % array (iLon))**2)
                   call mpas_dmpar_minloc_int(domain % dminfo, iCell_loc_1(1), iCell_loc_1Global, procout)
                   iCell_loc_1(1) = iCell_loc_1Global
                   !print*, iCell_loc_1(1)
                   !print*, latCell(-1)
                   a = sin((lat_obs % array(iLat) - latCell(iCell_loc_1(1)))/2)**2 + &
                                    cos(lat_obs % array(iLat))*cos(latCell(iCell_loc_1(1))) * &
                                    sin((lon_obs % array(iLon) - lonCell(iCell_loc_1(1)))/2)**2
                   distance = 6371.0*2.0*atan( sqrt(a) / sqrt(1-a) )
                   print*, lat_obs % array(iLat), latCell(iCell_loc_1(1))
                   print*, lon_obs % array(iLon), lonCell(iCell_loc_1(1))
                   print*, distance

                   if (distance .ge. tolerance) then
                      if (bottom_right .eq. -1) then 
                          bottom_right = -32
                      endif
                      if (top_right .eq. -1) then 
                          top_right = -32
                      endif
                      if (bottom_left .eq. -1) then 
                          bottom_left = -32
                      endif
                      if (top_left .eq. -1) then 
                          top_left = -32
                      endif
                      !exit
                   else
                      if ((latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat))*(lonCell_tmp(iCell_loc_1(1)) - lon_obs % array (iLon)) .le. 0.0) then

                         if (latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat) .le. 0.0 .and. bottom_right .eq. -1) then
                                 bottom_right = iCell_loc_1(1)
                                 !print*, 'bottom_right', latCell(bottom_right), lonCell(bottom_right)
                         elseif (latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat) .ge. 0.0 .and. top_left .eq. -1) then
                                 top_left = iCell_loc_1(1)
                                 !print*, 'top_left', latCell(top_left), lonCell(top_left)
                         endif
                            
                      else

                         if (latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat) .le. 0.0 .and. bottom_left .eq. -1) then
                                 bottom_left = iCell_loc_1(1)
                                 !print*, 'bottom_left', latCell(bottom_left), lonCell(bottom_left)
                         elseif (latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat) .ge. 0.0 .and. top_right .eq. -1) then
                                 top_right = iCell_loc_1(1)
                                 !print*, 'top_right', latCell(top_right), lonCell(top_right)
                         endif

                      endif
                   endif

                   latCell_tmp(iCell_loc_1) = 1e32
                   lonCell_tmp(iCell_loc_1) = 1e32
               end do

               print*, bottom_left, top_left, bottom_right, top_right

               print*, 'blah'

              !print*, 'top_left = ', top_left, latCell(top_left), lonCell(top_left)
              !print*, 'top_right = ', top_right, latCell(top_right), lonCell(top_right)
              !print*, 'bottom_left = ', bottom_left, latCell(bottom_left), lonCell(bottom_left)
              !print*, 'bottom_right = ', bottom_right, latCell(bottom_right), lonCell(bottom_right)
              !print*, lat_obs % array (iLat), lon_obs % array (iLon)

              print*, latCell(-1)

              if (bottom_left .ne. -32 .and. bottom_right .ne. -32) then
                  xfrac = (lat_obs % array (iLat) - latCell(bottom_left))/(latCell(bottom_right)-latCell(bottom_left))
              elseif (top_left .ne. -32 .and. top_right .ne. -32) then 
                  xfrac = (lat_obs % array (iLat) - latCell(top_left))/(latCell(top_right)-latCell(top_left))
              elseif (bottom_left .eq. -32 .or. top_left .eq. -32) then
                  xfrac = 1
              !elseif (bottom_right .ne. 32 .or. top_right .ne. -32) then
              !   xfrac = 0
              else
                  xfrac = 0
              endif 

              if (bottom_left .ne. -32 .and. top_left .ne. -32) then
                  yfrac = (lon_obs % array (iLon) - lonCell(bottom_left))/(lonCell(top_left) - lonCell(bottom_left))
              elseif (bottom_right .ne. -32 .and. top_right .ne. -32) then
                  yfrac = (lon_obs % array (iLon) - lonCell(bottom_right))/(lonCell(top_right) - lonCell(bottom_right))
              elseif (bottom_left .eq. -32 .or. bottom_right .eq. -32) then
                  yfrac = 1
              else
                  yfrac = 0
              endif

               refZMid_tmp = refZMid
               dep_loc_1 = minloc(abs(dep_obs % array (iDep) - refZMid_tmp))
               refZMid_tmp(dep_loc_1) = 1e32
               dep_loc_2 = minloc(abs(dep_obs % array (iDep) - refZMid_tmp))

               if (dep_obs % array (iDep) - refZMid(dep_loc_1(1)) .ge. 0) then
                  zfrac = (dep_obs % array (iDep) - refZMid(dep_loc_1(1))) / (refZMid(dep_loc_2(1)) - refZMid(dep_loc_1(1)))
                  dep_low_idx = dep_loc_1(1)
                  dep_high_idx = dep_loc_2(1)
                  
               else
                  zfrac = (dep_obs % array (iDep) - refZMid(dep_loc_2(1))) / (refZMid(dep_loc_1(1)) - refZMid(dep_loc_2(1)))
                  dep_low_idx = dep_loc_2(1)
                  dep_high_idx = dep_loc_1(1)
               endif

               wgtMatrix(iLat+iLon+iDep-2, 1) = xfrac
               wgtMatrix(iLat+iLon+iDep-2, 2) = yfrac
               wgtMatrix(iLat+iLon+iDep-2, 3) = zfrac
               wgtMatrix(iLat+iLon+iDep-2, 4) = bottom_left
               wgtMatrix(iLat+iLon+iDep-2, 5) = top_left
               wgtMatrix(iLat+iLon+iDep-2, 6) = bottom_right
               wgtMatrix(iLat+iLon+iDep-2, 7) = top_right
               wgtMatrix(iLat+iLon+iDep-2, 8) = dep_low_idx
               wgtMatrix(iLat+iLon+iDep-2, 9) = dep_high_idx
               wgtMatrix(iLat+iLon+iDep-2, 10) = distance

            enddo
         enddo
      enddo

end subroutine interp_MPAS_to_ObsGrid
          
subroutine interp_ObsGrid_to_MPAS(meshPool, verticalMeshPool, lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, wgtMatrix, domain, iErr)

      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      type (field1DReal), intent(in) :: lat_obs, lon_obs, dep_obs
      integer, intent(in) :: nLat_obs, nLon_obs, nVertLevels_obs

      !! internal variables !!
      real (kind=RKIND) :: xfrac, yfrac, zfrac
      integer :: iCell, iVertLevels
      integer, dimension(1) :: lat_loc_1, lat_loc_2, lon_loc_1, lon_loc_2, dep_loc_1, dep_loc_2
      integer :: lat_left_idx, lat_right_idx, lon_left_idx, lon_right_idx, dep_low_idx, dep_high_idx
      integer, pointer :: nCells, nVertLevels
      real (kind=RKIND), dimension(:), pointer :: latCell, lonCell
      real (kind=RKIND), allocatable, dimension(:) :: lat_obs_tmp, lon_obs_tmp, dep_obs_tmp
      real (kind=RKIND), allocatable, dimension(:,:), intent(out) :: wgtMatrix
      type (mpas_pool_type), pointer, intent(in) :: meshPool, verticalMeshPool

      call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
      call mpas_pool_get_array(meshPool, 'latCell', latCell)
      call mpas_pool_get_array(meshPool, 'lonCell', lonCell)

          ! note that when this routine is called the first time,
          ! this will only take zMid at intial time so this
          ! interpolation only works for fixed zlevels
          ! to modify, weight matrix could only be generated in 2D

      allocate( wgtMatrix(nCells + nVertLevels - 1,9) )

      do iVertLevels=1,nVertLevels
          do iCell=1,nCells

             lat_obs_tmp = lat_obs % array
             lon_obs_tmp = lon_obs % array

             lat_loc_1 = minloc(abs(latCell(iCell)-lat_obs_tmp))
             if (latCell(iCell)-lat_obs % array (lat_loc_1(1)) .ge. 0.0) then !! this will only work on the interior !! 
                if (lat_loc_1(1) .ne. nLat_obs) then
                    lat_obs_tmp(1:lat_loc_1(1)) = 1e32  !! here we are ASSUMING that the latitude is arranged in a linear order; we can generalize this later
                                                        !! this should be adjusted depending on how the latitude is stored in the data file 
                    lat_loc_2 = minloc(abs(latCell(iCell)-lat_obs_tmp))
                else
                    lat_loc_2 = 1
                endif
             else
                 if (lat_loc_1(1) .ne. 1) then
                    lat_obs_tmp(lat_loc_1(1):nLat_obs) = 1e32
                    lat_loc_2 = minloc(abs(latCell(iCell)-lat_obs_tmp))
                 else
                    lat_loc_2 = 1
                 endif
             endif 

             ! note that the resulting stuff has been organized from >180 to
             ! 360, then 
             lon_loc_1 = minloc(abs(lonCell(iCell)-lon_obs_tmp))
             if (lonCell(iCell)-lon_obs % array (lon_loc_1(1)) .ge. 0.0) then !! this will only work on the interior !! 
                if (lon_loc_1(1) .ne. nLon_obs/2+1) then
                    lon_obs_tmp(lon_loc_1(1)) = 1e32  !! here we are ASSUMING that the longitude is arranged in a linear order; we can generalize this later
                                                        !! this should be adjusted depending on how the longitude is stored in the data file 
                    lon_loc_2 = minloc(abs(lonCell(iCell)-lon_obs_tmp))
                else
                    lon_loc_2 = 1
                endif
             else
                 if (lon_loc_1(1) .ne. nLon_obs/2) then
                    lon_obs_tmp(lon_loc_1(1)) = 1e32
                    lon_loc_2 = minloc(abs(lonCell(iCell)-lon_obs_tmp))
                 else
                    lon_loc_2 = 1
                 endif
             endif 

             if ( latCell(iCell)-lat_obs % array (lat_loc_1(1)) .ge. 0.0 ) then
                if (lat_loc_1(1) .eq. lat_loc_2(1)) then
                    xfrac = 0.0
                    lat_left_idx  = lat_loc_1(1)
                    lat_right_idx = lat_loc_2(1)
                elseif (lat_loc_1(1) .ne. nLat_obs) then
                    xfrac = (latCell(iCell) - lat_obs % array (lat_loc_1(1)))/(lat_obs % array (lat_loc_2(1))  - lat_obs % array (lat_loc_1(1)))
                    lat_left_idx  = lat_loc_1(1)
                    lat_right_idx = lat_loc_2(1)
                elseif (lat_loc_1(1) .eq. nLat_obs) then
                    xfrac = (latCell(iCell) - lat_obs % array (nLat_obs))/(lat_obs % array (1) + 2*4*ATAN(1.d0) - lat_obs % array (nLat_obs))
                    lat_left_idx = 1
                    lat_right_idx = nLat_obs
                endif
             else
                if (lat_loc_1(1) .eq. lat_loc_2(1)) then
                    xfrac = 0.0
                    lat_left_idx  = lat_loc_2(1)
                    lat_right_idx = lat_loc_1(1)
                elseif (lat_loc_1(1) .ne. 1) then
                     xfrac = (latCell(iCell) - lat_obs % array (lat_loc_2(1)))/(lat_obs % array (lat_loc_1(1)) - lat_obs % array (lat_loc_2(1)))
                    lat_left_idx  = lat_loc_2(1)
                    lat_right_idx = lat_loc_1(1)
                elseif (lat_loc_1(1) .eq. 1)  then
                    xfrac = ( latCell(iCell) - (lat_obs % array (nLat_obs) - 2*4*ATAN(1.d0)) )/ &
                            ( lat_obs % array (1) - ( lat_obs % array (nLat_obs) - 2*4*ATAN(1.d0) ) )
                    lat_left_idx = nLat_obs
                    lat_right_idx = 1
                endif
             endif
    
             if ( lonCell(iCell)-lon_obs % array (lon_loc_1(1)) .ge. 0.0 ) then
                if (lon_loc_1(1) .eq. lon_loc_2(1)) then
                    yfrac = 0.0
                    lon_left_idx  = lon_loc_1(1)
                    lon_right_idx = lon_loc_2(1)
                elseif (lon_loc_1(1) .ne. nLon_obs/2) then
                    yfrac = (lonCell(iCell) - lon_obs % array (lon_loc_1(1)))/(lon_obs % array (lon_loc_2(1))  - lon_obs % array (lon_loc_1(1)))
                    lon_left_idx  = lon_loc_1(1)
                    lon_right_idx = lon_loc_2(1)
                elseif (lon_loc_1(1) .eq. nLon_obs/2) then
                    yfrac = (lonCell(iCell) - lon_obs % array (nLon_obs/2))/(lon_obs % array (nLon_obs/2+1) + 2*4*ATAN(1.d0) - lon_obs % array (nLon_obs/2))
                    lon_left_idx = nLon_obs/2
                    lon_right_idx = nLon_obs/2+1
                endif
             else
                if (lon_loc_1(1) .eq. lon_loc_2(1)) then
                    yfrac = 0.0
                    lon_left_idx  = lon_loc_2(1)
                    lon_right_idx = lon_loc_1(1)
                elseif (lon_loc_1(1) .ne. nLon_obs/2+1) then
                    yfrac = (lonCell(iCell) - lon_obs % array (lon_loc_2(1)))/(lon_obs % array (lon_loc_1(1)) - lon_obs % array (lon_loc_2(1)))
                    lon_left_idx  = lon_loc_2(1)
                    lon_right_idx = lon_loc_1(1)
                elseif (lon_loc_1(1) .eq. nLon_obs/2+1)  then
                    yfrac = ( lonCell(iCell) - (lon_obs % array (nLon_obs/2) - 2*4*ATAN(1.d0)) )/ &
                            ( lon_obs % array (nLon_obs/2) - ( lon_obs % array (nLon_obs/2+1) - 2*4*ATAN(1.d0) ) )
                    lon_left_idx = nLon_obs/2
                    lon_right_idx = nLon_obs/2+1
                endif
             endif
 
    
             if (refZMid(iVertLevels) .lt. minval(dep_obs % array)) then
                zfrac = 1e32
             else

                dep_obs_tmp = dep_obs % array
                dep_loc_1 = minloc(abs(refZMid(iVertLevels) - dep_obs_tmp ))
   
                if (refZMid(iVertLevels) .ge. -1) then
                    zfrac = 0
                    dep_low_idx = dep_loc_1(1)
                    dep_high_idx = dep_loc_1(1)
                elseif (refZMid(iVertLevels) - dep_obs % array (dep_loc_1(1)) .ge. 0) then
                    dep_low_idx = dep_loc_1(1)-1
                    dep_high_idx = dep_loc_1(1)
                    zfrac = (refZMid(iVertLevels) - dep_obs % array (dep_low_idx)) / (dep_obs % array (dep_high_idx) - dep_obs % array (dep_low_idx))
                elseif (refZMid(iVertLevels) - dep_obs % array (dep_loc_1(1)) .lt. 0) then
                    dep_low_idx = dep_loc_1(1)
                    dep_high_idx = dep_loc_1(1)+1
                    zfrac = (refZMid(iVertLevels) - dep_obs % array (dep_low_idx)) / (dep_obs % array (dep_high_idx) - dep_obs % array (dep_low_idx))
                endif
   
             endif

             wgtMatrix(iCell + iVertLevels - 1,1)  = xfrac
             wgtMatrix(iCell + iVertLevels - 1,2)  = yfrac
             wgtMatrix(iCell + iVertLevels - 1,3)  = zfrac
             wgtMatrix(iCell + iVertLevels - 1,4)  = lat_left_idx 
             wgtMatrix(iCell + iVertLevels - 1,5)  = lat_right_idx 
             wgtMatrix(iCell + iVertLevels - 1,6)  = lon_left_idx 
             wgtMatrix(iCell + iVertLevels - 1,7)  = lon_right_idx 
             wgtMatrix(iCell + iVertLevels - 1,8)  = dep_low_idx 
             wgtMatrix(iCell + iVertLevels - 1,9)  = dep_high_idx 

          enddo
      enddo

end subroutine interp_ObsGrid_to_MPAS

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

end module ocn_interp_mat

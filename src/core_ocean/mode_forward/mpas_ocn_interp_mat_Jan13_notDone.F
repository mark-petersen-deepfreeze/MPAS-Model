!!! Weight matrices for interpolation !!!

module ocn_interp_mat

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timer

   use ocn_constants

   implicit none
   private
   save

   public :: interp_MPAS_to_ObsGrid, interp_ObsGrid_to_MPAS

   real (kind=RKIND), dimension(:), pointer :: latCell, lonCell, refZMid

contains

subroutine interp_MPAS_to_ObsGrid(meshPool, verticalMeshPool, lat_obs, lon_obs, dep_obs, nLat_obs, nLon_obs, nVertLevels_obs, wgtMatrix, domain, iErr)
     
      type (domain_type), intent(inout) :: domain
      integer, intent(inout) :: iErr
      type (field1DReal), intent(in) :: lat_obs, lon_obs, dep_obs
      integer, intent(in) :: nLat_obs, nLon_obs, nVertLevels_obs
      integer :: iCell

      !! internal variables !!
      real (kind=RKIND) :: xfrac, yfrac, zfrac
      integer :: iDep, iLat, iLon
      integer, dimension(1) :: iCell_loc_1, dep_loc_1, dep_loc_2
      integer :: bottom_left, top_left, top_right, bottom_right, dep_low_idx, dep_high_idx 
      integer, pointer :: nCells, nVertLevels
      real (kind=RKIND) :: a_bottom_left, a_bottom_right, a_top_left, a_top_right, distance_bottom_left, distance_bottom_right, distance_top_left, distance_top_right
      real (kind=RKIND), allocatable, dimension(:) :: latCell_tmp, lonCell_tmp, refZMid_tmp, latCell_positive, latCell_negative, lonCell_left, lonCell_right
      real (kind=RKIND), dimension(:), pointer :: refZMid, latCell, lonCell
      real (kind=RKIND), allocatable, dimension(:,:), intent(out) :: wgtMatrix
      type (mpas_pool_type), pointer, intent(in) :: meshPool, verticalMeshPool

      call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
      call mpas_pool_get_array(meshPool, 'latCell', latCell)
      call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
      call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      
      allocate( wgtMatrix(nLat_obs + nLon_obs + nVertLevels_obs-2, 13) )

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
         do iLat = 1,2!1,nLat_obs
            do iLon = 1,2!1,nLon_obs

               if (iLat < 20) then
                   latCell_tmp = latCell_negative
               elseif (iLat > (nLat_obs - 20)) then
                    latCell_tmp = latCell_positive
               else
                    latCell_tmp = latCell
               endif

               if (lon_obs % array (iLon) .le. 5) then
                   lonCell_tmp = lonCell_left
               elseif (lon_obs % array (iLon) .ge. (nLon_obs - 5)) then
                   lonCell_tmp = lonCell_right
               else
                   lonCell_tmp = lonCell
               endif

               print*, lat_obs % array (iLat), lon_obs % array (iLon)
               !print*, minval(latCell_negative), maxval(latCell_negative)
               !print*, minval(latCell_positive), maxval(latCell_positive)
               !print*, minval(lonCell_left), maxval(lonCell_left)
               !print*, minval(lonCell_right), maxval(lonCell_right)


               do while (bottom_right .eq. -1 .or. bottom_left .eq. -1 .or. top_left .eq. -1 .or. top_right .eq. -1)
                   iCell_loc_1 = minloc((latCell_tmp-lat_obs % array (iLat))**2 + (lonCell_tmp-lon_obs % array (iLon))**2)
                   !print*, iCell_loc_1(1)
                   !print*, latCell(-1)

                   if ((latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat))*(lonCell_tmp(iCell_loc_1(1)) - lon_obs % array (iLon)) .le. 0.0) then
                      if (latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat) .le. 0.0 .and. bottom_right .eq. -1) then
                                bottom_right = iCell_loc_1(1)
                                a_bottom_right = sin((lat_obs % array(iLat) - latCell(bottom_right))/2)**2 + &
                                                  cos(lat_obs % array(iLat))*cos(latCell(bottom_right)) * &
                                                  sin((lon_obs % array(iLon) - lonCell(bottom_right))/2)**2
                                distance_bottom_right = 6371.0*2.0*atan( sqrt(a_bottom_right) / sqrt(1-a_bottom_right) )
                                print*, 'bottom_right', latCell(bottom_right), lonCell(bottom_right)
                      elseif (top_left .eq. -1) then
                                top_left = iCell_loc_1(1)
                                a_top_left = sin((lat_obs % array(iLat) - latCell(top_left))/2)**2 + & 
                                                  cos(lat_obs % array(iLat))*cos(latCell(top_left)) * &
                                                  sin((lon_obs % array(iLon) - lonCell(top_left))/2)**2
                                distance_top_left = 6371.0*2.0*atan( sqrt(a_top_left) / sqrt(1-a_top_left) )
                                print*, 'top_left', latCell(top_left), lonCell(top_left)
                      endif
                   else
                      if (latCell_tmp(iCell_loc_1(1)) - lat_obs % array (iLat) .le. 0.0 .and. bottom_left .eq. -1) then
                                bottom_left = iCell_loc_1(1)
                                a_bottom_left = sin((lat_obs % array(iLat) - latCell(bottom_left))/2)**2 + &
                                                  cos(lat_obs % array (iLat))*cos(latCell(bottom_left)) * &
                                                  sin((lon_obs % array (iLon) - lonCell(bottom_left))/2)**2
                                distance_bottom_left = 6371.0*2.0*atan( sqrt(a_bottom_left) / sqrt(1-a_bottom_left) )
                                print*, 'bottom_left', latCell(bottom_left), lonCell(bottom_left)
                      elseif (top_right .eq. -1) then
                                top_right = iCell_loc_1(1)
                                a_top_right = sin((lat_obs % array(iLat) - latCell(top_right))/2)**2 + &
                                                  cos(lat_obs % array(iLat)) *cos(latCell(top_right)) * &
                                                  sin((lon_obs % array(iLon) - lonCell(top_right))/2)**2
                                distance_top_right = 6371.0*2.0*atan( sqrt(a_top_right) / sqrt(1-a_top_right) )
                                print*, 'top_right', latCell(top_right), lonCell(top_right)
                      endif
                   endif

                   latCell_tmp(iCell_loc_1) = 1e32
                   lonCell_tmp(iCell_loc_1) = 1e32
               end do
                   print*, 'here'

              !print*, 'top_left = ', top_left, latCell(top_left), lonCell(top_left)
              !print*, 'top_right = ', top_right, latCell(top_right), lonCell(top_right)
              !print*, 'bottom_left = ', bottom_left, latCell(bottom_left), lonCell(bottom_left)
              !print*, 'bottom_right = ', bottom_right, latCell(bottom_right), lonCell(bottom_right)
              !print*, lat_obs % array (iLat), lon_obs % array (iLon)

              !print*, latCell(-1)

              xfrac = (lat_obs % array (iLat) - latCell(bottom_left))/(latCell(bottom_right)-latCell(bottom_left))
              yfrac = (lon_obs % array (iLon) - lonCell(bottom_left))/(lonCell(top_left) - lonCell(bottom_left))

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
               wgtMatrix(iLat+iLon+iDep-2, 10) = distance_bottom_left
               wgtMatrix(iLat+iLon+iDep-2, 11) = distance_bottom_right
               wgtMatrix(iLat+iLon+iDep-2, 12) = distance_top_left
               wgtMatrix(iLat+iLon+iDep-2, 13) = distance_top_right

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
                endif
             else
                 if (lat_loc_1(1) .ne. 1) then
                    lat_obs_tmp(lat_loc_1(1):nLat_obs) = 1e32
                 endif
             endif 
             lat_loc_2 = minloc(abs(latCell(iCell)-lat_obs_tmp))
    
             lon_loc_1 = minloc(abs(lonCell(iCell)-lon_obs_tmp))
             lon_obs_tmp(lon_loc_1) = 1e32
             lon_loc_2 = minloc(abs(lonCell(iCell)-lon_obs_tmp))

             !print*, lat_loc_1, lat_loc_2
             !print*, latCell(iCell), lat_obs % array (lat_loc_1(1)), lat_obs % array (lat_loc_2(1))

             if ( latCell(iCell)-lat_obs % array (lat_loc_1(1)) .ge. 0.0 ) then
                if (lat_loc_1(1) .eq. nLat_obs) then
                    xfrac = (latCell(iCell) - lat_obs % array (nLat_obs))/(lat_obs % array (1) + 2*4*ATAN(1.d0) - lat_obs % array (nLat_obs))
                    lat_left_idx = 1
                    lat_right_idx = nLat_obs
                else
                   if (lat_loc_1(1) .eq. lat_loc_2(1)) then
                      xfrac = 0.0
                   else
                      xfrac = (latCell(iCell) - lat_obs % array (lat_loc_1(1)))/(lat_obs % array (lat_loc_2(1)) - lat_obs % array (lat_loc_1(1)))
                   endif
                   lat_left_idx  = lat_loc_1(1)
                   lat_right_idx = lat_loc_2(1)
                endif
             else
                if (lat_loc_1(1) .eq. 1)  then
                    xfrac = ( latCell(iCell) - (lat_obs % array (nLat_obs) - 2*4*ATAN(1.d0)) )/ &
                            ( lat_obs % array (1) - ( lat_obs % array (nLat_obs) - 2*4*ATAN(1.d0) ) )
                    lat_left_idx = nLat_obs
                    lat_right_idx = 1
                else
                   if (lat_loc_1(1) .eq. lat_loc_2(1)) then
                      xfrac = 0.0
                   else
                      xfrac = (latCell(iCell) - lat_obs % array (lat_loc_2(1)))/(lat_obs % array (lat_loc_1(1)) - lat_obs % array (lat_loc_2(1)))
                   endif
                   lat_left_idx  = lat_loc_2(1)
                   lat_right_idx = lat_loc_1(1)
                endif
             endif
    
             if (lonCell(iCell) - lon_obs % array (lon_loc_1(1)) .ge. 0) then
                yfrac = (lonCell(iCell) - lon_obs % array (lon_loc_1(1)))/(lon_obs % array (lon_loc_2(1)) - lon_obs % array (lon_loc_1(1)))
                lon_left_idx  = lon_loc_1(1)
                lon_right_idx = lon_loc_2(1)
             else
                yfrac = (lonCell(iCell) - lon_obs % array (lon_loc_2(1)))/(lon_obs % array (lon_loc_1(1)) - lon_obs % array (lon_loc_2(1)))
                lon_left_idx  = lon_loc_2(1)
                lon_right_idx = lon_loc_1(1)
             endif
    
             dep_obs_tmp = dep_obs % array
             dep_loc_1 = minloc(abs(refZMid(iVertLevels) - dep_obs_tmp ))
             dep_obs_tmp(dep_loc_1) = 1e32
             dep_loc_2 = minloc(abs(refZMid(iVertLevels) - dep_obs_tmp ))

             if (refZMid(iVertLevels) - dep_obs % array (dep_loc_1(1)) .ge. 0) then
                zfrac = (refZMid(iVertLevels) - dep_obs % array (dep_loc_1(1))) / (dep_obs % array (dep_loc_2(1)) - dep_obs % array (dep_loc_1(1)))
                dep_low_idx = dep_loc_1(1)
                dep_high_idx = dep_loc_2(1)
             else
                zfrac = (refZMid(iVertLevels) - dep_obs % array (dep_loc_2(1))) / (dep_obs % array (dep_loc_1(1)) - dep_obs % array (dep_loc_2(1)))
                dep_low_idx = dep_loc_2(1)
                dep_high_idx = dep_loc_1(1)
             endif

             !if (xfrac .gt. 0) then ! .or. yfrac .lt. 0 .or. zfrac .lt. 0) then
             !    print*, latCell(iCell), lat_obs % array(lat_loc_1(1)), lat_obs % array (lat_loc_2(1)), xfrac
             !endif

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

end module ocn_interp_mat

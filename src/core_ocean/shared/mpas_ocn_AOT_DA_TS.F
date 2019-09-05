! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_AOT_DA_TS
!
!> \brief MPAS AOT data assimilation for TS
!> \author  Todd Ringler
!> \date    06/09/2015
!> \details
!>  This module contains routines for assimilating temperature and salinity
!>  observations via the AOT algorithm.
!
!-----------------------------------------------------------------------

module ocn_AOT_DA_TS

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use ocn_constants
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

   public :: ocn_AOT_DA_TS_compute, &
             ocn_AOT_DA_TS_compute_debug, &
             ocn_AOT_DA_TS_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_AOT_DA_TS_compute
!
!> \brief   computes AOT DA term for TS 
!> \author  Todd Ringler
!> \date    06/09/2015
!> \details
!>  This routine computes the AOT DA term for temperature and salinity. 
!
!-----------------------------------------------------------------------

   subroutine ocn_AOT_DA_TS_compute(nTracers, nCellsSolve, maxLevelCell, &
                  tracer_interp, tracer_obs, tracer_mu, tracer_tend, layerThickness,  err)!{{{
      implicit none
      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      ! one dimensional arrays
      integer, dimension(:), intent(in) :: &
         maxLevelCell

      ! two dimensional arrays
      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickness

      ! three dimensional arrays
      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
         tracer_interp, tracer_obs 

      ! scalars
      integer, intent(in) :: nCellsSolve, nTracers
      real (kind=RKIND), intent(in) :: tracer_mu
      
!      real (kind=RKIND), dimension(:,:,:), pointer, intent(in) :: tracerGroup

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
        tracer_tend

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: Error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer :: iCell, iLevel, iTracer, timeLevel

      err = 0

      !$omp do schedule(runtime) private(iLevel, iTracer)
      do iCell=1,nCellsSolve
        do iLevel=1,maxLevelCell(iCell)
          do iTracer=1,nTracers
             tracer_tend(iTracer, iLevel, iCell) = tracer_tend(iTracer, iLevel, iCell)  - layerThickness(iLevel,iCell) * tracer_mu * &
                                                   (tracer_interp(iTracer, iLevel, iCell) -  &
                                                    tracer_obs(iTracer, iLevel, iCell))
          enddo
        enddo
      enddo


      !$omp end do

   !--------------------------------------------------------------------

   end subroutine ocn_AOT_DA_TS_compute!}}}

   subroutine ocn_AOT_DA_TS_compute_debug(nTracers, nCellsSolve, maxLevelCell, &
                  tracer_interp, tracer_obs, tracer_mu, tracer_tend, err)!{{{
      implicit none
      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      ! one dimensional arrays
      integer, dimension(:), intent(in) :: &
         maxLevelCell

!      ! two dimensional arrays
!      real (kind=RKIND), dimension(:,:), intent(in) :: &
!         layerThickness

      ! three dimensional arrays
      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
         tracer_interp
      real (kind=RKIND), intent(in) :: &
         tracer_obs 

      ! scalars
      integer, intent(in) :: nCellsSolve, nTracers
      !integer, pointer, intent(in) :: tracerIndex
      real (kind=RKIND), intent(in) :: tracer_mu
      
!      real (kind=RKIND), dimension(:,:,:), pointer, intent(in) :: tracerGroup

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
        tracer_tend

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: Error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer :: iCell, iLevel, iTracer, timeLevel
      
      err = 0
      
      !$omp do schedule(runtime) private(iLevel, iTracer)
!     print *, 'size of array', shape(tracer_tend)
      do iCell=1,nCellsSolve
        do iLevel=1,maxLevelCell(iCell)
          do iTracer=1,nTracers
             tracer_tend(iTracer, iLevel, iCell) =  tracer_tend(iTracer, iLevel, iCell) - tracer_mu * &
                                                  (tracer_interp(iTracer, iLevel, iCell) -  &
                                                    2.0)!tracer_obs)
           enddo
        enddo
      enddo

      !$omp end do

   !--------------------------------------------------------------------

   end subroutine ocn_AOT_DA_TS_compute_debug!}}}
!***********************************************************************
!
!  routine ocn_AOT_DA_TS_init
!
!> \brief   Initializes ocean AOT DA for TS
!> \author  Todd Ringler
!> \date    06/09/2015
!> \details
!>  This routine initializes fields required for data assimilation of TS
!
!-----------------------------------------------------------------------

   subroutine ocn_AOT_DA_TS_init(tracer_mu, err)!{{{


      integer, intent(out) :: err !< Output: error flag
      real (kind=RKIND), intent(out) :: tracer_mu
      logical, pointer :: config_AOT_DA_on
      real (kind=RKIND), pointer :: config_AOT_DA_mu_TS
      err = 0

      call mpas_pool_get_config(ocnConfigs, 'config_AOT_DA_mu_TS', config_AOT_DA_mu_TS)
      call mpas_pool_get_config(ocnConfigs, 'config_AOT_DA_on', config_AOT_DA_on)

      tracer_mu = 0.0

      if (config_AOT_DA_on) then
          tracer_mu = config_AOT_DA_mu_TS
      endif


   !--------------------------------------------------------------------

   end subroutine ocn_AOT_DA_TS_init!}}}

!***********************************************************************

end module ocn_AOT_DA_TS
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker

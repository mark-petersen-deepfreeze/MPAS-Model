! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_AOT_DA_velocity
!
!> \brief MPAS ocean AOT data assimilation for horizontal velocity
!> \author  Todd Ringler
!> \date    5 January 2012
!> \details
!>  This module contains the routine for updating the horizontal velocity
!>  tendency with the AOT data assimilation algorithm (current version: AOT
!>  only on the normalVelocity, no interpolation)
!>  NOTE: THIS ROUTINE ONLY HOLDS IF THE INTERPOLATOR BACK TO THE MPAS GRID IS
!>  LINEAR!! (currently, these interpolators are all found in the
!>  mpas_ocn_init_interpolation.F file and they are all linear)
!
!-----------------------------------------------------------------------

module ocn_AOT_DA_velocity

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timer
   use ocn_constants

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

   public :: ocn_AOT_DA_velocity_tend, &
             ocn_AOT_DA_debug_velocity_tend, &
             ocn_AOT_DA_velocity_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!   real (kind=RKIND) :: velocityTend_mu 


!***********************************************************************

contains

!***********************************************************************
!
!  ocn_AOT_DA_velocity_tend
!
!> \brief   Computes AOT data assimilation for horizontal velocity
!> \author  Todd Ringler
!> \date    5 January 2012
!> \details
!>  This routine computes the horizontal velocity
!>  tendency with regards to the AOT data assimilation algorithm
!
!-----------------------------------------------------------------------

   subroutine ocn_AOT_DA_velocity_tend(meshPool, velocityX_interp, velocityY_interp, & 
                                        velocityX_interp_obs, velocityY_interp_obs, &
                                        velocityTend_mu, tend, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         velocityX_interp_obs, &  !< Input: interpolated observed x-horizontal velocity
         velocityY_interp_obs, &  !< Input: interpolated observed y-horizontal velocity
         velocityX_interp, &      !< Input: interpolated x-horizontal velocity
         velocityY_interp         !< Input: interpolated y-horizontal velocity

      type (mpas_pool_type), intent(in) :: &
         meshPool          !< Input: mesh information

      real (kind=RKIND), intent(in) :: velocityTend_mu
      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         tend          !< Input/Output: velocity tendency

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer :: iEdge, k, nEdges, cell1, cell2
      integer, dimension(:,:), pointer :: cellsOnEdge
      integer, dimension(:), pointer :: nEdgesArray
      integer, dimension(:), pointer :: maxLevelEdgeTop
      real (kind=RKIND), dimension(:), pointer :: angleEdge
!      real (kind=RKIND), dimension(:), allocatable :: normalVelocity_obs
      !-----------------------------------------------------------------
      !
      ! call relevant routines for computing tendencies
      ! note that the user can choose multiple options and the
      !   tendencies will be added together
      !
      !-----------------------------------------------------------------

      err = 0

      call mpas_timer_start('AOT DA horizontal velocity')
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'angleEdge', angleEdge)
      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)

      nEdges = nEdgesArray( 1 )

!      do iEdge = 1, nEdges
!         do k=1, maxLevelEdgeTop(iEdge)
!            allocate( normalVelocity_obs(k,nEdges) )
!         enddo
!      enddo
!
!      do iEdge = 1, nEdges
!         cell1 = cellsOnEdge(1, iEdge)
!         cell2 = cellsOnEdge(2, iEdge)
!         do k=1, maxLevelEdgeTop(iEdge)
!            normalVelocity_obs(k,iEdge) = cos(angleEdge(iEdge)) * (0.5*(velocityX_interp_obs(k,cell1) + velocityX_interp_obs(k,cell2))) + &
!                                     sin(angleEdge(iEdge)) * (0.5*(velocityY_interp_obs(k,cell1) + velocityY_interp_obs(k,cell2)))
!         enddo
!      enddo
 
               !write(135790,*), 'tendency', maxval(tend)
         !$omp do schedule(runtime) private(k)
         do iEdge = 1, nEdges
          cell1 = cellsOnEdge(1, iEdge)
          cell2 = cellsOnEdge(2, iEdge)
          do k = 1, maxLevelEdgeTop(iEdge)
          !write(1230,*) tend(k,iEdge)
               !tend(k,iEdge) = tend(k,iEdge) - velocityTend_mu *(  (cos(angleEdge(iEdge)) * (0.5*(velocityX_interp(k,cell1) + velocityX_interp(k,cell2))) + &
               !                   sin(angleEdge(iEdge)) * (0.5*(velocityY_interp(k,cell1) + velocityY_interp(k,cell2)))  ) - & 
               !               (  cos(angleEdge(iEdge)) * (0.5*(velocityX_interp_obs(k,cell1) + velocityX_interp_obs(k,cell2))) + &
               !                   sin(angleEdge(iEdge)) * (0.5*(velocityY_interp_obs(k,cell1) + velocityY_interp_obs(k,cell2)))  )   )
               tend(k,iEdge) = tend(k,iEdge) + velocityTend_mu * & 
                              (  cos(angleEdge(iEdge)) * (0.5*(velocityX_interp_obs(k,cell1) - velocityX_interp(k,cell1) + velocityX_interp_obs(k,cell2) - velocityX_interp(k,cell2))) + &
                                  sin(angleEdge(iEdge)) * (0.5*(velocityY_interp_obs(k,cell1) - velocityY_interp(k,cell1) + velocityY_interp_obs(k,cell2) - velocityY_interp(k,cell2)))     )
!               write(1231,*), 'normal vel', (cos(angleEdge(iEdge)) * &
!(0.5*(velocityX_interp(k,cell1) + velocityX_interp(k,cell2))) + &
!                                  sin(angleEdge(iEdge)) * &
!(0.5*(velocityY_interp(k,cell1) + velocityY_interp(k,cell2))) )
!               write(1232,*), 'normal vel obs', (cos(angleEdge(iEdge)) * &
!(0.5*(velocityX_interp_obs(k,cell1) + velocityX_interp_obs(k,cell2))) + &
!                                  sin(angleEdge(iEdge)) * &
!(0.5*(velocityY_interp_obs(k,cell1) + velocityY_interp_obs(k,cell2))) )
               !  write(135791,*) 'observed min,max, ', minval(cos(angleEdge(iEdge)) * (0.5*(velocityX_interp(k,cell1) + velocityX_interp(k,cell2))) + sin(angleEdge(iEdge)) * (0.5*(velocityY_interp(k,cell1) + velocityY_interp(k,cell2))) ), maxval(cos(angleEdge(iEdge)) * (0.5*(velocityX_interp(k,cell1) + velocityX_interp(k,cell2))) + sin(angleEdge(iEdge)) * (0.5*(velocityY_interp(k,cell1) + velocityY_interp(k,cell2))) )
               ! write(135792,*) 'interp min, max, ', minval(cos(angleEdge(iEdge)) * (0.5*(velocityX_interp_obs(k,cell1) + velocityX_interp_obs(k,cell2))) + sin(angleEdge(iEdge)) * (0.5*(velocityY_interp_obs(k,cell1) + velocityY_interp_obs(k,cell2))) ), maxval(cos(angleEdge(iEdge)) * (0.5*(velocityX_interp_obs(k,cell1) + velocityX_interp_obs(k,cell2))) + sin(angleEdge(iEdge)) * (0.5*(velocityY_interp_obs(k,cell1) + velocityY_interp_obs(k,cell2))) )
           enddo
         enddo
                !write(135791,*), 'observed min, max, X', minval(velocityX_interp_obs), maxval(velocityX_interp_obs)
                !write(135792,*), 'observed min, max, Y', minval(velocityY_interp_obs), maxval(velocityY_interp_obs)
                !write(135793,*), 'interp min, max, X', minval(velocityX_interp), maxval(velocityX_interp)
                !write(135794,*), 'interp min, max, Y', minval(velocityY_interp), maxval(velocityY_interp)
                !write(135795,*), 'tendency new', maxval(tend)
         !$omp end do

      call mpas_timer_stop('AOT DA horizontal velocity')

   !--------------------------------------------------------------------

   end subroutine ocn_AOT_DA_velocity_tend!}}}

!***********************************************************************
!
!  ocn_AOT_DA_debug_velocity_tend
!
!> \brief   Computes AOT data assimilation for horizontal velocity
!> \author  Todd Ringler
!> \date    5 January 2012
!> \details
!>  This routine computes the horizontal velocity
!>  tendency with regards to the AOT data assimilation algorithm
!
!-----------------------------------------------------------------------

   subroutine ocn_AOT_DA_debug_velocity_tend(meshPool, normalVelocity, & 
                                        velocityX_interp_obs, velocityY_interp_obs, &
                                        velocityTend_mu, tend, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalVelocity        !< Input: normal velocity
      real (kind=RKIND), intent(in) :: &
         velocityX_interp_obs, &  !< Input: x-horizontal velocity
         velocityY_interp_obs     !< Input: y-horizontal velocity

      type (mpas_pool_type), intent(in) :: &
         meshPool          !< Input: mesh information

      real (kind=RKIND), intent(in) :: velocityTend_mu
      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         tend          !< Input/Output: velocity tendency

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer :: iEdge, k, nEdges, cell1, cell2
      integer, dimension(:,:), pointer :: cellsOnEdge
      integer, dimension(:), pointer :: nEdgesArray
      integer, dimension(:), pointer :: maxLevelEdgeTop
      real (kind=RKIND), dimension(:), pointer :: angleEdge
      !-----------------------------------------------------------------
      !
      ! call relevant routines for computing tendencies
      ! note that the user can choose multiple options and the
      !   tendencies will be added together
      !
      !-----------------------------------------------------------------

      err = 0

      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'angleEdge', angleEdge)
      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)

      nEdges = nEdgesArray( 1 )

!      do iEdge = 1, nEdges
!         do k=1, maxLevelEdgeTop(iEdge)
!            allocate( normalVelocity_obs(k,nEdges) )
!         enddo
!      enddo
!
!      do iEdge = 1, nEdges
!         cell1 = cellsOnEdge(1, iEdge)
!         cell2 = cellsOnEdge(2, iEdge)
!         do k=1, maxLevelEdgeTop(iEdge)
!            normalVelocity_obs(k,iEdge) = cos(angleEdge(iEdge)) * (0.5*(velocityX_interp_obs(k,cell1) + velocityX_interp_obs(k,cell2))) + &
!                                     sin(angleEdge(iEdge)) * (0.5*(velocityY_interp_obs(k,cell1) + velocityY_interp_obs(k,cell2)))
!         enddo
!      enddo
 
         !$omp do schedule(runtime) private(k)

         do iEdge = 1, nEdges
          do k = 1, maxLevelEdgeTop(iEdge)
               tend(k,iEdge) = tend(k,iEdge) + velocityTend_mu * & 
                              (  (cos(angleEdge(iEdge))  + &
                                  sin(angleEdge(iEdge))  ) &
                                - normalVelocity(k,iEdge))
           enddo
         enddo
         print *, 'max tend =', max(maxval(tend), -minval(tend))
         !$omp end do


   !--------------------------------------------------------------------

   end subroutine ocn_AOT_DA_debug_velocity_tend!}}}


!***********************************************************************
!
!  ocn_vel_forcing_rayleigh_init
!
!> \brief   Initializes ocean Rayleigh friction
!> \author  Todd Ringler
!> \date    5 January 2012
!> \details
!>  This routine initializes quantities related to
!>  in the ocean.
!
!-----------------------------------------------------------------------

   subroutine ocn_AOT_DA_velocity_init(velocityTend_mu, err)!{{{

   !--------------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! call individual init routines for each parameterization
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag
      real (kind=RKIND), intent(out) :: velocityTend_mu
      logical, pointer :: config_AOT_DA_on
      real (kind=RKIND), pointer :: config_AOT_DA_mu_vel
      err = 0

      call mpas_pool_get_config(ocnConfigs, 'config_AOT_DA_mu_vel', config_AOT_DA_mu_vel)
      call mpas_pool_get_config(ocnConfigs, 'config_AOT_DA_on', config_AOT_DA_on)

      velocityTend_mu = 0.0

      if (config_AOT_DA_on) then
          velocityTend_mu = config_AOT_DA_mu_vel
      endif

   !--------------------------------------------------------------------

   end subroutine ocn_AOT_DA_velocity_init!}}}

!***********************************************************************

end module ocn_AOT_DA_velocity

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker

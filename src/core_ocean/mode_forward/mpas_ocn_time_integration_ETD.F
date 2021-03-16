! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_integration_ETD
!
!> \brief MPAS ocean ETD Time integration scheme for tracers
!> \author Sara Calandrini
!> \date   October 2020
!> \details
!>  This module contains the ETD time integration routine for tracers. 
!
!-----------------------------------------------------------------------

module ocn_time_integration_ETD

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_dmpar
   use mpas_threading
   use mpas_vector_reconstruction
   use mpas_spline_interpolation
   use mpas_timer

   use ocn_constants
   use ocn_tendency
   use ocn_diagnostics
   use ocn_gm

   use ocn_equation_of_state
   use ocn_vmix
   use ocn_time_average_coupled
   use ocn_wetting_drying

   use ocn_effective_density_in_land_ice

   use mpas_tracer_advection_helpers

   implicit none
   private
   save

   ! private module variables
   real (kind=RKIND) :: &
      coef3rdOrder       !< coefficient for blending high-order terms

   integer :: vertOrder  !< choice of order for vertical advection
   integer, parameter :: &! enumerator for supported vertical adv order
      vertOrder2=2,      &!< 2nd order
      vertOrder3=3,      &!< 3rd order
      vertOrder4=4        !< 4th order

   !logical :: del2On
   !logical :: tracerVmixOn
   real (kind=RKIND) :: eddyDiff2

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

   public :: ocn_time_integrator_ETD

   contains

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_time_integrator_ETD
!
!> \brief MPAS ocean ETD Time integration scheme for tracers
!> \author Sara Calandrini
!> \date   October 2020
!> \details
!>  This routine integrates one timestep (dt) using an ETD time integrator.
!
!-----------------------------------------------------------------------

   subroutine ocn_time_integrator_ETD(domain, dt) 
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! Advance model state forward in time by the specified time step using
   ! a 2nd order exponential integrator 
   !
   ! Input: domain - current model state in time level 1 (e.g.,
   ! time_levs(1)state%h(:,:)) plus mesh meta-data
   ! Output: domain - upon exit, time level 2 (e.g., time_levs(2)%state%h(:,:))
   ! contains  model state advanced forward in time by dt seconds
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      implicit none

      type (domain_type), intent(inout) :: domain !< Input/Output: domain information
      real (kind=RKIND), intent(in) :: dt !< Input: timestep
      type (block_type), pointer :: block
   
      type (mpas_pool_type), pointer :: tendPool 
      type (mpas_pool_type), pointer :: statePool
      type (mpas_pool_type), pointer :: tracersPool
      type (mpas_pool_type), pointer :: meshPool
      type (mpas_pool_type), pointer :: diagnosticsPool
      type (mpas_pool_type), pointer :: verticalMeshPool
      type (mpas_pool_type), pointer :: forcingPool
      type (mpas_pool_type), pointer :: scratchPool
      type (mpas_pool_type), pointer :: swForcingPool 
      type (mpas_pool_type), pointer :: tracersSurfaceFluxPool
      type (mpas_pool_type), pointer :: ETDPool
      type (mpas_pool_type), pointer :: tracersTendPool
 
      real (kind=RKIND), dimension(:,:,:), pointer :: TracerGroupTend !< [in,out] Tracer tendency to which advection added
      real (kind=RKIND), dimension(:,:,:), pointer :: tracerGroup !< [in] Current tracer values
      !real (kind=RKIND), dimension(:,:), pointer :: normalThicknessFlux !< [in] Thickness weighted horizontal velocity
      real (kind=RKIND), dimension(:,:), pointer :: normalTransportVelocity 
      real (kind=RKIND), dimension(:), pointer :: sshCur 
      real (kind=RKIND), dimension(:,:), pointer :: vertAleTransportTop !< [in] Vertical velocity
      real (kind=RKIND), dimension(:,:), pointer :: layerThickness !< [in] Thickness field 
      real (kind=RKIND), dimension(:,:), pointer ::layerThicknessEdge !< [in] Thickness at edge 
      real (kind=RKIND), dimension(:,:), pointer :: vertDiffTopOfCell !< [in] Vertical mixing coefficients
      real (kind=RKIND), dimension(:,:,:), pointer ::  vertNonLocalFlux  !< [in] Non local flux at interfaces
      real (kind=RKIND), dimension(:,:), pointer :: tracerGroupSurfaceFlux !< [in] Surface flux for tracers nonlocal computation
      real (kind=RKIND), dimension(:), pointer :: JacZ 
      real (kind=RKIND), dimension(:,:,:), pointer :: phi1JTot 
      logical, pointer :: config_cvmix_kpp_nonlocal_with_implicit_mix

      !type (field2DReal), pointer :: normalThicknessFluxField
      type (field2DReal), pointer :: normalizedRelativeVorticityEdgeField

      real (kind=RKIND), dimension(:,:), pointer, contiguous :: &
         advCoefs, advCoefs3rd ! advection coefficients
      integer, dimension(:), pointer, contiguous :: &
         maxLevelCell,    &! index of max level at each cell
         maxLevelEdgeTop, &! max level at edge with both cells active
         nAdvCellsForEdge  ! number of advective cells for each edge
      integer, dimension(:,:), pointer, contiguous :: &!
         highOrderAdvectionMask, &! mask for higher order contributions
         edgeSignOnCell,  &! sign at cell edge for fluxes
         advCellsForEdge   ! index of advective cells for each edge
      integer, pointer:: nCells, nEdges, nVertLevels
      real (kind=RKIND), dimension(:), pointer :: xCell, yCell, zCell
      real (kind=RKIND), dimension(:,:), pointer :: velocityCell

      real (kind=RKIND) :: analytic
      integer :: k, err1, err, iCell, iEdge, iTracer, CFL_pow, NLayers, num_tracers   
      real (kind=RKIND), dimension(:,:,:), allocatable :: TracerGroupTendHor
      real (kind=RKIND), dimension(:,:), allocatable :: phi1J, normalThicknessFlux
      real (kind=RKIND), dimension(:), allocatable :: rhs, tracer_cur, horiz_1stage, horiz 

      CFL_pow = 20  
      vertOrder = 2
      coef3rdOrder = 0.0_RKIND !0.25
      !eddyDiff2 = 10.0 
      eddyDiff2 = 0.0001_RKIND

      print*, 'dt ', dt

      !--------------------
      ! FIRST STAGE
      !--------------------

      block => domain % blocklist
      do while (associated(block))

         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
         call mpas_pool_get_subpool(tendPool, 'tracersTend', tracersTendPool)
         call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
         call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)
         call mpas_pool_get_subpool(forcingPool, 'tracersSurfaceFlux', tracersSurfaceFluxPool)
         call mpas_pool_get_subpool(block % structs, 'ETD', ETDPool)
         call mpas_pool_get_subpool(block % structs, 'verticalMesh', verticalMeshPool)

         call mpas_threading_barrier()

         call mpas_pool_get_dimension(block % dimensions, 'nCells', nCells)
         call mpas_pool_get_dimension(block % dimensions, 'nEdges', nEdges)

         call mpas_pool_get_array(diagnosticsPool, 'normalTransportVelocity', normalTransportVelocity)
         call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)
         call mpas_pool_get_array(diagnosticsPool, 'vertAleTransportTop', vertAleTransportTop)
         call mpas_pool_get_array(diagnosticsPool, 'vertNonLocalFlux', vertNonLocalFlux)
         call mpas_pool_get_array(diagnosticsPool, 'vertDiffTopOfCell', vertDiffTopOfCell)   
         call mpas_pool_get_array(diagnosticsPool, 'velocityCell', velocityCell)
 
         call mpas_pool_get_array(tracersSurfaceFluxPool, 'SurfaceFlux', tracerGroupSurfaceFlux)

         call mpas_pool_get_array(statePool, 'ssh', sshCur, 1)
         call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)

         call mpas_pool_get_array(meshPool, 'advCoefs3rd', advCoefs3rd)
         call mpas_pool_get_array(meshPool, 'advCoefs', advCoefs)
         call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
         call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
         call mpas_pool_get_array(meshPool, 'highOrderAdvectionMask', highOrderAdvectionMask)
         call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)
         call mpas_pool_get_array(meshPool, 'nAdvCellsForEdge', nAdvCellsForEdge)
         call mpas_pool_get_array(meshPool, 'advCellsForEdge', advCellsForEdge)
         call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
         call mpas_pool_get_array(meshPool, 'xCell', xCell)
         call mpas_pool_get_array(meshPool, 'yCell', yCell)
         call mpas_pool_get_array(meshPool, 'zCell', zCell)

         call mpas_pool_get_array(tracersPool, 'debugTracers', tracerGroup, 1) 
         call mpas_pool_get_array(tracersTendPool, 'debugTracersTend', TracerGroupTend)
         !call mpas_pool_get_array(tracersPool, 'activeTracers', tracerGroup, 1)
         !call mpas_pool_get_array(tracersTendPool, 'activeTracersTend', TracerGroupTend)

         call mpas_pool_get_array(ETDPool, 'JacZ', JacZ)
         call mpas_pool_get_array(ETDPool, 'phi1JTot', phi1JTot)

         num_tracers = size(tracerGroup,dim=1)
         allocate(TracerGroupTendHor(num_tracers, nVertLevels, nCells))  
         allocate(phi1J(nVertLevels, nVertLevels)) 
         allocate(normalThicknessFlux(nVertLevels, nEdges+1))
         allocate(rhs(nVertLevels))
         allocate(tracer_cur(nVertLevels))
         allocate(horiz_1stage(nVertLevels))
         allocate(horiz(nVertLevels))

         do iCell = 1, nCells
            TracerGroupTend(:,:,iCell) = 0.0_RKIND
            TracerGroupTendHor(:,:,iCell) = 0.0_RKIND
         end do
 
         !
         ! transport velocity for the tracer.
         !
         !$omp do schedule(runtime) private(k)
         do iEdge = 1, nEdges
            do k = 1, nVertLevels
               normalThicknessFlux(k, iEdge) = normalTransportVelocity(k, iEdge) * layerThicknessEdge(k, iEdge)
            end do
         end do
         !$omp end do
 
         !Halo update for pv_edge
         call mpas_dmpar_field_halo_exch(domain, 'normalizedRelativeVorticityEdge')

         !Computation of the horizontal advection
         call ocn_tracer_hor_advection_std(tracerGroup, advCoefs, advCoefs3rd, &
            nAdvCellsForEdge, advCellsForEdge, normalThicknessFlux, layerThickness, &
            layerThickness, dt, meshPool, TracerGroupTend, maxLevelCell, maxLevelEdgeTop, &
            highOrderAdvectionMask, edgeSignOnCell)
         !Computation of the horizontal diffusion 
         !call ocn_tracer_hdiff_del2_tend(meshPool, layerThicknessEdge, tracerGroup, TracerGroupTend, err1)

         !Halo update tracer tendencies 
         call mpas_dmpar_field_halo_exch(domain, 'debugTracersTend')
         !call mpas_dmpar_field_halo_exch(domain, 'activeTracersTend')

         do iCell = 1, nCells
            TracerGroupTendHor(:,:,iCell)=TracerGroupTend(:,:,iCell)
         end do     

         !Computation of w 
         call ocn_vert_transport_velocity_top(meshPool, verticalMeshPool, scratchPool, & 
                 layerThickness, layerThicknessEdge, normalTransportVelocity,&
                 sshCur, dt, vertAleTransportTop, err)

         !do k = 2, nVertLevels
         !   vertAleTransportTop(k,1) = (-(4.0/(250.0**4))*(xCell(1)-250.0)**3)*(1.0-((-25.0*(k-1)+250.0)**2)/(250.0**2))
         !   vertAleTransportTop(k,floor(0.5*nCells)) = (-(4.0/(250.0**4))*(xCell(floor(0.5*nCells))-250.0)**3)*(1.0-((-25.0*(k-1)+250.0)**2)/(250.0**2))
         !   vertAleTransportTop(k,floor(0.5*nCells)+1) = (-(4.0/(250.0**4))*(xCell(floor(0.5*nCells)+1)-250.0)**3)*(1.0-((-25.0*(k-1)+250.0)**2)/(250.0**2))
         !   vertAleTransportTop(k,nCells) = (-(4.0/(250.0**4))*(xCell(nCells)-250.0)**3)*(1.0-((-25.0*(k-1)+250.0)**2)/(250.0**2))
         !end do

         !do k = 2, nVertLevels
         !   do iCell = 1, nCells
         !      !k=2
         !      analytic = - (-(4.0/(251.25**4))*(xCell(iCell)-251.25)**3)*(1.0-((-25.125*(k-1)+251.25)**2)/(251.25**2)) 
         !      print*, abs(vertAleTransportTop(k,iCell) - analytic)
         !      !print*, xCell(iCell) 
         !      !print*, vertAleTransportTop(k,iCell)
         !   end do
         !   print*, '----------------------------------------'
         !end do
         !print*, '----------------------------------------'
         !print*, '------------FINE ITERAZIONE-------------'
         !print*, '----------------------------------------' 

         !do iCell = 1,nCells
         !   vertAleTransportTop(1,iCell) = 0.0_RKIND
         !   vertAleTransportTop(maxLevelCell(iCell)+1,iCell) = 0.0_RKIND
         !   do k = 2, maxLevelCell(iCell)
         !      vertAleTransportTop(k,iCell) = (-(4.0/(250.0**4))*(xCell(iCell)-250.0)**3)*(1.0-((-25.0*(k-1)+250.0)**2)/(250.0**2))     
         !   end do
         !end do
 
         !Computation of the vertical advection 
         call ocn_tracer_vert_advection_std(tracerGroup, advCoefs, advCoefs3rd, &
            nAdvCellsForEdge, advCellsForEdge, normalThicknessFlux, vertAleTransportTop, layerThickness, &
            layerThickness, dt, meshPool, TracerGroupTend, maxLevelCell, maxLevelEdgeTop, &
            highOrderAdvectionMask, edgeSignOnCell)
         !Computation of the vertical diffusion  
         !call ocn_tracer_vert_diff_tend(meshPool, dt, vertDiffTopOfCell, layerThickness, tracerGroup, TracerGroupTend, &
         !            vertNonLocalFlux, tracerGroupSurfaceFlux, config_cvmix_kpp_nonlocal_with_implicit_mix, err)
         
         !Halo update tracer tendencies 
         call mpas_dmpar_field_halo_exch(domain, 'debugTracersTend')
         !call mpas_dmpar_field_halo_exch(domain, 'activeTracersTend') 

         !call mpas_pool_get_subpool(domain % blocklist % structs, 'tend', tendPool)
         !call mpas_pool_get_subpool(tendPool, 'tracersTend', tracersTendPool)
         !call mpas_pool_begin_iteration(tracersTendPool)
         !do while ( mpas_pool_get_next_member(tracersTendPool, groupItr) )
            !if ( groupItr % memberType == MPAS_POOL_FIELD ) then
            !   call mpas_dmpar_field_halo_exch(domain, trim(groupItr %memberName))
            !end if
         !end do

         do iTracer = 1, num_tracers 
            do iCell = 1, nCells 
               Nlayers = maxLevelCell(iCell)
               phi1J(:,:) = 0.0_RKIND
               rhs(:) = TracerGroupTend(iTracer, :, iCell)
               tracer_cur(:) = tracerGroup(iTracer, :, iCell)
               tracer_cur(:) = tracer_cur(:) * layerThickness(:,iCell)
               call jacobianVert(meshPool, ETDPool, vertDiffTopOfCell, layerThickness, vertAleTransportTop, iCell) 
               !print*, 'JacZ ', JacZ(:) 
               call ocn_phi_function(ETDPool, CFL_pow, phi1J, NLayers, dt)
               !do k = 1, Nlayers
               !   print*, 'phi1J ' , phi1J(k,:)
               !end do 
               phi1JTot(iCell,:,:) = phi1J(:,:) 
             
               CALL DGEMM('N','N',NLayers,1,NLayers,dt,phi1J,Nlayers,rhs,NLayers,1.0_RKIND,tracer_cur,NLayers) 
               do k = 1, Nlayers               
                  if (layerThickness(k,iCell)==0.0) then 
                     tracerGroup(iTracer, k, iCell) = -1e+34
                  else
                     tracerGroup(iTracer, k, iCell) = tracer_cur(k) / layerThickness(k,iCell)
                  end if
               end do
               !tracerGroup(iTracer, :, iCell) = tracer_cur(:) / layerThickness(:,iCell)  
            end do 
         end do
         !do iCell = 1, nCells
         !   print*, 'tracer 1 iCell ', tracerGroup(1, :, iCell) 
         !end do

         !Halo update tracer values
         call mpas_dmpar_field_halo_exch(domain, 'debugTracers', timeLevel=2)
         !call mpas_dmpar_field_halo_exch(domain, 'activeTracers', timeLevel=2) 

         !call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         !call mpas_pool_begin_iteration(tracersPool)
         !do while ( mpas_pool_get_next_member(tracersPool, groupItr) )
         !   if ( groupItr % memberType == MPAS_POOL_FIELD ) then
         !      call mpas_dmpar_field_halo_exch(domain, groupItr % memberName, timeLevel=2)
         !   end if
         !end do

         block => block % next
      end do 

      print*,'first stage complete'

      !--------------------
      !SECOND STAGE
      !--------------------

      block => domain % blocklist
         do while (associated(block))
       
         call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
         call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
         call mpas_pool_get_subpool(block % structs, 'forcing', forcingPool)
         call mpas_pool_get_subpool(forcingPool, 'tracersSurfaceFlux', tracersSurfaceFluxPool)

         call mpas_threading_barrier()

         call mpas_pool_get_dimension(block % dimensions, 'nCells', nCells)
         call mpas_pool_get_dimension(block % dimensions, 'nEdges', nEdges)
         call mpas_pool_get_array(diagnosticsPool, 'normalTransportVelocity', normalTransportVelocity)
         call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)
        
         call mpas_pool_get_array(diagnosticsPool, 'vertNonLocalFlux', vertNonLocalFlux)
         call mpas_pool_get_array(diagnosticsPool, 'vertDiffTopOfCell', vertDiffTopOfCell)

         call mpas_pool_get_array(tracersSurfaceFluxPool, 'SurfaceFlux', tracerGroupSurfaceFlux)

         call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)
         call mpas_pool_get_array(meshPool, 'advCoefs3rd', advCoefs3rd)
         call mpas_pool_get_array(meshPool, 'advCoefs', advCoefs)
         call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
         call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
         call mpas_pool_get_array(meshPool, 'highOrderAdvectionMask', highOrderAdvectionMask)
         call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)
         call mpas_pool_get_array(meshPool, 'nAdvCellsForEdge', nAdvCellsForEdge)
         call mpas_pool_get_array(meshPool, 'advCellsForEdge', advCellsForEdge)
         call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)

         call mpas_pool_get_array(tracersPool, 'debugTracers', tracerGroup, 1)
         call mpas_pool_get_array(tracersTendPool, 'debugTracersTend', TracerGroupTend)
         !call mpas_pool_get_array(tracersPool, 'activeTracers', tracerGroup, 1)
         !call mpas_pool_get_array(tracersTendPool, 'activeTracersTend', TracerGroupTend)

         num_tracers = size(tracerGroup,dim=1)     

         !
         ! transport velocity for the tracer.
         !
         !$omp do schedule(runtime) private(k)
         do iEdge = 1, nEdges
            do k = 1, nVertLevels
               normalThicknessFlux(k, iEdge) = normalTransportVelocity(k, iEdge) * layerThicknessEdge(k, iEdge)
            end do
         end do
         !$omp end do
 
         do iCell = 1, nCells
            TracerGroupTend(:,:, iCell) = 0.0_RKIND 
         end do
         rhs(:) = 0.0_RKIND

         !Halo update for pv_edge
         call mpas_dmpar_field_halo_exch(domain, 'normalizedRelativeVorticityEdge')

         !Computation of the horizontal advection
         call ocn_tracer_hor_advection_std(tracerGroup, advCoefs, advCoefs3rd, &
            nAdvCellsForEdge, advCellsForEdge, normalThicknessFlux, layerThickness, &
            layerThickness, dt, meshPool, TracerGroupTend, maxLevelCell, maxLevelEdgeTop, &
            highOrderAdvectionMask, edgeSignOnCell)
         !Computation of the horizontal  diffusion 
         !call ocn_tracer_hdiff_del2_tend(meshPool, layerThicknessEdge, tracerGroup, TracerGroupTend, err1)

         !Halo update tracer tendencies 
         call mpas_dmpar_field_halo_exch(domain, 'debugTracersTend') 
         !call mpas_dmpar_field_halo_exch(domain, 'activeTracersTend')

         do iTracer = 1, num_tracers
            do iCell = 1, nCells
               Nlayers = maxLevelCell(iCell)
               phi1J(:,:) = phi1JTot(iCell,:,:)
               tracer_cur(:) = tracerGroup(iTracer, :, iCell) * layerThickness(:,iCell)
               horiz_1stage(:) = TracerGroupTend(iTracer,:,iCell)
               horiz(:) = TracerGroupTendHor(iTracer,:,iCell)
               do k = 1, Nlayers
                  rhs(k) = horiz_1stage(k) - horiz(k)
               end do 
            
               CALL DGEMM('N','N',NLayers,1,NLayers,0.5_RKIND*dt,phi1J,Nlayers,rhs,NLayers,1.0_RKIND,tracer_cur,NLayers)
               do k = 1, Nlayers               
                  if (layerThickness(k,iCell)==0.0) then 
                     tracerGroup(iTracer, k, iCell) = -1e+34
                  else
                     tracerGroup(iTracer, k, iCell) = tracer_cur(k) / layerThickness(k,iCell)
                  end if
               end do
               !tracerGroup(iTracer, :, iCell) = tracer_cur(:) / layerThickness(:,iCell)
            end do
         end do

         !Halo update tracer values
         call mpas_dmpar_field_halo_exch(domain, 'debugTracers', timeLevel=2)
         !call mpas_dmpar_field_halo_exch(domain, 'activeTracers', timeLevel=2)

         block => block % next
      end do

      print*,'second stage complete'

      deallocate(phi1J, rhs, tracer_cur, TracerGroupTendHor, horiz_1stage, horiz, normalThicknessFlux)

   end subroutine ocn_time_integrator_ETD 

   subroutine ocn_tracer_hor_advection_std(tracers, adv_coefs, adv_coefs_3rd, nAdvCellsForEdge, advCellsForEdge, &!{{{
                                             normalThicknessFlux, layerThickness, verticalCellSize, dt, meshPool, &
                                             tend, maxLevelCell, maxLevelEdgeTop, &
                                             highOrderAdvectionMask, edgeSignOnCell)

   !|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   !
   !  routine ocn_tracer_hor_advection_std
   ! 
   !>  This routine computes the standard tracer horizontal advection tendencity. 
   !
   !-----------------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers !< Input: current tracer values
      real (kind=RKIND), dimension(:,:), intent(in) :: adv_coefs !< Input: Advection coefficients for 2nd order advection
      real (kind=RKIND), dimension(:,:), intent(in) :: adv_coefs_3rd !< Input: Advection coeffs for blending in 3rd/4th order
      integer, dimension(:), intent(in) :: nAdvCellsForEdge !< Input: Number of advection cells for each edge
      integer, dimension(:,:), intent(in) :: advCellsForEdge !< Input: List of advection cells for each edge
      real (kind=RKIND), dimension(:,:), intent(in) :: normalThicknessFlux !<Input: Thichness weighted velocity 
      real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness !< Input:Thickness
      real (kind=RKIND), dimension(:,:), intent(in) :: verticalCellSize !<Input: Distance between vertical interfaces of a cell
      real (kind=RKIND), intent(in) :: dt !< Input: Timestep
      type (mpas_pool_type), intent(in) :: meshPool !< Input: Mesh information
      real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend !<Input/Output: Tracer tendency
      integer, dimension(:), pointer :: maxLevelCell !< Input: Index to max level at cell center
      integer, dimension(:), pointer :: maxLevelEdgeTop !< Input: Index to max level at edge with non-land cells on both sides
      integer, dimension(:,:), pointer :: highOrderAdvectionMask !< Input: Mask for high order advection
      integer, dimension(:, :), pointer :: edgeSignOnCell !< Input: Sign for flux from edge on each cell.

      integer :: i, iCell, iEdge, k, iTracer, cell1, cell2
      integer :: nVertLevels, num_tracers
      integer, pointer :: nCells, nEdges, nCellsSolve, maxEdges
      integer, dimension(:), pointer :: nEdgesOnCell
      integer, dimension(:,:), pointer :: cellsOnEdge, cellsOnCell, edgesOnCell

      real (kind=RKIND) :: tracer_weight, invAreaCell1
      real (kind=RKIND), dimension(:), pointer :: dvEdge, areaCell 
      real (kind=RKIND), dimension(:,:), allocatable :: tracer_cur, high_order_horiz_flux 

      real (kind=RKIND), parameter :: eps = 1.e-10_RKIND

      ! Get dimensions
      call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
      call mpas_pool_get_dimension(meshPool, 'maxEdges', maxEdges)
      nVertLevels = size(tracers,dim=2)
      num_tracers = size(tracers,dim=1)

      ! Initialize pointers
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell) 

      allocate(high_order_horiz_flux(nVertLevels,nEdges))
      allocate(tracer_cur(nVertLevels,nCells+1)) 

      call mpas_threading_barrier()

      !print*, 'coef3rdOrder ', coef3rdOrder

      ! Loop over tracers. One tracer is advected at a time. It is copied into a
      ! temporary array in order to improve locality
      do iTracer = 1, num_tracers
        ! Initialize variables for use in this iTracer iteration
        !$omp do schedule(runtime)
        do iCell = 1, nCells
           tracer_cur(:, iCell) = tracers(iTracer, :, iCell) 
        end do
        !$omp end do

        !$omp do schedule(runtime)
        do iEdge = 1, nEdges
           high_order_horiz_flux(:, iEdge) = 0.0_RKIND
        end do
        !$omp end do

        !  Compute the high order horizontal flux
        !$omp do schedule(runtime) private(cell1, cell2, k, tracer_weight, i, iCell)
        do iEdge = 1, nEdges
          cell1 = cellsOnEdge(1, iEdge)
          cell2 = cellsOnEdge(2, iEdge)

          ! Compute 2nd order fluxes where needed.
          do k = 1, maxLevelEdgeTop(iEdge)
            tracer_weight = iand(highOrderAdvectionMask(k, iEdge)+1, 1) * (dvEdge(iEdge) * 0.5_RKIND) &
                           * normalThicknessFlux(k, iEdge)

            high_order_horiz_flux(k, iEdge) = high_order_horiz_flux(k, iedge) + tracer_weight &
                                            * (tracer_cur(k, cell1) + tracer_cur(k, cell2))
          end do ! k loop

          ! Compute 3rd or 4th fluxes where requested.
          do i = 1, nAdvCellsForEdge(iEdge) 
            iCell = advCellsForEdge(i,iEdge)
            do k = 1, maxLevelCell(iCell)
              tracer_weight = highOrderAdvectionMask(k, iEdge) * (adv_coefs(i,iEdge) + coef3rdOrder &
                            * sign(1.0_RKIND,normalThicknessFlux(k,iEdge))*adv_coefs_3rd(i,iEdge))

              tracer_weight = normalThicknessFlux(k,iEdge)*tracer_weight
              high_order_horiz_flux(k,iEdge) = high_order_horiz_flux(k,iEdge) + tracer_weight * tracer_cur(k,iCell)
            end do ! k loop
          end do ! i loop over nAdvCellsForEdge
        end do ! iEdge loop
        !$omp end do

        ! Accumulate the scaled high order horizontal tendencies
        !$omp do schedule(runtime) private(invAreaCell1, i, iEdge, k)
        do iCell = 1, nCells
          invAreaCell1 = 1.0_RKIND / areaCell(iCell)
          do i = 1, nEdgesOnCell(iCell)
            iEdge = edgesOnCell(i, iCell)
            do k = 1, maxLevelEdgeTop(iEdge)
              tend(iTracer, k, iCell) = tend(iTracer, k, iCell) + edgeSignOnCell(i, iCell) * high_order_horiz_flux(k, iEdge) &
                                      * invAreaCell1
            end do
          end do
        end do
        !$omp end do

      end do ! iTracer loop

      call mpas_threading_barrier()

      deallocate(tracer_cur, high_order_horiz_flux)

   end subroutine ocn_tracer_hor_advection_std!}}}

   subroutine ocn_tracer_vert_advection_std(tracers, adv_coefs, adv_coefs_3rd, nAdvCellsForEdge, advCellsForEdge, &!{{{
                                             normalThicknessFlux, w, layerThickness, verticalCellSize, dt, meshPool, &
                                             tend, maxLevelCell, maxLevelEdgeTop, &
                                             highOrderAdvectionMask, edgeSignOnCell)

   !|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
   !
   !  routine ocn_tracer_vert_advection_std
   ! 
   !>  This routine computes the standard vertical tracer advection tendencity.
   !
   !-----------------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers !< Input: current tracer values
      real (kind=RKIND), dimension(:,:), intent(in) :: adv_coefs !< Input: Advection coefficients for 2nd order advection
      real (kind=RKIND), dimension(:,:), intent(in) :: adv_coefs_3rd !< Input: Advection coeffs for blending in 3rd/4th order
      integer, dimension(:), intent(in) :: nAdvCellsForEdge !< Input: Number of advection cells for each edge
      integer, dimension(:,:), intent(in) :: advCellsForEdge !< Input: List of advection cells for each edge
      real (kind=RKIND), dimension(:,:), intent(in) :: normalThicknessFlux !< Input: Thichness weighted velocitiy
      real (kind=RKIND), dimension(:,:), intent(in) :: w !< Input: Vertical velocity
      real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness !< Input:Thickness
      real (kind=RKIND), dimension(:,:), intent(in) :: verticalCellSize !< Input: Distance between vertical interfaces of a cell
      real (kind=RKIND), intent(in) :: dt !< Input: Timestep
      type (mpas_pool_type), intent(in) :: meshPool !< Input: Mesh information
      real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend !< Input/Output: Tracer tendency
      integer, dimension(:), pointer :: maxLevelCell !< Input: Index to max level at cell center
      integer, dimension(:), pointer :: maxLevelEdgeTop !< Input: Index to max level at edge with non-land cells on both sides
      integer, dimension(:,:), pointer :: highOrderAdvectionMask !< Input: Mask for high order advection
      integer, dimension(:, :), pointer :: edgeSignOnCell !< Input: Sign for flux from edge on each cell.

      integer :: i, iCell, iEdge, k, iTracer, cell1, cell2
      integer :: nVertLevels, num_tracers
      integer, pointer :: nCells, nEdges, nCellsSolve, maxEdges
      integer, dimension(:), pointer :: nEdgesOnCell
      integer, dimension(:,:), pointer :: cellsOnEdge, cellsOnCell, edgesOnCell

      real (kind=RKIND) :: tracer_weight, invAreaCell1
      real (kind=RKIND) :: verticalWeightK, verticalWeightKm1
      real (kind=RKIND), dimension(:), pointer :: dvEdge, areaCell, verticalDivergenceFactor
      real (kind=RKIND), dimension(:,:), allocatable :: tracer_cur, high_order_vert_flux

      real (kind=RKIND), parameter :: eps = 1.e-10_RKIND

      ! Get dimensions
      call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
      call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
      call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
      call mpas_pool_get_dimension(meshPool, 'maxEdges', maxEdges)
      nVertLevels = size(tracers,dim=2)
      num_tracers = size(tracers,dim=1)

      ! Initialize pointers
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)

      allocate(verticalDivergenceFactor(nVertLevels))
      verticalDivergenceFactor = 1.0_RKIND
 
      allocate(tracer_cur(nVertLevels,nCells+1))
      allocate(high_order_vert_flux(nVertLevels+1,nCells))

      call mpas_threading_barrier()

      !print*, 'vertOrder ', vertOrder

      ! Loop over tracers. One tracer is advected at a time. It is copied into a
      ! temporary array in order to improve locality
      do iTracer = 1, num_tracers
        ! Initialize variables for use in this iTracer iteration
        !$omp do schedule(runtime)
        do iCell = 1, nCells
           tracer_cur(:, iCell) = tracers(iTracer, :, iCell)

           high_order_vert_flux(:, iCell) = 0.0_RKIND
        end do
        !$omp end do

        !  Compute the high order vertical flux. Also determine bounds on
        !  tracer_cur.
        !$omp do schedule(runtime) private(k, verticalWeightK,
        !verticalWeightKm1)
        do iCell = 1, nCells
          k = max(1, min(maxLevelCell(iCell), 2))
          verticalWeightK = verticalCellSize(k-1, iCell) / (verticalCellSize(k, iCell) + verticalCellSize(k-1, iCell))
          verticalWeightKm1 = verticalCellSize(k, iCell) / (verticalCellSize(k, iCell) + verticalCellSize(k-1, iCell))
          high_order_vert_flux(k,iCell) = w(k,iCell)*(verticalWeightK*tracer_cur(k,iCell)+verticalWeightKm1*tracer_cur(k-1,iCell))

          do k=3,maxLevelCell(iCell)-1
             select case (vertOrder)
             case (vertOrder4)
               high_order_vert_flux(k, iCell) = mpas_tracer_advection_vflux4(tracer_cur(k-2,iCell),tracer_cur(k-1,iCell),  &
                                      tracer_cur(k,iCell),tracer_cur(k+1,iCell), w(k,iCell))
             case (vertOrder3)
               high_order_vert_flux(k, iCell) = mpas_tracer_advection_vflux3(tracer_cur(k-2,iCell),tracer_cur(k-1,iCell),  &
                                      tracer_cur(k,iCell),tracer_cur(k+1,iCell), w(k,iCell), coef3rdOrder )
             case (vertOrder2)
               verticalWeightK = verticalCellSize(k-1, iCell) / (verticalCellSize(k, iCell) + verticalCellSize(k-1, iCell))
               verticalWeightKm1 = verticalCellSize(k, iCell) / (verticalCellSize(k, iCell) + verticalCellSize(k-1, iCell))
               high_order_vert_flux(k,iCell) = w(k, iCell) * (verticalWeightK * tracer_cur(k, iCell) &
                                             + verticalWeightKm1 * tracer_cur(k-1, iCell))
             end select ! vertOrder
          end do

          k = max(1, maxLevelCell(iCell))
          verticalWeightK = verticalCellSize(k-1, iCell) / (verticalCellSize(k,iCell) + verticalCellSize(k-1, iCell))
          verticalWeightKm1 = verticalCellSize(k, iCell) / (verticalCellSize(k,iCell) + verticalCellSize(k-1, iCell))
          high_order_vert_flux(k,iCell) = w(k,iCell)*(verticalWeightK*tracer_cur(k,iCell)+verticalWeightKm1*tracer_cur(k-1,iCell))
        end do ! iCell Loop
        !$omp end do

        ! Accumulate the scaled high order vertical tendencies.
        !$omp do schedule(runtime) private(k)
        do iCell = 1, nCellsSolve
          do k = 1,maxLevelCell(iCell)
            tend(iTracer, k, iCell) = tend(iTracer, k, iCell) + verticalDivergenceFactor(k) * (high_order_vert_flux(k+1, iCell) &
                                    - high_order_vert_flux(k, iCell))
          end do ! k loop
        end do ! iCell loop
        !$omp end doi

      end do ! iTracer loop

      call mpas_threading_barrier()

      deallocate(tracer_cur, high_order_vert_flux)
      deallocate(verticalDivergenceFactor)

   end subroutine ocn_tracer_vert_advection_std!}}}

   subroutine ocn_tracer_hdiff_del2_tend(meshPool, layerThicknessEdge, tracers, tend, err)!{{{
      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: meshPool !< Input: Mesh information

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThicknessEdge !< Input: thickness at edges

      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
        tracers !< Input: tracer quantities

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
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

      integer :: iCell, iEdge, cell1, cell2
      integer :: i, k, iTracer, num_tracers, nCells
      integer, pointer :: nVertLevels
      integer, dimension(:), pointer :: nCellsArray

      integer, dimension(:), pointer :: maxLevelEdgeTop, nEdgesOnCell
      integer, dimension(:,:), pointer :: cellsOnEdge, edgesOnCell, edgeSignOnCell

      real (kind=RKIND) :: invAreaCell
      real (kind=RKIND) :: tracer_turb_flux, flux, r_tmp

      real (kind=RKIND), dimension(:), pointer :: areaCell, dvEdge, dcEdge
      real (kind=RKIND), dimension(:), pointer :: meshScalingDel2

      err = 0

      !if (.not.del2On) return

      call mpas_timer_start("tracer del2")

      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
      num_tracers = size(tracers, dim=1)

      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
      call mpas_pool_get_array(meshPool, 'meshScalingDel2', meshScalingDel2)

      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)

      nCells = nCellsArray( 1 )

      !
      ! compute a boundary mask to enforce insulating boundary conditions in the
      ! horizontal
      !
      !$omp do schedule(runtime)
      do iCell = 1, nCells
        invAreaCell = 1.0_RKIND / areaCell(iCell)
        do i = 1, nEdgesOnCell(iCell)
          iEdge = edgesOnCell(i, iCell)
          cell1 = cellsOnEdge(1,iEdge)
          cell2 = cellsOnEdge(2,iEdge)
 
          r_tmp = meshScalingDel2(iEdge) * eddyDiff2 * dvEdge(iEdge) / dcEdge(iEdge)

          do k = 1, maxLevelEdgeTop(iEdge)
            do iTracer = 1, num_tracers
              ! \kappa_2 \nabla \phi on edge
              tracer_turb_flux = tracers(iTracer, k, cell2) - tracers(iTracer, k, cell1)

              ! div(h \kappa_2 \nabla \phi) at cell center
              flux = layerThicknessEdge(k, iEdge) * tracer_turb_flux * r_tmp

              tend(iTracer, k, iCell) = tend(iTracer, k, iCell) - edgeSignOnCell(i, iCell) * flux * invAreaCell
            end do
          end do

        end do
      end do
      !$omp end do

      call mpas_timer_stop("tracer del2")

   !--------------------------------------------------------------------

   end subroutine ocn_tracer_hdiff_del2_tend!}}}

   subroutine ocn_tracer_vert_diff_tend(meshPool, dt, vertDiffTopOfCell, layerThickness, tracers, tend, &
                  vertNonLocalFlux, tracerGroupSurfaceFlux, config_cvmix_kpp_nonlocal_with_implicit_mix, &
                  err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         meshPool          !< Input: mesh information

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         vertDiffTopOfCell !< Input: vertical mixing coefficients

      real (kind=RKIND), intent(in) :: &
         dt            !< Input: time step

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickness, &             !< Input: thickness at cell center
         tracerGroupSurfaceFlux        !< Input: surface flux for tracers nonlocal computation

      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
         vertNonLocalFlux             !non local flux at interfaces

      logical, intent(in) :: config_cvmix_kpp_nonlocal_with_implicit_mix
      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
         tracers        !< Input: tracers

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
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

      integer :: iCell, k, num_tracers, N, nCells, iTracer
      integer, pointer :: nVertLevels
      integer, dimension(:), pointer :: nCellsArray

      integer, dimension(:), pointer :: maxLevelCell

      real (kind=RKIND), dimension(:,:), allocatable :: tracer_cur 

      err = 0

      !if(.not.tracerVmixOn) return
                  
      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray) 
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      num_tracers = size(tracers, dim=1)

      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)

      nCells = nCellsArray( 1 )

      allocate(tracer_cur(nVertLevels,nCells+1))

      call mpas_timer_start('vmix tracers tend imp loop', .false.)

      do iTracer = 1, num_tracers
        ! Initialize variables for use in this iTracer iteration
        !$omp do schedule(runtime)
         do iCell = 1, nCells
            tracer_cur(:, iCell) = tracers(iTracer, :, iCell)
         end do
         do iCell = 1, nCells
            N = maxLevelCell(iCell) 
            do k = 2, N
               tend(iTracer, k, iCell) = tend(iTracer, k, iCell) + 2.0_RKIND*vertDiffTopOfCell(k,iCell)*(tracer_cur(k-1, iCell)-tracer_cur(k,iCell)) / & 
                                         (layerThickness(k-1,iCell) + layerThickness(k,iCell))
            end do
            do k = 1, N-1 
               tend(iTracer, k, iCell) = tend(iTracer, k, iCell) - 2.0_RKIND*vertDiffTopOfCell(k+1,iCell)*(tracer_cur(k, iCell)-tracer_cur(k+1,iCell)) / &
                                         (layerThickness(k,iCell) + layerThickness(k+1,iCell))
            end do  
         end do 
      end do

      call mpas_timer_stop('vmix tracers tend imp loop')

      deallocate(tracer_cur)

   end subroutine ocn_tracer_vert_diff_tend!}}}

   subroutine jacobianVert(meshPool, ETDPool, vertDiffTopOfCell, layerThickness, w, iCell) 

      type (mpas_pool_type), intent(in) ::  meshPool                     !< Input: mesh information
      type (mpas_pool_type), intent(in) ::  ETDPool
      real (kind=RKIND), dimension(:,:), intent(in) :: vertDiffTopOfCell !< Input: vertical mixing coefficients
      real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness    !< Input: thickness at cell center
      real (kind=RKIND), dimension(:,:), intent(in) :: w                 !< Input: Vertical velocity 
      integer, intent(in) :: iCell

      real (kind=RKIND), dimension(:), pointer :: JacZ 
      integer, dimension(:), pointer :: maxLevelCell !< Input: Index to max level at cell center
      real (kind=RKIND), dimension(:,:), allocatable :: JacVertAdv, JacVertDiff, JacVert
      integer :: i, j, k, N
      
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(ETDPool, 'JacZ', JacZ)

      N = maxLevelCell(iCell)
      allocate(JacVertAdv(N,N))
      allocate(JacVertDiff(N,N))
      allocate(JacVert(N,N))

      JacVertAdv(:,:) = 0.0_RKIND
      JacVertDiff(:,:) = 0.0_RKIND

      JacVertAdv(1,1) = w(2,iCell) * layerThickness(2, iCell) / (layerThickness(2, iCell) + layerThickness(1, iCell)) 
      JacVertAdv(1,2) = w(2,iCell) * layerThickness(1, iCell) / (layerThickness(2, iCell) + layerThickness(1, iCell))

      do k = 2, N-1
         JacVertAdv(k,k-1) = - w(k,iCell) * layerThickness(k, iCell) / (layerThickness(k, iCell) + layerThickness(k-1, iCell))
         JacVertAdv(k,k) = w(k+1,iCell) * layerThickness(k+1, iCell) / (layerThickness(k, iCell) + layerThickness(k+1, iCell)) - & 
                           w(k,iCell) * layerThickness(k-1, iCell) / (layerThickness(k, iCell) + layerThickness(k-1, iCell))
         JacVertAdv(k,k+1) = w(k+1,iCell) * layerThickness(k, iCell) / (layerThickness(k, iCell) + layerThickness(k+1, iCell)) 
      end do 

      JacVertAdv(N,N-1) = - w(N,iCell) * layerThickness(N, iCell) / (layerThickness(N, iCell) + layerThickness(N-1, iCell))
      JacVertAdv(N,N) = - w(N,iCell) * layerThickness(N-1, iCell) / (layerThickness(N, iCell) + layerThickness(N-1, iCell))

      !JacVertDiff(1,1) = - 2.0_RKIND*vertDiffTopOfCell(2,iCell) / (layerThickness(2, iCell) + layerThickness(1, iCell))
      !JacVertDiff(1,2) =   2.0_RKIND*vertDiffTopOfCell(2,iCell) / (layerThickness(2, iCell) + layerThickness(1, iCell))

      !do k = 2, N-1
      !   JacVertDiff(k,k-1) = 2.0_RKIND*vertDiffTopOfCell(k,iCell) / (layerThickness(k, iCell) + layerThickness(k-1, iCell))
      !   JacVertDiff(k,k) = - 2.0_RKIND*vertDiffTopOfCell(k,iCell) / (layerThickness(k, iCell) + layerThickness(k-1, iCell)) & 
      !                      - 2.0_RKIND*vertDiffTopOfCell(k+1,iCell) / (layerThickness(k, iCell) + layerThickness(k+1, iCell))  
      !   JacVertDiff(k,k+1) = 2.0_RKIND*vertDiffTopOfCell(k+1,iCell) / (layerThickness(k, iCell) + layerThickness(k+1, iCell))
      !end do

      !JacVertDiff(N,N-1) = 2.0_RKIND*vertDiffTopOfCell(N,iCell) / (layerThickness(N, iCell) + layerThickness(N-1, iCell))
      !JacVertDiff(N,N) = - 2.0_RKIND*vertDiffTopOfCell(N,iCell) / (layerThickness(N, iCell) + layerThickness(N-1, iCell))

      do i = 1, N
         do j = 1, N
            JacVert(i,j) = JacVertAdv(i,j) + JacVertDiff(i,j)
         end do
      end do 

      !Assemble the jacobian by colum using JacVert
      !
      !do j = 1, N
      !   JacZ((j-1)*N+1 : j*N) = JacVert(:,j) 
      !end do
      do i = 1, N
         do j = 1, N
            JacZ((j-1)*N+i) = JacVert(i,j)
         end do
      end do

      deallocate(JacVertAdv, JacVertDiff, JacVert)

   end subroutine jacobianVert 

   subroutine ocn_phi_function(ETDPool, CFL_pow, phi1A, NLayers, dt)!{{{

   !***********************************************************************
   !
   !  routine ocn_phi_function
   !
   !> \brief   Computes phi1(A) for the ETD time-stepping scheme
   !> \author  Sara Calandrini
   !> \date    July 2020
   !> \details
   !>  This routine computes the phi1 function of the linear operator for
   !>  the tracers based on the scaling and squaring algorithm.
   !
   !-----------------------------------------------------------------------

      type (mpas_pool_type), intent(in) ::  ETDPool
      integer, intent(in) :: CFL_pow, NLayers
      real (kind=RKIND), intent(inout), dimension(:,:) :: phi1A
      real (kind=RKIND), intent(in) :: dt     

      real (kind=RKIND), dimension(:), pointer :: Jac
      real (kind=RKIND), dimension(NLayers,NLayers) :: A
      real (kind=RKIND), dimension(NLayers,NLayers) :: Temp
      real (kind=RKIND), dimension(NLayers,NLayers) :: Temp2
      real (kind=RKIND), dimension(NLayers,NLayers) :: Id

      !allocate(A(4,4), Temp(4,4), Temp2(4,4), Id(4,4))

      real (kind=RKIND) :: power = 0.5_RKIND
      integer :: k1, k2

      call mpas_pool_get_array(ETDPool, 'JacZ', Jac)

      phi1A(:,:)=0.0_RKIND
      Temp(:,:)=0.0_RKIND
      Temp2(:,:)=0.0_RKIND
      Id(:,:)=0.0_RKIND

      !print*, 'CFL_pow', CFL_pow

      do k1 = 1, NLayers
         phi1A(k1,k1)=1
         Id(k1,k1)=1
         do k2 = 1, NLayers
           A(k1,k2) = (dt/(2.0_RKIND**CFL_pow))*Jac((k2-1)*NLayers+k1)
         end do
      end do

      !to remove
      !do k1=1, NLayers
           !print*, A(k1,1),A(k1,2),A(k1,3),A(k1,4)
      !end do
      !print*, '-------------------------'
      !to remove

      CALL DGEMM('N','N',NLayers,NLayers,NLayers,1.0_RKIND,A,Nlayers,A,NLayers,0.0_RKIND,Temp,Nlayers) !A^2
      CALL DGEMM('N','N',NLayers,NLayers,NLayers,1.0_RKIND,A,Nlayers,Temp,NLayers,0.0_RKIND,Temp2,Nlayers) !A^3
      CALL DGEMM('N','N',NLayers,NLayers,NLayers,0.5_RKIND,A,Nlayers,Id,NLayers,1.0_RKIND,phi1A,Nlayers) !I+1/2A
      CALL DGEMM('N','N',NLayers,NLayers,NLayers,1./6._RKIND,Temp,Nlayers,Id,NLayers,1.0_RKIND,phi1A,Nlayers) !I+1/2A+1/6A^2
      CALL DGEMM('N','N',NLayers,NLayers,NLayers,1./24._RKIND,Temp2,Nlayers,Id,NLayers,1.0_RKIND,phi1A,Nlayers) !I+1/2A+1/6A^2+1/24A^3

      do k1 = 1, CFL_pow
         Temp(:,:)=0.0_RKIND
         Temp2(:,:)=0.0_RKIND
         CALL DGEMM('N','N',NLayers,NLayers,NLayers,1.0_RKIND,phi1A,Nlayers,phi1A,NLayers,0.0_RKIND,Temp,Nlayers) !phi1A^2
         CALL DGEMM('N','N',NLayers,NLayers,NLayers,1.0_RKIND,A,Nlayers,Temp,NLayers,0.0_RKIND,Temp2,Nlayers) !A*phi1A^2
         if (k1==1) then
            power = 0.5_RKIND
         elseif (k1==2) then
            power = 1.0_RKIND
         elseif (k1>2) then
           power = 2.0_RKIND**(k1-2)
         end if
         !to remove
         !print*,power
         !print*, '---------do-power------------'
         !to remove
         CALL DGEMM('N','N',NLayers,NLayers,NLayers,power,Temp2,Nlayers,Id,NLayers,1.0_RKIND,phi1A,Nlayers) !phi1A + 2^(k1-1)*A*phi1A^2
         !to remove
         !do k2=1, NLayers
               !print*,phi1A(k2,1),phi1A(k2,2),phi1A(k2,3),phi1A(k2,4)
         !end do
         !print*, '----------do-phi1A-----------'
         !to remove 
      end do
     
   end subroutine ocn_phi_function!}}

end module ocn_time_integration_ETD

module mpas_ocn_call_python
   implicit none
   private
   save

   public :: call_python

   contains

   subroutine call_python()
 
     use, intrinsic :: iso_c_binding

       interface
          subroutine remap() bind (c)
          end subroutine remap
       end interface

   end subroutine call_python
  
end module mpas_ocn_call_python 

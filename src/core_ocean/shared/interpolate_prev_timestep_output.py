""" SCRIPT FOR CONVERTING THE MPAS GRID TO OBSERVED GRID LIVE
Python-Fortran code interface credit to Noah Brenowitz, see https://www.noahbrenowitz.com/post/calling-fortran-from-python/
Remapper code from make_mpas_to_lat_lon_mapping.py (Xylar Asay-Davis)

This code is designed to ensure that the accuracy of the difference of the interpolation of the simulation and the interpolated observed data.  If this step is NOT utilized, we are simply running a nudging data assimilation scheme.
"""


import cffi
ffibuilder = cffi.FFI()

header= """
extern void remap(void);
"""
module = """
from my_plugin import ffi
import xarray
from pyremap import MpasMeshDescriptor, Remapper, get_lat_lon_descriptor

@ffi.def_extern()
def remap():
   #replace with the MPAS mesh name
   inGridName = 'SOMA_32'
   inGridFileName = '/lustre/scratch4/turquoise/ecarlson10/mpas_ocean_runs/mpas_o_soma_tests/ocean_32_km/soma/32km/default/forward/output/output_data_interp'
   inDescriptor = MpasMeshDescriptor(inGridFileName, inGridName)
   # modify the resolution of the global lat-lon grid as desired
   outDescriptor = get_lat_lon_descriptor(dLon=0.1, dLat=0.1, lonMin = -16.5, lonMax = 16.5, latMin = 21.5, latMax = 48.5)
   outGridName = outDescriptor.meshName
   
   mappingFileName = 'map_{}_to_{}.nc'.format(inGridName, outGridName)
   
   remapper = Remapper(inDescriptor, outDescriptor, mappingFileName)
   
   remapper.build_mapping_file(method='bilinear')
   outFileName = 'output_AOT_{}.nc'.format(outGridName)
   ds = xarray.open_dataset(inGridFileName)
   dsOut = xarray.Dataset()

   dsOut['velocityZonal_interp'] = ds['velocityZonal']
   dsOut['velocityMeridional_interp'] = ds['velocityMeridional']
   dsOut['temperature_interp'] = ds['temperature']
   dsOut['salinity_interp'] = ds['salinity']

   dsOut = remapper.remap(dsOut)
   dsOut.to_netcdf(outFileName)
   print ('I worked!!!!')
"""

with open("plugin.h","w") as f:
     f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin",r'''
     #include "plugin.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libplugin.*",verbose=True)

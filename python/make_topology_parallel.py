# This script is used to generate cpp files that instantiate the
# make_topology_parallel function with template arguments passed 
# from CMake to reduce compilation times with GPU compilers
# Nicolas Bertin

import sys

args = sys.argv[1:]

if len(args) > 1:
    # generate global header file
    filename = 'topology_parallel_types.h'
    with open(filename, 'w') as f:
        for arg in args:
            f.write('extern template Topology* make_topology_parallel<ForceType::%s>(System* system, Force* force, Mobility* mobility, TParams& topolparams);\n' % arg)
        
else:
    # generate individual cpp files
    arg = args[0]
    filename = 'topology_%s.cpp' % arg
    with open(filename, 'w') as f:
        f.write('#include "exadis_pybind.h"\n')
        f.write('template Topology* make_topology_parallel<ForceType::%s>(System* system, Force* force, Mobility* mobility, TParams& topolparams);\n' % arg)

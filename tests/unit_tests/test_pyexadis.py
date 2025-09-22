import os, sys, io
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
except ImportError:
    raise ImportError('Cannot import pyexadis')

def test_init():
    pyexadis.initialize(verbose=False)
    pyexadis.finalize()
    print('pass')
    
if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name == "test_import":
        print('pass')
    elif name == "test_init":
        test_init()
    else:
        raise ValueError(f"Invalid test name = '{name}'")

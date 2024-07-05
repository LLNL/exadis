import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetwork, VisualizeNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


def test_nodal_xslip():
    """test_nodal_xslip:
    Example of nodal x-slip mechanisms in an elemental network.
    See Bertin et al., Acta Materialia 271, 119884
    """
    
    pyexadis.initialize()
    
    G = ExaDisNet()
    G.read_paradis('ta-elemental-single.data')
    N = DisNetManager(G)

    vis = VisualizeNetwork()
    
    state = {
        "crystal": 'bcc',
        "burgmag": 2.85e-10,
        "mu": 55.0e9,
        "nu": 0.339,
        "a": 1.0,
        "maxseg": 10.0,
        "minseg": 2.0,
        "rtol": 0.3,
        "nextdt": 1e-13,
    }
    
    calforce  = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ec_junc_fact=0.3, Ngrid=32, cell=N.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=2600.0, Mscrew=20.0, Mclimb=1e-4, vmax=3400.0)
    timeint   = TimeIntegration(integrator='Trapezoid', state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Proximity', state=state)
    topology  = Topology(topology_mode='TopologySerial', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint, 
                          collision=collision, topology=topology, remesh=remesh, vis=vis,
                          state=state, max_step=500, loading_mode='strain_rate', erate=-2e8,
                          print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                          write_freq=10, write_dir='output')
    sim.run(N, state)
    
    pyexadis.finalize()


if __name__ == "__main__":
    test_nodal_xslip()

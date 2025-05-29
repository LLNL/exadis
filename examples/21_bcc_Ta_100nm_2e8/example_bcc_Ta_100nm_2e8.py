import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


def example_bcc_Ta_100nm_2e8():
    """example_bcc_Ta_100nm_2e8:
    Example of a 100nm MD-like simulation of bcc Ta loaded
    at a strain rate of 2e8/s.
    E.g. see Bertin et al., Acta Materialia 271, 119884
    """
    
    pyexadis.initialize()
    
    state = {
        "crystal": 'bcc',
        "burgmag": 2.85e-10,
        "mu": 55.0e9,
        "nu": 0.339,
        "a": 1.0,
        "maxseg": 15.0,
        "minseg": 3.0,
        "rtol": 0.3,
        "rann": 0.6,
        "nextdt": 5e-13,
    }
    
    Lbox = 300.0
    G = ExaDisNet()
    G.generate_prismatic_config(state["crystal"], Lbox, 12, 0.21*Lbox, state["maxseg"], uniform=True)
    net = DisNetManager(G)
    
    vis = None
    
    calforce  = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=2600.0, Mscrew=20.0, Mclimb=1e-4, vmax=3400.0)
    timeint   = TimeIntegration(integrator='Trapezoid', multi=10, state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint, 
                              collision=collision, topology=topology, remesh=remesh, vis=vis,
                              loading_mode='strain_rate', erate=2e8, edir=np.array([0.,0.,1.]),
                              max_step=10000, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=2, plot_pause_seconds=0.0001,
                              write_freq=100, write_dir='output_bcc_Ta_100nm_2e8')
    sim.run(net, state)
    
    pyexadis.finalize()


if __name__ == "__main__":
    example_bcc_Ta_100nm_2e8()

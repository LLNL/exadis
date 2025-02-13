import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, VisualizeNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


def test_oprec():
    """test_oprec:
    Example of a simulation that stores OpRec files and
    of a replay of the simulation from the OpRec files
    """
    
    # Use argument "0" or no argument to run the simulation 
    # Use argument "1" to relay the simulation from the OpRec files
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    if arg == "0":
        # Run simulation
        oprec_replay = False
        output_dir = 'output_oprec'
        
    elif arg == "1":
        # Replay the simulation from OpRec files
        oprec_replay = True
        output_dir = 'output_oprec_replay'
        oprec_files = 'output_oprec/oprec.*.exadis'
        
    else:
        raise ValueError("argument value must be 0 or 1")
    
    
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
    G.generate_prismatic_config(state["crystal"], Lbox, 4, 0.21*Lbox, state["maxseg"])
    N = DisNetManager(G)
    
    #vis = None
    vis = VisualizeNetwork()
    
    calforce  = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=32, cell=N.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=2600.0, Mscrew=20.0, Mclimb=1e-4, vmax=3400.0)
    timeint   = TimeIntegration(integrator='Trapezoid', multi=10, state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint, 
                              collision=collision, topology=topology, remesh=remesh, vis=vis,
                              loading_mode='strain_rate', erate=2e7, edir=np.array([0.,0.,1.]),
                              max_step=200, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                              write_freq=10, write_dir=output_dir,
                              oprecwritefreq=20, oprecfilefreq=100, oprecposfreq=10)
    if oprec_replay:
        sim.replay(N, state, oprec_files)
    else:
        sim.run(N, state)
    
    pyexadis.finalize()


if __name__ == "__main__":
    test_oprec()

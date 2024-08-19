import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, NodeConstraints, SimulateNetwork, VisualizeNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
    from pyexadis_utils import insert_frank_read_src
except ImportError:
    raise ImportError('Cannot import pyexadis')


def test_collision():
    
    pyexadis.initialize()
    
    Lbox = 300.0
    cell = pyexadis.Cell(h=Lbox*np.eye(3), is_periodic=[1,1,1])
    nodes, segs = [], []
    
    burg = np.array([1.0, 0.0, 0.0])
    plane = np.array([0.0, 1.0, 0.0])
    thetadeg = -90.0
    length = 0.5*Lbox
    center = 0.5*Lbox * np.ones(3)
    nodes, segs = insert_frank_read_src(cell, nodes, segs, burg, plane, length, center, theta=thetadeg)
    
    burg = np.array([2.0, 0.0, 0.0]) # junction
    plane = np.array([0.0, 0.0, 1.0])
    center += np.array([0.2*Lbox, 0.0, 0.0])
    nodes, segs = insert_frank_read_src(cell, nodes, segs, burg, plane, length, center, theta=thetadeg)
    
    N = DisNetManager(ExaDisNet(cell, nodes, segs))

    vis = VisualizeNetwork()
    
    state = {
        "burgmag": 2.55e-10,
        "mu": 54.6e9,
        "nu": 0.324,
        "a": 6.0,
        "maxseg": 0.10*Lbox,
        "minseg": 0.02*Lbox,
        "rtol": 1.0,
        "nextdt": 5e-13
    }
    
    calforce  = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=32, cell=N.cell)
    mobility  = MobilityLaw(mobility_law='SimpleGlide', state=state, mob=1000.0)
    timeint   = TimeIntegration(integrator='Trapezoid', state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Proximity', state=state)
    topology  = None
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint, 
                          collision=collision, topology=topology, remesh=remesh, vis=vis,
                          state=state, max_step=200, loading_mode='stress',
                          applied_stress=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5e8]),
                          print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                          write_freq=10, write_dir='output')
    sim.run(N, state)
    
    pyexadis.finalize()


if __name__ == "__main__":
    test_collision()

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


def init_bcc_junction(disloc_length, phi1, phi2):
    Lbox = 1.0*disloc_length
    cell = pyexadis.Cell(h=Lbox*np.eye(3), is_periodic=[1,1,1])
    nodes, segs = [], []
    
    b1 = 1.0/np.sqrt(3.0)*np.array([-1.0, 1.0, 1.0])
    p1 = np.array([1.0, -1.0, 0.0])
    b2 = 1.0/np.sqrt(3.0)*np.array([1.0, -1.0, 1.0])
    p2 = np.array([0.0, 1.0, 1.0])
    
    linter = np.cross(p1, p2)
    linter = linter / np.linalg.norm(linter)
    y1 = np.cross(linter, p1)
    y1 = y1 / np.linalg.norm(y1)
    ldir1 = np.cos(phi1*np.pi/180.0)*linter+np.sin(phi1*np.pi/180.0)*y1
    y2 = np.cross(linter, p2)
    y2 = y2 / np.linalg.norm(y2)
    ldir2 = np.cos(phi2*np.pi/180.0)*linter+np.sin(phi2*np.pi/180.0)*y2
    
    center = 0.5*Lbox * np.ones(3)
    delta = 1.0 * np.array([1.0, 1.0, 0.0]);
    
    nodes, segs = insert_frank_read_src(cell, nodes, segs, b1, p1, disloc_length, center+delta, linedir=ldir1)
    nodes, segs = insert_frank_read_src(cell, nodes, segs, b2, p2, disloc_length, center-delta, linedir=ldir2)
    
    N = DisNetManager(ExaDisNet(cell, nodes, segs))
    return N


def test_bcc_junction():
    
    pyexadis.initialize()
    
    disloc_length = 200.0 # length of dislocation lines
    phi1 = 20.0 # angle (deg) of the first dislocation with intersection direction
    phi2 = 20.0 # angle (deg) of the second dislocation with intersection direction
    N = init_bcc_junction(disloc_length, phi1, phi2)

    vis = VisualizeNetwork()
    
    state = {
        "crystal": 'bcc',
        "burgmag": 2.85e-10,
        "mu": 55.0e9,
        "nu": 0.339,
        "a": 1.0,
        "maxseg": 0.10*disloc_length,
        "minseg": 0.02*disloc_length,
        "rtol": 1.0,
        "nextdt": 5e-13,
        "maxdt": 5e-13,
    }
    
    calforce  = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=16, cell=N.cell)
    mobility  = MobilityLaw(mobility_law='BCC_0B', state=state, Medge=2600.0, Mscrew=20.0, Mclimb=1e-4, vmax=3400.0)
    timeint   = TimeIntegration(integrator='Trapezoid', state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Proximity', state=state)
    topology  = Topology(topology_mode='TopologySerial', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint, 
                          collision=collision, topology=topology, remesh=remesh, vis=vis,
                          state=state, max_step=200, loading_mode='stress',
                          print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                          write_freq=10, write_dir='output')
    sim.run(N, state)
    
    pyexadis.finalize()


if __name__ == "__main__":
    test_bcc_junction()

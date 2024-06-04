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


def init_fcc_junction(junction_type, disloc_length, phi1, phi2):
    Lbox = 1.0*disloc_length
    cell = pyexadis.Cell(h=Lbox*np.eye(3), is_periodic=[1,1,1])
    nodes, segs = [], []
    
    b1 = 1.0/np.sqrt(2.0)*np.array([0.0, 1.0, 1.0])
    p1 = np.array([1.0, 1.0, -1.0])
    
    if junction_type == 'lomer':
        b2 = 1.0/np.sqrt(2.0)*np.array([1.0, 0.0,-1.0])
        p2 = np.array([1.0, 1.0, 1.0])
    elif junction_type == 'glissile':
        b2 = 1.0/np.sqrt(2.0)*np.array([1.0, 0.0,-1.0])
        p2 = np.array([1.0,-1.0, 1.0])
    elif junction_type == 'hirth':
        b2 = 1.0/np.sqrt(2.0)*np.array([0.0,-1.0, 1.0])
        p2 = np.array([1.0, 1.0, 1.0])
    elif junction_type == 'colinear':
        b2 = 1.0/np.sqrt(2.0)*np.array([0.0,-1.0,-1.0])
        p2 = np.array([1.0,-1.0, 1.0])
    else:
        raise ValueError('Unknown junction type %s'%junction_type)
    
    linter = np.cross(p1, p2)
    linter = linter / np.linalg.norm(linter)
    y1 = np.cross(linter, p1)
    y1 = y1 / np.linalg.norm(y1)
    ldir1 = np.cos(phi1*np.pi/180.0)*linter+np.sin(phi1*np.pi/180.0)*y1
    y2 = np.cross(linter, p2)
    y2 = y2 / np.linalg.norm(y2)
    ldir2 = np.cos(phi2*np.pi/180.0)*linter+np.sin(phi2*np.pi/180.0)*y2
    
    center = 0.5*Lbox * np.ones(3)
    delta = 0.01 * disloc_length * np.array([1.0, 1.0, 0.0]);
    
    nodes, segs = insert_frank_read_src(cell, nodes, segs, b1, p1, disloc_length, center+delta, linedir=ldir1)
    nodes, segs = insert_frank_read_src(cell, nodes, segs, b2, p2, disloc_length, center-delta, linedir=ldir2)
    
    N = DisNetManager(ExaDisNet(cell, nodes, segs))
    return N


def test_fcc_junctions():
    
    pyexadis.initialize()
    
    junction_type = 'lomer'
    #junction_type = 'glissile'
    #junction_type = 'hirth'
    #junction_type = 'colinear'
    
    disloc_length = 1000.0 # length of dislocation lines
    phi1 = 30.0 # angle (deg) of the first dislocation with intersection direction
    phi2 = 30.0 # angle (deg) of the second dislocation with intersection direction
    N = init_fcc_junction(junction_type, disloc_length, phi1, phi2)

    vis = VisualizeNetwork()
    
    params = {
        "crystal": 'fcc',
        "burgmag": 2.55e-10,
        "mu": 54.6e9,
        "nu": 0.324,
        "a": 6.0,
        "maxseg": 0.10*disloc_length,
        "minseg": 0.02*disloc_length,
        "rtol": 5.0,
        "nextdt": 1e-12,
        "maxdt": 1e-10,
    }
    
    calforce  = CalForce(force_mode='DDD_FFT_MODEL', params=params, Ngrid=32, cell=N.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', params=params, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint   = TimeIntegration(integrator='Trapezoid', params=params, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Proximity', params=params)
    topology  = Topology(topology_mode='TopologySerial', params=params, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', params=params)
    
    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint, 
                          collision=collision, topology=topology, remesh=remesh, vis=vis,
                          max_step=200, loading_mode='stress',
                          print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                          write_freq=10, write_dir='output')
    sim.run(N)
    
    pyexadis.finalize()


if __name__ == "__main__":
    test_fcc_junctions()

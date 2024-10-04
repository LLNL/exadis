import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, NodeConstraints, SimulateNetwork, VisualizeNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
    from pyexadis_utils import insert_frank_read_src
except ImportError:
    raise ImportError('Cannot import pyexadis')


def init_cross_slip_config(Lbox):
    cell = pyexadis.Cell(h=Lbox*np.eye(3), is_periodic=[1,1,1])
    nodes, segs = [], []
    
    theta = 0.0 # screw
    burg = 1.0/np.sqrt(2.0)*np.array([1.,1.,0.])
    plane = np.array([1.,-1.,1.])
    
    # Crystal orientation
    plane = plane / np.linalg.norm(plane)
    b = burg / np.linalg.norm(burg)
    y = np.cross(plane, b)
    y = y / np.linalg.norm(y)
    ldir = np.cos(theta*np.pi/180.0)*b+np.sin(theta*np.pi/180.0)*y
    x = np.cross(ldir, plane)
    x = x / np.linalg.norm(x)
    Rorient = np.array([x, ldir, plane])
    
    burg = np.matmul(Rorient, burg)
    plane = np.matmul(Rorient, plane)
    ldir = -np.matmul(Rorient, ldir)
    
    disloc_length = 0.7*Lbox
    center = np.array(cell.center())
    c1 = center + np.array([0.1*Lbox,0.,0.05*Lbox])
    c2 = center - np.array([0.1*Lbox,0.,0.])
    
    nodes, segs = insert_frank_read_src(cell, nodes, segs, -burg, plane, disloc_length, c1, linedir=ldir)
    for n in nodes:
        n[3] = NodeConstraints.PINNED_NODE
    nodes, segs = insert_frank_read_src(cell, nodes, segs, +burg, plane, disloc_length, c2, linedir=ldir)
    
    N = DisNetManager(ExaDisNet(cell, nodes, segs))
    return N, Rorient


def test_cross_slip():
    
    pyexadis.initialize()
    
    Lbox = 1000.0
    N, Rorient = init_cross_slip_config(Lbox)

    vis = VisualizeNetwork(view_init={"elev": 5, "azim": -110})
    
    state = {
        "crystal": 'fcc',
        "Rorient": Rorient,
        "burgmag": 2.55e-10,
        "mu": 54.6e9,
        "nu": 0.324,
        "a": 2.0,
        "maxseg": 0.05*Lbox,
        "minseg": 0.02*Lbox,
        "rtol": 2.0,
        "rann": 1.0,
        "nextdt": 1e-12,
        "maxdt": 1e-10,
    }
    
    calforce   = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=16, cell=N.cell)
    mobility   = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint    = TimeIntegration(integrator='Trapezoid', state=state, force=calforce, mobility=mobility)
    collision  = Collision(collision_mode='Proximity', state=state)
    topology   = Topology(topology_mode='TopologySerial', state=state, force=calforce, mobility=mobility)
    remesh     = Remesh(remesh_rule='LengthBased', state=state)
    cross_slip = CrossSlip(cross_slip_mode='ForceBasedSerial', state=state, force=calforce)
    
    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint, collision=collision,
                          topology=topology, remesh=remesh, cross_slip=cross_slip, vis=vis,
                          state=state, max_step=1000, loading_mode='stress',
                          applied_stress=np.array([0.0, 0.0, 0.0, 1e8, 0.0, 0.0]),
                          print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                          write_freq=10, write_dir='output')
    sim.run(N, state)
    
    pyexadis.finalize()


if __name__ == "__main__":
    test_cross_slip()

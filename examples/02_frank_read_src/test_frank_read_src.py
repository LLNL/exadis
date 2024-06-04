import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, NodeConstraints, DisNetManager, SimulateNetwork, VisualizeNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


def init_frank_read_src_loop(box_length, arm_length, burg_vec=np.array([1.0,0.0,0.0]), pbc=False):
    """
    Example of a function to generate an initial
    Frank-Read source configuration
    """
    print("init_frank_read_src_loop: length = %f" % (arm_length))
    cell = pyexadis.Cell(h=box_length*np.eye(3), is_periodic=[pbc,pbc,pbc])
    center = np.array(cell.center())
    
    rn    = np.array([[0.0, -arm_length/2.0, 0.0,         NodeConstraints.PINNED_NODE],
                      [0.0,  0.0,            0.0,         NodeConstraints.UNCONSTRAINED],
                      [0.0,  arm_length/2.0, 0.0,         NodeConstraints.PINNED_NODE],
                      [0.0,  arm_length/2.0, -arm_length, NodeConstraints.PINNED_NODE],
                      [0.0, -arm_length/2.0, -arm_length, NodeConstraints.PINNED_NODE]])
    rn[:,0:3] += center
    
    N = rn.shape[0]
    links = np.zeros((N, 8))
    for i in range(N):
        pn = np.cross(burg_vec, rn[(i+1)%N,:3]-rn[i,:3])
        pn = pn / np.linalg.norm(pn)
        links[i,:] = np.concatenate(([i, (i+1)%N], burg_vec, pn))

    N = DisNetManager(ExaDisNet(cell, rn, links))
    return N
    

def test_force():
    """
    Example of a function to compute nodal forces using
    the pyexadis binding to ExaDiS
    """
    pyexadis.initialize()
    
    Lbox = 1000.0
    N = init_frank_read_src_loop(box_length=Lbox, arm_length=0.125*Lbox, pbc=False)
    
    mu, nu, a = 50e9, 0.3, 1.0
    Ec = mu / 4.0 / np.pi * np.log(a/0.1)
    applied_stress = np.array([0.0, 0.0, 0.0, 0.0, -4.0e8, 0.0])
    
    # exadis
    params = {"burgmag": 3e-10, "mu": mu, "nu": nu, "a": a, "maxseg": 0.04*Lbox, "minseg": 0.01*Lbox}
    calforce = CalForce(params=params, Ec=Ec, force_mode='LineTension')
    force_dict = calforce.NodeForce(N, applied_stress=applied_stress)
    f_pyexadis = force_dict["nodeforces"]
    print('f_pyexadis',f_pyexadis)

    pyexadis.finalize()


def main():
    """
    Example of a script to perform a simple Frank-Read source
    simulation using the pyexadis binding to ExaDiS
    """
    pyexadis.initialize()
    
    Lbox = 1000.0
    N = init_frank_read_src_loop(box_length=Lbox, arm_length=0.125*Lbox, pbc=False)

    vis = VisualizeNetwork()
    
    params = {"burgmag": 3e-10, "mu": 50e9, "nu": 0.3, "a": 1.0, "maxseg": 0.04*Lbox, "minseg": 0.01*Lbox, "rann": 2.0}
    
    calforce  = CalForce(force_mode='LineTension', params=params)
    mobility  = MobilityLaw(mobility_law='SimpleGlide', params=params)
    timeint   = TimeIntegration(integrator='EulerForward', dt=1.0e-8, params=params)
    collision = Collision(collision_mode='Retroactive', params=params)
    topology  = None
    remesh    = Remesh(remesh_rule='LengthBased', params=params)
    
    sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint, 
                          collision=collision, topology=topology, remesh=remesh, vis=vis,
                          max_step=200, loading_mode='stress',
                          applied_stress=np.array([0.0, 0.0, 0.0, 0.0, -4.0e8, 0.0]),
                          print_freq=10, plot_freq=10, plot_pause_seconds=0.0001,
                          write_freq=10, write_dir='output')
    sim.run(N)
    
    pyexadis.finalize()


if __name__ == "__main__":
    #test_force()
    main()

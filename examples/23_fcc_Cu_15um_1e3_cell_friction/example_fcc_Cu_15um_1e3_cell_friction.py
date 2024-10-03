import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError:
    raise ImportError('Cannot import pyexadis')


def plot_friction_field():
    """plot_friction_field:
    Plot the 2D friction field to be used in the simulation
    Here the field contains values ranging from 0 to 1. The 
    friction stress acting on a dislocation segment will be
    scaled by the value of the field at the segment position.
    """
    # Load the file
    field_file = 'friction_field.dat'
    Nfield = np.loadtxt(field_file, max_rows=3).astype(int)
    field = np.loadtxt(field_file, skiprows=3).reshape(Nfield[:2])
    # Plot the field
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cntr = ax.contourf(field.T, levels=100, cmap='jet')
    cbar = fig.colorbar(cntr, ax=ax)
    ax.set_aspect('equal')
    ax.set_title(field_file)
    plt.show()


def example_fcc_Cu_15um_1e3_cell_friction():
    """example_fcc_Cu_15um_1e3_cell_friction:
    Example of a 15um simulation of fcc Cu loaded at a
    strain rate of 1e3/s using a spatially varying friction
    field to generate cellular network structures
    """
    
    pyexadis.initialize()
    
    state = {
        "crystal": 'fcc',
        "burgmag": 2.55e-10,
        "mu": 54.6e9,
        "nu": 0.324,
        "a": 6.0,
        "maxseg": 2000.0,
        "minseg": 300.0,
        "rtol": 10.0,
        "rann": 10.0,
        "nextdt": 1e-10,
        "maxdt": 1e-9,
    }
    
    output_dir = 'output_fcc_Cu_15um_1e3_cell_fric_20MPa'
    
    if 1:
        # Initial configuration
        G = ExaDisNet()
        G.read_paradis('../22_fcc_Cu_15um_1e3/180chains_16.10e.data')
        net = DisNetManager(G)
        restart = None
    else:
        # Restart configuration
        net, restart = read_restart(state=state, restart_file=os.path.join(output_dir, 'restart.1000.exadis'))
    
    vis = None
    
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell, drift=1)
    mobility  = MobilityLaw(mobility_law='FCC_0_FRIC', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0,
                            Fedge=20e6, Fscrew=20e6, friction_field='friction_field.dat')
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    
    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint, 
                              collision=collision, topology=topology, remesh=remesh, vis=vis,
                              loading_mode='strain_rate', erate=1e3, edir=np.array([0.,0.,1.]),
                              max_strain=0.01, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                              write_freq=100, write_dir=output_dir, restart=restart)
    sim.run(net, state)
    
    pyexadis.finalize()
    

if __name__ == "__main__":
    #plot_friction_field()
    example_fcc_Cu_15um_1e3_cell_friction()
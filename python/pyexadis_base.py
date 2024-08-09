"""@package docstring

ExaDiS python interface

Implements base classes for use of ExaDiS functions/data structures within python

Nicolas Bertin
bertin1@llnl.gov
"""

import os
import numpy as np
from typing import Tuple

import pyexadis
try:
    # Try importing DisNetManager and DisNet_Base from OpenDiS
    from framework.disnet_manager import DisNetManager
    from framework.disnet_base import DisNet_Base
except ImportError:
    # Use dummy DisNetManager and DisNet_Base if OpenDiS is not available
    class DisNetManager:
        def __init__(self, disnet):
            self.disnet = disnet
        def get_disnet(self, disnet_type=None):
            return self.disnet
        def export_data(self):
            return self.get_disnet().export_data()
        @property
        def cell(self):
            return self.disnet.cell
    class DisNet_Base:
        pass

from enum import IntEnum
class NodeConstraints(IntEnum):
    UNCONSTRAINED = 0
    PINNED_NODE = 7

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Cannot import matplotlib')


class ExaDisNet(DisNet_Base):
    """ExaDisNet: wrapper class for exadis dislocation network
    Implements basic functions to manipulate the network
    """
    def __init__(self, *args):
        if len(args) == 3:
            cell, nodes, segs = args[0], args[1], args[2]
            self.net = pyexadis.ExaDisNet(cell=cell, nodes=nodes, segs=segs)
        elif len(args) == 1:
            self.net = args[0] # pyexadis.ExaDisNet object
        else:
            self.net = pyexadis.ExaDisNet()
        
    def read_paradis(self, datafile):
        self.net = pyexadis.read_paradis(datafile)
        
    def write_data(self, datafile):
        self.net.write_data(datafile)
        
    def generate_prismatic_config(self, crystal, Lbox, numsources, radius, maxseg=-1, seed=1234):
        if crystal == 'BCC' or crystal == 'bcc':
            crystal = pyexadis.Crystal(pyexadis.BCC_CRYSTAL)
        elif crystal == 'FCC' or crystal == 'fcc':
            crystal = pyexadis.Crystal(pyexadis.FCC_CRYSTAL)
        self.net = pyexadis.generate_prismatic_config(crystal, Lbox, numsources, radius, maxseg, seed)
    
    def import_data(self, data):
        cell = data.get("cell")
        cell = pyexadis.Cell(h=cell.get("h"), origin=cell.get("origin"), is_periodic=cell.get("is_periodic"))
        nodes = data.get("nodes")
        nodes_array = np.hstack((nodes["tags"], nodes["positions"], nodes["constraints"]))
        segs = data.get("segs")
        segs_array = np.hstack((segs["nodeids"], segs["burgers"], segs["planes"]))
        self.net = pyexadis.ExaDisNet(cell=cell, nodes=nodes_array, segs=segs_array)
    
    def export_data(self):
        cell = self.net.get_cell()
        cell = {"h": np.array(cell.h), "origin": np.array(cell.origin), "is_periodic": cell.get_pbc()}
        nodes = self.get_nodes_data()
        segs = self.get_segs_data()
        data = {"cell": cell, "nodes": nodes, "segs": segs}
        return data
    
    @property
    def cell(self):
        return self.net.get_cell()
        
    def get_nodes_data(self):
        nodes_array = np.array(self.net.get_nodes_array())
        nodes_dict = {
            "tags": nodes_array[:,0:2].astype(int),
            "positions": nodes_array[:,2:5],
            "constraints": nodes_array[:,5:6].astype(int)
        }
        return nodes_dict
    
    def get_tags(self):
        return self.get_nodes_data()["tags"]
    
    def get_positions(self):
        return self.get_nodes_data()["positions"]
    
    def get_forces(self):
        return np.array(self.net.get_forces())
        
    def get_velocities(self):
        return np.array(self.net.get_velocities())
        
    def get_segs_data(self):
        segs_array = np.array(self.net.get_segs_array())
        segs_dict = {
            "nodeids": segs_array[:,0:2].astype(int),
            "burgers": segs_array[:,2:5],
            "planes": segs_array[:,5:8]
        }
        return segs_dict 
    

def get_exadis_params(state):
    """get_exadis_params: helper function to get exadis global state object
    """
    params = pyexadis.Params(
        burgmag=state["burgmag"],
        mu=state["mu"],
        nu=state["nu"],
        a=state["a"],
        maxseg=state["maxseg"],
        minseg=state["minseg"]
    )
    if "crystal" in state: params.set_crystal(state["crystal"])
    if "Rorient" in state: params.Rorient = state["Rorient"]
    if "rann" in state: params.rann = state["rann"]
    if "rtol" in state: params.rtol = state["rtol"]
    if "maxdt" in state: params.maxdt = state["maxdt"]
    if "nextdt" in state: params.nextdt = state["nextdt"]
    if "split3node" in state: params.split3node = state["split3node"]
    return params

def get_exadis_force(force_module, state, params):
    if not isinstance(force_module, CalForce):
        force_python = CalForcePython(force_module, state)
        force = pyexadis.make_force_python(params=params, force=force_python)
    else:
        force_python = None
        force = force_module.force
    return force, force_python

def get_exadis_mobility(mobility_module, state, params):
    if not isinstance(mobility_module, MobilityLaw):
        mobility_python = MobilityLawPython(mobility_module, state)
        mobility = pyexadis.make_mobility_python(params=params, mobility=mobility_python)
    else:
        mobility_python = None
        mobility = mobility_module.mobility
    return mobility, mobility_python
    

class CalForce:
    """CalForce: wrapper class for calculating forces on dislocation network
    """
    def __init__(self, state: dict, force_mode: str='LineTension', **kwargs) -> None:
        self.force_mode = force_mode
        self.params = get_exadis_params(state)
        self.mu = self.params.mu
        self.nu = self.params.nu
        Ec = kwargs.get('Ec', -1.0)
        Ecore_junc_fact = kwargs.get('Ec_junc_fact', 1.0)
        coreparams = pyexadis.Force_CORE_Params(Ec, Ecore_junc_fact)
        
        if self.force_mode in ['LineTension', 'LINE_TENSION_MODEL']:
            self.force = pyexadis.make_force_lt(params=self.params, coreparams=coreparams)
            
        elif self.force_mode == 'DDD_FFT_MODEL':
            Ngrid = kwargs.get('Ngrid')
            cell = kwargs.get('cell')
            if not isinstance(cell, pyexadis.Cell):
                cell = pyexadis.Cell(h=cell.h, origin=cell.origin, is_periodic=cell.is_periodic)
            self.force = pyexadis.make_force_ddd_fft(params=self.params, coreparams=coreparams, 
                                                     Ngrid=Ngrid, cell=cell)
            
        elif self.force_mode == 'SUBCYCLING_MODEL':
            Ngrid = kwargs.get('Ngrid')
            cell = kwargs.get('cell')
            if not isinstance(cell, pyexadis.Cell):
                cell = pyexadis.Cell(h=cell.h, origin=cell.origin, is_periodic=cell.is_periodic)
            drift = kwargs.get('drift', 1)
            self.force = pyexadis.make_force_subcycling(params=self.params, coreparams=coreparams,
                                                        Ngrid=Ngrid, cell=cell, drift=drift)
            
        else:
            raise ValueError('Unknown force %s' % force_mode)
            
    def NodeForce(self, N: DisNetManager, state: dict) -> dict:
        applied_stress = state["applied_stress"]
        G = N.get_disnet(ExaDisNet)
        f = pyexadis.compute_force(G.net, force=self.force, applied_stress=applied_stress)
        state["nodeforces"] = np.array(f)
        state["nodeforcetags"] = G.get_tags()
        return state
    
    def PreCompute(self, N: DisNetManager, state: dict) -> dict:
        G = N.get_disnet(ExaDisNet)
        pyexadis.pre_compute_force(G.net, force=self.force)
        return state
    
    def OneNodeForce(self, N: DisNetManager, state: dict, tag, update_state=True) -> np.array:
        applied_stress = state["applied_stress"]
        G = N.get_disnet(ExaDisNet)
        # find node index
        tags = G.get_tags()
        ind = np.where((tags[:,0]==tag[0])&(tags[:,1]==tag[1]))[0]
        if ind.size != 1:
            raise ValueError("Cannot find node tag (%d,%d) in OneNodeForce" % tuple(tag))
        # compute node force
        f = pyexadis.compute_node_force(G.net, ind[0], force=self.force, applied_stress=applied_stress)
        f = np.array(f)
        # update force dictionary if needed
        if update_state:
            if "nodeforces" in state and "nodeforcetags" in state:
                nodeforcetags = state["nodeforcetags"]
                ind = np.where((nodeforcetags[:,0]==tag[0])&(nodeforcetags[:,1]==tag[1]))[0]
                if ind.size == 1:
                    state["nodeforces"][ind[0]] = f
                else:
                    state["nodeforces"] = np.vstack((state["nodeforces"], f))
                    state["nodeforcetags"] = np.vstack((state["nodeforcetags"], tag))
            else:
                state["nodeforces"] = np.array([f])
                state["nodeforcetags"] = np.array([tag])
        return f


class CalForcePython:
    """CalForcePython: wrapper class for python-based force object
    This allows to call arbitrary force python modules from within ExaDiS
    """
    def __init__(self, force_module, state):
        self.force = force_module
        self.state = state
        
    def NodeForce(self, net):
        N = DisNetManager(ExaDisNet(net))
        self.force.NodeForce(N, self.state)
        nodeforces = self.state["nodeforces"]
        nodetags = self.state["nodeforcetags"]
        net.set_forces(nodeforces, nodetags)
        
    def PreCompute(self, net):
        N = DisNetManager(ExaDisNet(net))
        self.force.PreCompute(N, self.state)
        
    def OneNodeForce(self, net, tag):
        N = DisNetManager(ExaDisNet(net))
        return self.force.OneNodeForce(N, self.state, tag)


class MobilityLaw:
    """MobilityLaw: wrapper class for mobility laws
    """
    def __init__(self, state: dict, mobility_law: str='SimpleGlide', **kwargs) -> None:
        self.mobility_law = mobility_law
        params = get_exadis_params(state)
        
        if self.mobility_law == 'SimpleGlide':
            Medge = kwargs.get('Medge', -1.0)
            Mscrew = kwargs.get('Mscrew', -1.0)
            if Medge > 0.0 and Mscrew > 0.0:
                mobparams = pyexadis.Mobility_GLIDE_Params(Medge, Mscrew)
            else:
                mob = kwargs.get('mob', 1.0)
                mobparams = pyexadis.Mobility_GLIDE_Params(mob)
            self.mobility = pyexadis.make_mobility_glide(params=params, mobparams=mobparams)
            
        elif self.mobility_law == 'BCC_0B':
            Medge = kwargs.get('Medge')
            Mscrew = kwargs.get('Mscrew')
            Mclimb = kwargs.get('Mclimb')
            vmax = kwargs.get('vmax', -1.0)
            mobparams = pyexadis.Mobility_BCC_0B_Params(Medge, Mscrew, Mclimb, vmax)
            self.mobility = pyexadis.make_mobility_bcc_0b(params=params, mobparams=mobparams)
            
        elif self.mobility_law == 'BCC_NL':
            tempK = kwargs.get('tempK', 300.0)
            vmax = kwargs.get('vmax', -1.0)
            mobparams = pyexadis.Mobility_BCC_NL_Params(tempK, vmax)
            self.mobility = pyexadis.make_mobility_bcc_nl(params=params, mobparams=mobparams)
            
        elif self.mobility_law == 'FCC_0':
            Medge = kwargs.get('Medge')
            Mscrew = kwargs.get('Mscrew')
            vmax = kwargs.get('vmax', -1.0)
            mobparams = pyexadis.Mobility_FCC_0_Params(Medge, Mscrew, vmax)
            self.mobility = pyexadis.make_mobility_fcc_0(params=params, mobparams=mobparams)
            
        else:
            raise ValueError('Unknown mobility law %s' % mobility_law)
        
    def Mobility(self, N: DisNetManager, state: dict) -> dict:
        G = N.get_disnet(ExaDisNet)
        f = state["nodeforces"]
        nodetags = state.get("nodeforcetags", np.empty((0,2)))
        v = pyexadis.compute_mobility(G.net, mobility=self.mobility, nodeforces=f, nodetags=nodetags)
        state["nodevels"] = np.array(v)
        state["nodeveltags"] = G.get_tags()
        return state
        
    def OneNodeMobility(self, N: DisNetManager, state: dict, tag, f, update_state=True) -> np.array:
        G = N.get_disnet(ExaDisNet)
        # find node index
        tags = G.get_tags()
        ind = np.where((tags[:,0]==tag[0])&(tags[:,1]==tag[1]))[0]
        if ind.size != 1:
            raise ValueError("Cannot find node tag (%d,%d) in OneNodeMobility" % tuple(tag))
        # compute node force
        v = pyexadis.compute_node_mobility(G.net, ind[0], mobility=self.mobility, fi=np.array(f))
        v = np.array(v)
        # update force dictionary if needed
        if update_state:
            if "nodevels" in state and "nodeveltags" in state:
                nodeveltags = state["nodeveltags"]
                ind = np.where((nodeforcetags[:,0]==tag[0])&(nodeforcetags[:,1]==tag[1]))[0]
                if ind.size == 1:
                    state["nodevels"][ind[0]] = v
                else:
                    state["nodevels"] = np.vstack((state["nodevels"], v))
                    state["nodeveltags"] = np.vstack((state["nodeveltags"], tag))
            else:
                state["nodevels"] = np.array([v])
                state["nodeveltags"] = np.array([tag])
        return f


class MobilityLawPython:
    """MobilityLawPython: wrapper class for python-based mobility object
    This allows to call arbitrary mobility python modules from within ExaDiS
    """
    def __init__(self, mobility_module, state):
        self.mobility = mobility_module
        self.state = state
        
    def Mobility(self, net):
        G = ExaDisNet(net)
        # we need to update the current state forces, as forces
        # may have only been updated internally within ExaDiS
        self.state["nodeforces"] = G.get_forces()
        self.state["nodeforcetags"] = G.get_tags()
        N = DisNetManager(G)
        self.mobility.Mobility(N, self.state)
        nodevels = self.state["nodevels"]
        nodetags = self.state["nodeveltags"]
        net.set_velocities(nodevels, nodetags)
        
    def OneNodeMobility(self, net, tag, f):
        N = DisNetManager(ExaDisNet(net))
        return self.mobility.OneNodeMobility(N, self.state, tag, f)


class TimeIntegration:
    """TimeIntegration: wrapper class for time-integrator
    """
    def __init__(self, state: dict, integrator: str='EulerForward',
                 dt: float=1e-8, **kwargs) -> None:
        self.integrator_type = integrator
        self.dt = dt
        params = get_exadis_params(state)
        self.force_python = None
        self.mobility_python = None

        self.Update_Functions = {
            'EulerForward': self.Update_EulerForward,
            'Trapezoid': self.Integrate,
            'Subcycling': self.Integrate,
        }
        
        if self.integrator_type == 'EulerForward':
            self.params = params
        elif self.integrator_type == 'Trapezoid':
            multi = kwargs.get('multi', 0)
            
            force_module = kwargs.get('force')
            if isinstance(force_module, CalForce):
                if force_module.force_mode == 'SUBCYCLING_MODEL':
                    raise ValueError('Force SUBCYCLING_MODEL can only be used with Subcycling integrator')
            force, self.force_python = get_exadis_force(force_module, state, params)
            
            mobility_module = kwargs.get('mobility')
            mobility, self.mobility_python = get_exadis_mobility(mobility_module, state, params)
            
            if multi > 1:
                intparams = pyexadis.Integrator_Multi_Params(multi)
                self.integrator = pyexadis.make_integrator_trapezoid_multi(params=params, intparams=intparams, 
                                                                           force=force, mobility=mobility)
            else:
                intparams = pyexadis.Integrator_Trapezoid_Params()
                self.integrator = pyexadis.make_integrator_trapezoid(params=params, intparams=intparams, 
                                                                     force=force, mobility=mobility)
        elif self.integrator_type == 'Subcycling':
            rgroups = kwargs.get('rgroups')
            rtolth = kwargs.get('rtolth', 1.0)
            rtolrel = kwargs.get('rtolrel', 0.1)
            
            force_module = kwargs.get('force')
            if force_module.force_mode != 'SUBCYCLING_MODEL':
                raise ValueError('Force SUBCYCLING_MODEL must be used with Subcycling integrator')
            force = force_module.force
            
            mobility_module = kwargs.get('mobility')
            mobility, self.mobility_python = get_exadis_mobility(mobility_module, state, params)
            
            intparams = pyexadis.Integrator_Subcycling_Params(rgroups, rtolth, rtolrel)
            self.integrator = pyexadis.make_integrator_subclycing(params=params, intparams=intparams, 
                                                                 force=force, mobility=mobility)
        else:
            raise ValueError('Unknown integrator %s' % integrator)
        
    def Update(self, N: DisNetManager, state: dict) -> None:
        G = N.get_disnet(ExaDisNet)
        self.Update_Functions[self.integrator_type](G, state)
        state["dt"] = self.dt
        return state

    def Update_EulerForward(self, G: ExaDisNet, state: dict) -> None:
        v = state["nodevels"]
        nodetags = state.get("nodeveltags", np.empty((0,2)))
        self.dt = pyexadis.integrate_euler(G.net, params=self.params, dt=self.dt, nodevels=v, nodetags=nodetags)
        
    def Integrate(self, G: ExaDisNet, state: dict) -> None:
        applied_stress = state["applied_stress"]
        v = state["nodevels"]
        nodetags = state["nodeveltags"]
        # update state dictionary if force/mobility are python-based
        # so that ExaDiS can internally call the wrappers with up-to-date state
        if self.force_python is not None:
            self.force_python.state = state
        if self.mobility_python is not None:
            self.mobility_python.state = state
        self.dt = pyexadis.integrate(G.net, integrator=self.integrator, nodevels=v, nodetags=nodetags, applied_stress=applied_stress)
            

class Collision:
    """Collision: wrapper class for handling collisions
    """
    def __init__(self, state: dict, collision_mode: str='Proximity', **kwargs) -> None:
        self.collision_mode = collision_mode
        params = get_exadis_params(state)
        if params.rann < 0.0:
            params.rann = 2.0*params.rtol
        self.collision = pyexadis.make_collision(params=params)
        
    def HandleCol(self, N: DisNetManager, state: dict) -> None:
        G = N.get_disnet(ExaDisNet)
        oldnodes_dict = state.get('oldnodes_dict', None)
        dt = state.get('dt', 0.0)
        if oldnodes_dict != None:
            xold = oldnodes_dict["positions"]
            pyexadis.handle_collision(G.net, collision=self.collision, xold=xold, dt=dt)
        else:
            pyexadis.handle_collision(G.net, collision=self.collision)
        return state


class Topology:
    """Topology: wrapper class for handling topology (e.g. split multi nodes)
    """
    def __init__(self, state: dict, topology_mode: str='TopologyParallel', **kwargs) -> None:
        self.topology_mode = topology_mode
        params = get_exadis_params(state)
        splitMultiNodeAlpha = kwargs.get('splitMultiNodeAlpha', 1e-3)
        
        force_module = kwargs.get('force')
        force, self.force_python = get_exadis_force(force_module, state, params)
        if self.topology_mode == 'TopologyParallel' and self.force_python is not None:
            raise TypeError('TopologyParallel requires pyexadis force module')
        
        mobility_module = kwargs.get('mobility')
        mobility, self.mobility_python = get_exadis_mobility(mobility_module, state, params)
        if self.topology_mode == 'TopologyParallel' and self.mobility_python is not None:
            raise TypeError('TopologyParallel requires pyexadis mobility module')
        
        topolparams = pyexadis.Topology_Params(splitMultiNodeAlpha)
        self.topology = pyexadis.make_topology(topology_mode, params=params, topolparams=topolparams,
                                               force=force, mobility=mobility)
        
    def Handle(self, N: DisNetManager, state: dict) -> None:
        dt = state.get('dt', 0.0)
        G = N.get_disnet(ExaDisNet)
        # update state dictionary if force/mobility are python-based
        # so that ExaDiS can internally call the wrappers with up-to-date state
        if self.force_python is not None:
            self.force_python.state = state
        if self.mobility_python is not None:
            self.mobility_python.state = state
        pyexadis.handle_topology(G.net, topology=self.topology, dt=dt)
        return state


class Remesh:
    """Remesh: wrapper class for remeshing operations
    """
    def __init__(self, state: dict, remesh_rule: str='LengthBased', **kwargs) -> None:
        self.remesh_rule = remesh_rule
        params = get_exadis_params(state)
        self.remesh = pyexadis.make_remesh(params=params)
        
    def Remesh(self, N: DisNetManager, state: dict) -> None:
        G = N.get_disnet(ExaDisNet)
        pyexadis.remesh(G.net, remesh=self.remesh)
        return state
        

class SimulateNetwork:
    """SimulateNetwork: simulation driver
    """
    def __init__(self, state: dict, calforce=None, mobility=None, timeint=None, 
                 collision=None, topology=None, remesh=None, vis=None,
                 burgmag: float=1.0,
                 loading_mode: str='stress',
                 applied_stress: np.ndarray=np.zeros(6),
                 erate: float=0.0,
                 edir: np.ndarray=np.array([0.,0.,1.]),
                 max_step: int=10,
                 print_freq: int=None,
                 plot_freq: int=None,
                 plot_pause_seconds: float=None,
                 write_freq: int=None,
                 write_dir: str='',
                 exadis_plastic_strain: bool=True,
                 **kwargs) -> None:
        self.calforce = calforce
        self.mobility = mobility
        self.timeint = timeint
        self.collision = collision
        self.topology = topology
        self.remesh = remesh
        self.vis = vis
        self.burgmag = burgmag
        self.loading_mode = loading_mode
        self.erate = erate
        self.edir = np.array(edir)
        self.max_step = max_step
        self.print_freq = print_freq
        self.plot_freq = plot_freq
        self.plot_pause_seconds = plot_pause_seconds
        self.write_freq = write_freq
        self.write_dir = write_dir
        if self.write_dir and not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)
        
        self.exadis_plastic_strain = exadis_plastic_strain
        self.Etot = np.zeros(6)
        self.strain = 0.0
        self.stress = 0.0
        self.density = 0.0
        self.results = []
        
        state["applied_stress"] = np.array(applied_stress)
    
    def write_results(self):
        """write_results: write simulation results into a file
        """
        if self.loading_mode == 'strain_rate':
            np.savetxt('%s/stress_strain_dens.dat'%self.write_dir, np.array(self.results), fmt='%d %e %e %e %e')
    
    def save_old_nodes(self, N: DisNetManager, state: dict):
        """save_old_nodes: save current nodal positions
        """
        if self.exadis_plastic_strain:
            # if exadis is calculating plastic strain (much faster)
            # then we don't need to save positions here
            state["oldnodes_dict"] = None
        else:
            # TO DO: get_nodes_data() function from DisNetManager
            state["oldnodes_dict"] = N.get_disnet(ExaDisNet).get_nodes_data()
        return state
    
    def plastic_strain(self, N: DisNetManager, state: dict):
        """plastic_strain: compute plastic strain
        """
        if self.exadis_plastic_strain:
            # if exadis is calculating plastic strain (much faster)
            # the values will be fetched in update_mechanics() to
            # account for topological operations nodal motion as well
            dEp, dWp = np.zeros(6), np.zeros(3)
        else:
            # TO DO: get network data from DisNetManager, make last active network default
            G = N.get_disnet(ExaDisNet)
            data = G.export_data()
            nodes = data.get("nodes")
            segs = data.get("segs")
            cell = G.cell
            
            oldnodes_dict = state["oldnodes_dict"]
            rold = oldnodes_dict["positions"]
            
            r = nodes.get("positions")
            segsnid = segs.get("nodeids")
            burgs = segs.get("burgers")
            vol = cell.volume()
            
            r1 = r[segsnid[:,0]]
            r2 = np.array(cell.closest_image(Rref=r1, R=r[segsnid[:,1]]))
            r3 = np.array(cell.closest_image(Rref=r1, R=rold[segsnid[:,0]]))
            r4 = np.array(cell.closest_image(Rref=r3, R=rold[segsnid[:,1]]))
            n = 0.5*np.cross(r2-r3, r1-r4)
            P = np.multiply(n[:,[0,0,0,1,1,1,2,2,2]], burgs[:,[0,1,2,0,1,2,0,1,2]])
            dEp = 0.5/vol*np.sum(P[:,[0,4,8,1,2,5]] + P[:,[0,4,8,3,6,7]], axis=0)
            dWp = 0.5/vol*np.sum(P[:,[1,2,5]] - P[:,[3,6,7]], axis=0)
            density = np.linalg.norm(r2-r1, axis=1).sum()/vol/self.burgmag**2
            self.density = density
            
        state["dEp"] = dEp
        state["dWp"] = dWp
        
        return state
    
    def update_mechanics(self, N: DisNetManager, state: dict):
        """update_mechanics: update applied stress and rotation if needed
        """
        if self.loading_mode == 'strain_rate':
            
            if self.exadis_plastic_strain:
                # get values of plastic strain computed internally in exadis
                dEp, dWp, self.density = N.get_disnet(ExaDisNet).net.get_plastic_strain()
                dEp = np.array(dEp).ravel()[[0,4,8,1,2,5]]
                dWp = np.array(dWp).ravel()[[1,2,5]]
            else:
                dEp, dWp = state["dEp"], state["dWp"]
            
            # TO DO: add rotation
            A = np.outer(self.edir, self.edir)
            A = np.hstack([np.diag(A), 2.0*A.ravel()[[1,2,5]]])
            dpstrain = np.dot(dEp, A)
            dstrain = self.erate * self.timeint.dt
            Eyoung = 2.0 * self.calforce.mu * (1.0 + self.calforce.nu)
            dstress = Eyoung * (dstrain - dpstrain)
            state["applied_stress"] += dstress * A
            
            self.Etot += dstrain * A
            self.strain = np.dot(self.Etot, A)
            self.stress = np.dot(state["applied_stress"], A)
            
        return state
    
    def step(self, N: DisNetManager, state: dict):
        """step: take a time step of DD simulation on DisNetManager N
        """
        self.calforce.NodeForce(N, state)

        self.mobility.Mobility(N, state)
        
        self.save_old_nodes(N, state)
        
        self.timeint.Update(N, state)
        
        self.plastic_strain(N, state)

        if self.collision is not None:
            self.collision.HandleCol(N, state)
            
        if self.topology is not None:
            self.topology.Handle(N, state)

        if self.remesh is not None:
            self.remesh.Remesh(N, state)
            
        self.update_mechanics(N, state)
        
    def run(self, N: DisNetManager, state: dict):
        
        import time
        t0 = time.perf_counter()
        
        if self.vis != None and self.plot_freq != None:
            self.vis.plot_disnet(N, trim=True, block=False)
            
        if self.write_freq != None:
            N.get_disnet(ExaDisNet).write_data(os.path.join(self.write_dir, 'config.0.data'))
        
        # time stepping
        for tstep in range(self.max_step):
            self.step(N, state)

            if self.print_freq != None:
                if (tstep+1) % self.print_freq == 0:
                    dt = self.timeint.dt if self.timeint else 0.0
                    Nnodes = N.get_disnet(ExaDisNet).net.number_of_nodes()
                    if self.loading_mode == 'strain_rate':
                        print("step = %d, nodes = %d, dt = %e, strain = %e, stress = %e"%(tstep+1, Nnodes, dt, self.strain, self.stress))
                        self.results.append([tstep+1, self.strain, self.stress, self.density, time.perf_counter()-t0])
                    else:
                        print("step = %d, nodes = %d, dt = %e"%(tstep+1, Nnodes, dt))

            if self.vis != None and self.plot_freq != None:
                if (tstep+1) % self.plot_freq == 0:
                    self.vis.plot_disnet(N, trim=True, block=False, pause_seconds=self.plot_pause_seconds)
            
            if self.write_freq != None:
                if (tstep+1) % self.write_freq == 0:
                    N.get_disnet(ExaDisNet).write_data(os.path.join(self.write_dir, 'config.%d.data'%(tstep+1)))
                    # dump current results into file
                    if self.print_freq != None: self.write_results()
            
        # write results
        if self.print_freq != None:
            self.write_results()

        # plot final configuration
        if self.vis != None:
            self.vis.plot_disnet(N, trim=True, block=False)
            
        t1 = time.perf_counter()
        print('RUN TIME: %f sec' % (t1-t0))
        
        return state


class SimulateNetworkPerf(SimulateNetwork):
    """SimulateNetworkPerf: exadis simulation driver optimized for performance
    """
    def __init__(self, *args, **kwargs) -> None:
        super(SimulateNetworkPerf, self).__init__(*args, **kwargs)
        self.exadis_plastic_strain = True
        
    def update_mechanics(self, system, state: dict):
        """update_mechanics: update applied stress and rotation if needed
        """
        if self.loading_mode == 'strain_rate':
            # get values of plastic strain computed internally in exadis
            dEp, dWp, self.density = system.get_plastic_strain()
            dEp = np.array(dEp).ravel()[[0,4,8,1,2,5]]
            dWp = np.array(dWp).ravel()[[1,2,5]]
            
            # TO DO: add rotation
            A = np.outer(self.edir, self.edir)
            A = np.hstack([np.diag(A), 2.0*A.ravel()[[1,2,5]]])
            dpstrain = np.dot(dEp, A)
            dstrain = self.erate * self.timeint.dt
            Eyoung = 2.0 * self.calforce.mu * (1.0 + self.calforce.nu)
            dstress = Eyoung * (dstrain - dpstrain)
            state["applied_stress"] += dstress * A
            system.set_applied_stress(state["applied_stress"])
            
            self.Etot += dstrain * A
            self.strain = np.dot(self.Etot, A)
            self.stress = np.dot(state["applied_stress"], A)
            
            return state
    
    def step(self, system, state: dict):
        """step: take a time step of DD simulation on system
        See exadis/src/driver.cpp
        """
        # Do some force pre-computation for the step if needed
        self.calforce.force.pre_compute(system)
        
        # Nodal force calculation
        self.calforce.force.compute(system)
        
        # Mobility calculation
        self.mobility.mobility.compute(system)
        
        # Time-integration (plastic_strain() and reset_glide_planes() are also called internally)
        self.timeint.dt = self.timeint.integrator.integrate(system)
        
        # Collision
        if self.collision is not None:
            self.collision.collision.handle(system)
        
        # Topology
        if self.topology is not None:
            self.topology.topology.handle(system)
        
        # Remesh
        if self.remesh is not None:
            self.remesh.remesh.remesh(system)
        
        # Update stress
        self.update_mechanics(system, state)
        
    def run(self, N: DisNetManager, state: dict):
        
        import time
        t0 = time.perf_counter()
        
        # convert DisNet to a complete exadis system object
        system = pyexadis.System(N.get_disnet(ExaDisNet).net, self.calforce.params)
        system.set_neighbor_cutoff(self.calforce.force.neighbor_cutoff)
        system.set_applied_stress(state["applied_stress"])
            
        if self.write_freq != None:
            system.write_data(os.path.join(self.write_dir, 'config.0.data'))
        
        # time stepping
        for tstep in range(self.max_step):
            self.step(system, state)

            if self.print_freq != None:
                if (tstep+1) % self.print_freq == 0:
                    dt = self.timeint.dt if self.timeint else 0.0
                    Nnodes = system.number_of_nodes()
                    if self.loading_mode == 'strain_rate':
                        print("step = %d, nodes = %d, dt = %e, strain = %e, stress = %e"%(tstep+1, Nnodes, dt, self.strain, self.stress))
                        self.results.append([tstep+1, self.strain, self.stress, self.density, time.perf_counter()-t0])
                    else:
                        print("step = %d, nodes = %d, dt = %e"%(tstep+1, Nnodes, dt))
            
            if self.write_freq != None:
                if (tstep+1) % self.write_freq == 0:
                    system.write_data(os.path.join(self.write_dir, 'config.%d.data'%(tstep+1)))
                    # dump current results into file
                    if self.print_freq != None: self.write_results()
            
        # write results
        if self.print_freq != None:
            self.write_results()
            
        t1 = time.perf_counter()
        system.print_timers()
        print('RUN TIME: %f sec' % (t1-t0))
        
        return state


try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except ImportError:
    print('-----------------------------------------')
    print(' cannot import matplotlib or mpl_toolkits')
    print('-----------------------------------------')

class VisualizeNetwork:
    """VisualizeNetwork: class for plotting dislocation network
    """    
    def __init__(self, bounds=None, fig=None, ax=None, **kwargs) -> None:
        self.bounds = bounds
        self.fig, self.ax = fig, ax
        if self.fig == None:
            try: self.fig = plt.figure(figsize=(8,8))
            except NameError: print('plt not defined'); return
        if self.ax == None:
            try: self.ax = plt.axes(projection='3d')
            except NameError: print('plt not defined'); return
        
    def plot_disnet(self, N: DisNetManager,
                    plot_nodes=True, plot_segs=True, plot_cell=True, trim=False,
                    fig=None, ax=None, block=False, pause_seconds=0.01):
        if fig == None: fig = self.fig
        if ax == None: ax = self.ax
            
        data = N.export_data()
        
        # cell
        cell = data.get("cell")
        cell_origin = np.array(cell.get("origin", np.zeros(3)))
        cell = pyexadis.Cell(h=cell.get("h"), origin=cell_origin, is_periodic=cell.get("is_periodic"))
        h = np.array(cell.h)
        cell_center = cell_origin + 0.5*np.sum(h, axis=0)
        
        # nodes
        nodes = data.get("nodes")
        rn = nodes.get("positions")
        if rn.size > 0:
            rn = np.array(cell.closest_image(Rref=cell_center, R=rn))
            
            # segments
            segs = data.get("segs")
            if segs.get("nodeids").size == 0: plot_segs = False
            p_segs = np.empty((0,6))
            if plot_segs:
                segsnid = segs.get("nodeids")
                r1 = rn[segsnid[:,0]]
                r2 = np.array(cell.closest_image(Rref=cell_center, R=rn[segsnid[:,1]]))
                # handle pbc properly for segments that cross the cell boundary
                hinv = np.linalg.inv(h)
                d = np.max(np.abs(np.dot(r2-r1, hinv.T)), axis=1)
                p_segs = np.hstack((r1[d<0.5], r2[d<0.5]))
        else:
            plot_nodes, plot_segs = False, False

        plt.cla()
        
        # plot segments
        if plot_segs:
            ls = p_segs.reshape((-1,2,3))
            lc = Line3DCollection(ls, linewidths=0.5, colors='b')
            ax.add_collection(lc)

        # plot nodes
        if plot_nodes:
            ax.scatter(rn[:,0], rn[:,1], rn[:,2], c='r', s=4)
        
        # plot cell
        c = cell_origin + np.array([np.zeros(3), h[0], h[0]+h[1], h[1], h[2],
                                    h[0]+h[2], h[0]+h[1]+h[2], h[1]+h[2]])
        if plot_cell:
            boxedges = np.array([[0,1],[1,2],[2,3],[3,0],
                                [4,5],[5,6],[6,7],[7,4],
                                [0,4],[1,5],[2,6],[3,7]])
            bc = Line3DCollection(c[boxedges], linewidths=0.5, colors='k', alpha=0.5)
            ax.add_collection(bc)
        
        if not self.bounds is None:
            ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
            ax.set_ylim(self.bounds[0][1], self.bounds[1][1])
            ax.set_zlim(self.bounds[0][2], self.bounds[1][2])
        else:
            # cell bounding box
            ax.set_xlim(np.min(c[:,0]), np.max(c[:,0]))
            ax.set_ylim(np.min(c[:,1]), np.max(c[:,1]))
            ax.set_zlim(np.min(c[:,2]), np.max(c[:,2]))
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        try: ax.set_box_aspect([1,1,1])
        except AttributeError:
            #print('ax.set_box_aspect does not work')
            pass

        plt.draw()
        plt.show(block=block)
        plt.pause(pause_seconds)

        return fig, ax

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
    # Try importing DisNetManager from opendis
    from framework.disnet_manager import DisNetManager
except ImportError:
    # Use dummy DisNetManager if opendis is not available
    class DisNetManager:
        def __init__(self, disnet):
            self.disnet = list(disnet.values())[0]
        def get_disnet(self, type):
            return self.disnet

from enum import IntEnum
class NodeConstraints(IntEnum):
    UNCONSTRAINED = 0
    PINNED_NODE = 7

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Cannot import matplotlib')


class ExaDisNet:
    """ExaDisNet: wrapper class for exadis dislocation network
    Implements basic functions to manipulate the network
    """
    def __init__(self, *args):
        if len(args) == 3:
            cell, nodes, segs = args[0], args[1], args[2]
            self.net = pyexadis.ExaDisNet(cell=cell, nodes=nodes, segs=segs)
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
        cell = pyexadis.Cell(h=cell.get("h"), is_periodic=cell.get("is_periodic")) # need cell.origin
        self.net = pyexadis.ExaDisNet(cell=cell, nodes=data.get("nodes"), segs=data.get("segs"))
    
    def export_data(self):
        cell = self.net.get_cell()
        cell = {"h": np.array(cell.h), "origin": np.array(cell.origin()), "is_periodic": cell.get_pbc()}
        nodes = np.array(self.net.get_nodes_array())
        segs = np.array(self.net.get_segs_array())
        data = {"cell": cell, "nodes": nodes, "segs": segs}
        return data
    
    @property
    def cell(self):
        return self.net.get_cell()
    
    def get_positions(self):
        nodes = np.array(self.net.get_nodes_array())
        return nodes[:,0:3]
        
    def get_pos_dict(self):
        return {k: v for k, v in enumerate(self.get_positions())}
    
    def set_positions(self, pos: np.ndarray):
        self.net.set_positions(pos)
    

# This is only needed until the visualization is modified
# to accept generic DisNet_BASE types
try:
    # Try importing DisNet, Cell from pydis
    from pydis.disnet import DisNet, Cell
except ImportError:
    # Use dummy DisNet if pydis is not available
    DisNet = ExaDisNet
    Cell = pyexadis.Cell


def get_exadis_params(dict_params):
    """get_exadis_params: helper function to get exadis global parameters object
    """
    params = pyexadis.Params(
        burgmag=dict_params["burgmag"],
        mu=dict_params["mu"],
        nu=dict_params["nu"],
        a=dict_params["a"],
        maxseg=dict_params["maxseg"],
        minseg=dict_params["minseg"]
    )
    if "crystal" in dict_params: params.set_crystal(dict_params["crystal"])
    if "Rorient" in dict_params: params.Rorient = dict_params["Rorient"]
    if "rann" in dict_params: params.rann = dict_params["rann"]
    if "rtol" in dict_params: params.rtol = dict_params["rtol"]
    if "maxdt" in dict_params: params.maxdt = dict_params["maxdt"]
    if "nextdt" in dict_params: params.nextdt = dict_params["nextdt"]
    if "split3node" in dict_params: params.split3node = dict_params["split3node"]
    return params
    

class CalForce:
    """CalForce: wrapper class for calculating forces on dislocation network
    """
    def __init__(self, force_mode: str='LineTension', **kwargs) -> None:
        self.force_mode = force_mode
        self.params = get_exadis_params(kwargs.get('params'))
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
                cell = pyexadis.Cell(h=cell.h, is_periodic=cell.is_periodic) # need cell.origin
            self.force = pyexadis.make_force_ddd_fft(params=self.params, coreparams=coreparams, 
                                                     Ngrid=Ngrid, cell=cell)
            
        elif self.force_mode == 'SUBCYCLING_MODEL':
            Ngrid = kwargs.get('Ngrid')
            cell = kwargs.get('cell')
            if not isinstance(cell, pyexadis.Cell):
                cell = pyexadis.Cell(h=cell.h, is_periodic=cell.is_periodic) # need cell.origin
            drift = kwargs.get('drift', 1)
            self.force = pyexadis.make_force_subcycling(params=self.params, coreparams=coreparams,
                                                        Ngrid=Ngrid, cell=cell, drift=drift)
            
        else:
            raise ValueError('Unknown force %s' % force_mode)
            
    def NodeForce(self, N: DisNetManager, applied_stress: np.ndarray) -> dict:
        G = N.get_disnet(ExaDisNet)
        f = pyexadis.compute_force(G.net, force=self.force, applied_stress=applied_stress)
        nodeforce_dict = {k: v for k, v in enumerate(f)}
        return nodeforce_dict


class MobilityLaw:
    """MobilityLaw: wrapper class for mobility laws
    """
    def __init__(self, mobility_law: str='SimpleGlide', mob: float=1, **kwargs) -> None:
        self.mobility_law = mobility_law
        params = get_exadis_params(kwargs.get('params'))
        
        if self.mobility_law == 'SimpleGlide':
            Medge = kwargs.get('Medge', -1.0)
            Mscrew = kwargs.get('Mscrew', -1.0)
            if Medge > 0.0 and Mscrew > 0.0:
                mobparams = pyexadis.Mobility_GLIDE_Params(Medge, Mscrew)
            else:
                mobparams = pyexadis.Mobility_GLIDE_Params(mob)
            self.mobility = pyexadis.make_mobility_glide(params=params, mobparams=mobparams)
            
        elif self.mobility_law == 'BCC_0B':
            Medge = kwargs.get('Medge')
            Mscrew = kwargs.get('Mscrew')
            Mclimb = kwargs.get('Mclimb')
            vmax = kwargs.get('vmax', -1.0)
            mobparams = pyexadis.Mobility_BCC_0B_Params(Medge, Mscrew, Mclimb, vmax)
            self.mobility = pyexadis.make_mobility_bcc_0b(params=params, mobparams=mobparams)
            
        elif self.mobility_law == 'FCC_0':
            Medge = kwargs.get('Medge')
            Mscrew = kwargs.get('Mscrew')
            vmax = kwargs.get('vmax', -1.0)
            mobparams = pyexadis.Mobility_FCC_0_Params(Medge, Mscrew, vmax)
            self.mobility = pyexadis.make_mobility_fcc_0(params=params, mobparams=mobparams)
            
        else:
            raise ValueError('Unknown mobility law %s' % mobility_law)
        
    def Mobility(self, N: DisNetManager, nodeforce_dict: dict) -> dict:
        G = N.get_disnet(ExaDisNet)
        f = np.array(list(nodeforce_dict.values()))
        v = pyexadis.compute_mobility(G.net, mobility=self.mobility, nodeforces=f)
        vel_dict = {k: val for k, val in enumerate(v)}
        return vel_dict


class TimeIntegration:
    """TimeIntegration: wrapper class for time-integrator
    """
    def __init__(self, integrator: str='EulerForward', 
                 dt: float=1e-8, **kwargs) -> None:
        self.integrator_type = integrator
        self.dt = dt
        params = get_exadis_params(kwargs.get('params'))

        self.Update_Functions = {
            'EulerForward': self.Update_EulerForward,
            'Trapezoid': self.Integrate,
            'Subcycling': self.Integrate,
        }
        
        if self.integrator_type == 'EulerForward':
            pass
        elif self.integrator_type == 'Trapezoid':
            multi = kwargs.get('multi', 0)
            force = kwargs.get('force').force
            mobility = kwargs.get('mobility').mobility
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
            force = kwargs.get('force').force
            mobility = kwargs.get('mobility').mobility
            intparams = pyexadis.Integrator_Subcycling_Params(rgroups, rtolth, rtolrel)
            self.integrator = pyexadis.make_integrator_subclycing(params=params, intparams=intparams, 
                                                                 force=force, mobility=mobility)
        else:
            raise ValueError('Unknown integrator %s' % integrator)
        
    def Update(self, N: DisNetManager, vel_dict: dict, applied_stress: np.ndarray=np.zeros(6)) -> None:
        G = N.get_disnet(ExaDisNet)
        self.Update_Functions[self.integrator_type](G, vel_dict, applied_stress)

    def Update_EulerForward(self, G: ExaDisNet, vel_dict: dict, applied_stress: np.ndarray) -> None:
        v = np.array(list(vel_dict.values()))
        self.dt = pyexadis.integrate_euler(G.net, dt=self.dt, nodevels=v)
        
    def Integrate(self, G: ExaDisNet, vel_dict: dict, applied_stress: np.ndarray) -> None:
        v = np.array(list(vel_dict.values()))
        self.dt = pyexadis.integrate(G.net, integrator=self.integrator, nodevels=v, applied_stress=applied_stress)
            

class Collision:
    """Collision: wrapper class for handling collisions
    """
    def __init__(self, collision_mode: str='Proximity', **kwargs) -> None:
        self.collision_mode = collision_mode
        params = get_exadis_params(kwargs.get('params'))
        if params.rann < 0.0:
            params.rann = 2.0*params.rtol
            
        self.collision = pyexadis.make_collision(params=params)
        
    def HandleCol(self, N: DisNetManager, **kwargs) -> None:
        G = N.get_disnet(ExaDisNet)
        oldpos_dict = kwargs.get('oldpos_dict')
        dt = kwargs.get('dt', 0.0)
        if oldpos_dict != None:
            xold = np.array(list(oldpos_dict.values()))
            pyexadis.handle_collision(G.net, collision=self.collision, xold=xold, dt=dt)
        else:
            pyexadis.handle_collision(G.net, collision=self.collision)


class Topology:
    """Topology: wrapper class for handling topology (e.g. split multi nodes)
    """
    def __init__(self, topology_mode: str='TopologyParallel', **kwargs) -> None:
        self.topology_mode = topology_mode
        params = get_exadis_params(kwargs.get('params'))
        splitMultiNodeAlpha = kwargs.get('splitMultiNodeAlpha', 1e-3)
        force = kwargs.get('force').force
        mobility = kwargs.get('mobility').mobility
        topolparams = pyexadis.Topology_Params(splitMultiNodeAlpha)
        self.topology = pyexadis.make_topology(topology_mode, params=params, topolparams=topolparams,
                                               force=force, mobility=mobility)
        
    def Handle(self, N: DisNetManager, **kwargs) -> None:
        dt = kwargs.get('dt', 0.0)
        G = N.get_disnet(ExaDisNet)
        pyexadis.handle_topology(G.net, topology=self.topology, dt=dt)


class Remesh:
    """Remesh: wrapper class for remeshing operations
    """
    def __init__(self, remesh_rule: str='LengthBased', **kwargs) -> None:
        self.remesh_rule = remesh_rule
        params = get_exadis_params(kwargs.get('params'))

        self.remesh = pyexadis.make_remesh(params=params)
        
    def Remesh(self, N: DisNetManager, **kwargs) -> None:
        G = N.get_disnet(ExaDisNet)
        pyexadis.remesh(G.net, remesh=self.remesh)
        

class SimulateNetwork:
    """SimulateNetwork: simulation driver
    """
    def __init__(self, calforce=None, mobility=None, timeint=None, 
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
        self.applied_stress = np.array(applied_stress)
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
    
    def write_results(self):
        """write_results: write simulation results into a file
        """
        if self.loading_mode == 'strain_rate':
            np.savetxt('%s/stress_strain_dens.dat'%self.write_dir, np.array(self.results), fmt='%d %e %e %e %e')
    
    def save_old_positions(self, N: DisNetManager):
        """save_old_positions: save current nodal positions
        """
        if self.exadis_plastic_strain:
            # if exadis is calculating plastic strain (much faster)
            # then we don't need to save positions here
            oldpos_dict = {}
        else:
            # TO DO: get_pos_dict() function from DisNetManager
            oldpos_dict = N.get_disnet(ExaDisNet).get_pos_dict()
        return oldpos_dict
    
    def plastic_strain(self, N: DisNetManager, oldpos_dict: dict):
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
            
            r = nodes[:,0:3]
            rold = np.array(list(oldpos_dict.values()))
            segsnid = segs[:,0:2].astype(int)
            burgs = segs[:,2:5]
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
        
        return dEp, dWp
    
    def update_mechanics(self, N: DisNetManager, dEp: np.ndarray, dWp: np.ndarray):
        """update_mechanics: update applied stress and rotation if needed
        """
        if self.loading_mode == 'strain_rate':
            
            if self.exadis_plastic_strain:
                # get values of plastic strain computed internally in exadis
                dEp, dWp, self.density = N.get_disnet(ExaDisNet).net.get_plastic_strain()
                dEp = np.array(dEp).ravel()[[0,4,8,1,2,5]]
                dWp = np.array(dWp).ravel()[[1,2,5]]
            
            # TO DO: add rotation
            A = np.outer(self.edir, self.edir)
            A = np.hstack([np.diag(A), 2.0*A.ravel()[[1,2,5]]])
            dpstrain = np.dot(dEp, A)
            dstrain = self.erate * self.timeint.dt
            Eyoung = 2.0 * self.calforce.mu * (1.0 + self.calforce.nu)
            dstress = Eyoung * (dstrain - dpstrain)
            self.applied_stress += dstress * A
            
            self.Etot += dstrain * A
            self.strain = np.dot(self.Etot, A)
            self.stress = np.dot(self.applied_stress, A)
    
    def step(self, N: DisNetManager):
        """step: take a time step of DD simulation on DisNet G
        """
        nodeforce_dict = self.calforce.NodeForce(N, self.applied_stress)

        vel_dict = self.mobility.Mobility(N, nodeforce_dict)
        
        oldpos_dict = self.save_old_positions(N)
        
        self.timeint.Update(N, vel_dict, self.applied_stress)
        
        dEp, dWp = self.plastic_strain(N, oldpos_dict)

        if self.collision is not None:
            self.collision.HandleCol(N, oldpos_dict=oldpos_dict, dt=self.timeint.dt)
            
        if self.topology is not None:
            self.topology.Handle(N, dt=self.timeint.dt)

        if self.remesh is not None:
            self.remesh.Remesh(N)
            
        self.update_mechanics(N, dEp, dWp)
        
    def run(self, N: DisNetManager):
        
        import time
        t0 = time.perf_counter()
        
        if self.vis != None and self.plot_freq != None:
            try: 
                fig = plt.figure(figsize=(8,8))
                ax = plt.axes(projection='3d')
            except NameError: print('plt not defined'); return
            # plot initial configuration
            G = N.get_disnet(DisNet)
            self.vis.plot_disnet(G, fig=fig, ax=ax, trim=True, block=False)
            
        if self.write_freq != None:
            N.get_disnet(ExaDisNet).write_data(os.path.join(self.write_dir, 'config.0.data'))
        
        # time stepping
        for tstep in range(self.max_step):
            self.step(N)

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
                    G = N.get_disnet(DisNet)
                    self.vis.plot_disnet(G, fig=fig, ax=ax, trim=True, block=False, pause_seconds=self.plot_pause_seconds)
            
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
            G = N.get_disnet(DisNet)
            self.vis.plot_disnet(G, fig=fig, ax=ax, trim=True, block=False)
            
        t1 = time.perf_counter()
        print('RUN TIME: %f sec' % (t1-t0))


class SimulateNetworkPerf(SimulateNetwork):
    """SimulateNetworkPerf: exadis simulation driver optimized for performance
    """
    def __init__(self, *args, **kwargs) -> None:
        super(SimulateNetworkPerf, self).__init__(*args, **kwargs)
        self.exadis_plastic_strain = True
        
    def update_mechanics(self, system):
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
            self.applied_stress += dstress * A
            system.set_applied_stress(self.applied_stress)
            
            self.Etot += dstrain * A
            self.strain = np.dot(self.Etot, A)
            self.stress = np.dot(self.applied_stress, A)
    
    def step(self, system):
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
        self.update_mechanics(system)
        
    def run(self, N: DisNetManager):
        
        import time
        t0 = time.perf_counter()
        
        # convert DisNet to a complete exadis system object
        system = pyexadis.System(N.get_disnet(ExaDisNet).net, self.calforce.params)
        system.set_neighbor_cutoff(self.calforce.force.neighbor_cutoff)
        system.set_applied_stress(self.applied_stress)
            
        if self.write_freq != None:
            system.write_data(os.path.join(self.write_dir, 'config.0.data'))
        
        # time stepping
        for tstep in range(self.max_step):
            self.step(system)

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
    def __init__(self, bounds=None, **kwargs) -> None:
        self.bounds = bounds
        
    def closest_image(self, cell, Rref, R):
        # TO BE REMOVED: only for backward compatibility for now
        try:
            R = np.array(cell.closest_image(Rref=Rref, R=R))
        except:
            R = cell.map_to(R, Rref)
        return R

    def plot_disnet(self, DM: DisNet, # should be DisNetManager eventually
                    plot_nodes=True, plot_segs=True, plot_cell=True, trim=False,
                    fig=None, ax=None, block=False, pause_seconds=0.01):
        if fig==None:
            try: fig = plt.figure(figsize=(8,8))
            except NameError: print('plt not defined'); return
        if ax==None:
            try: ax = plt.axes(projection='3d')
            except NameError: print('plt not defined'); return
            
        data = DM.export_data() # TO DO: export_data() from DisNetManager
        
        # cell
        cell = data.get("cell")
        cell_origin = np.array(cell.get("origin", np.zeros(3))) # TO DO: cell.origin
        cell = Cell(h=cell.get("h"), origin=cell_origin, is_periodic=cell.get("is_periodic"))
        h = np.array(cell.h)
        cell_center = cell_origin + 0.5*np.sum(h, axis=0)
        
        # nodes
        rn = np.array(data.get("nodes"))
        if rn.size > 0:
            rn = self.closest_image(cell, Rref=cell_center, R=rn[:,0:3])
            
            # segments
            segs = np.array(data.get("segs"))
            if segs.size == 0: plot_segs = False
            p_segs = np.empty((0,6))
            if plot_segs:
                segsnid = segs[:,0:2].astype(int)
                r1 = rn[segsnid[:,0]]
                r2 = self.closest_image(cell, Rref=cell_center, R=rn[segsnid[:,1]])
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

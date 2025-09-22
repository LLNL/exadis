import os, sys, io
import numpy as np

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
except ImportError:
    raise ImportError('Cannot import pyexadis')


class CalForceFFT:
    def __init__(self, state: dict, **kwargs) -> None:
        from pyexadis_base import get_module_arg, get_exadis_params
        params = get_exadis_params(state)
        self.Ngrid = get_module_arg('CalForceFFT', kwargs, 'Ngrid')
        if isinstance(self.Ngrid, int): self.Ngrid = 3*[self.Ngrid]
        cell = get_module_arg('CalForceFFT', kwargs, 'cell')
        fftparams = pyexadis.Force.ForceFFT.Params(Ngrid=self.Ngrid)
        self.forcefft = pyexadis.Force.ForceFFT.make(params=params, fparams=fftparams, cell=cell)
        
    def PreCompute(self, N: DisNetManager, state: dict) -> dict:
        G = N.get_disnet(ExaDisNet)
        self.forcefft.pre_compute_force(G.net)
        return state
    
    def NodeForce(self, N: DisNetManager, state: dict, pre_compute=True) -> dict:
        if pre_compute:
            self.PreCompute(N, state)

        G = N.get_disnet(ExaDisNet)
        f = self.forcefft.compute_force(G.net, applied_stress=np.zeros(6), pre_compute=pre_compute)
        
        state["nodeforces"] = np.array(f)
        state["nodeforcetags"] = N.export_data()["nodes"]["tags"]
        return state
    
    def OneNodeForce(self, N: DisNetManager, state: dict, tag, update_state=True) -> np.array:
        f = np.zeros(3)
        raise TypeError("OneNodeForce not implemented for CalForceFFT")
        return f
        

def test_force(name='lt'):
    
    pyexadis.initialize(verbose=False)
    
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
    
    G = ExaDisNet().read_paradis('../../examples/22_fcc_Cu_15um_1e3/180chains_16.10e.data', verbose=False)
    #G = ExaDisNet().generate_prismatic_config(state["crystal"], 50000.0, 1, 2000.0, maxseg=state["maxseg"], seed=1234)
    N = DisNetManager(G)
    
    state["applied_stress"] = np.array([10e6, 5e6, 20e6, 3e6, 7e6, 1e6])
    
    if name == 'lt':
        calforce = CalForce(force_mode='LINE_TENSION_MODEL', state=state)
    elif name == 'cutoff':
        calforce = CalForce(force_mode='CUTOFF_MODEL', state=state, cutoff=7500.0)
    elif name == 'ddd_fft':
        calforce = CalForce(force_mode='DDD_FFT_MODEL', state=state, Ngrid=64, cell=N.cell)
    elif name == 'fft':
        calforce = CalForceFFT(state=state, Ngrid=64, cell=N.cell)
    else:
        raise ValueError(f"Invalid force type = '{name}'")
    
    calforce.PreCompute(N, state)
    calforce.NodeForce(N, state, pre_compute=False)
    
    results = ''
    for f in state["nodeforces"]:
        results += '%e %e %e\n' % tuple(f)
    
    if 0:
        # write reference results in a file
        with open(f"expected_output/test_force_{name}.dat", 'w') as f:
            f.write(results)
    else:
        # print current results in the console
        print(results)
    
    pyexadis.finalize()
    

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    test_force(name)

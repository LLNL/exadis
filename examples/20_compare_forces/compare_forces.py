import os, sys
import numpy as np

# Import pydis
sys.path.extend([
    os.path.abspath('../../../../python'),
    os.path.abspath('../../../../lib'),
    os.path.abspath('../../../../core/pydis/python')
])
try:
    from framework.disnet_manager import DisNetManager
    from pydis.disnet import DisNet, Cell
    from pydis.calforce.calforce_disnet import CalForce as CalForce_pydis
except ImportError:
    raise ImportError('Cannot import pydis')

# Import pyexadis
pyexadis_path = '../../python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet
    from pyexadis_base import CalForce as CalForce_pyexadis
except ImportError:
    raise ImportError('Cannot import pyexadis')


def init_circular_loop(Lbox=10.0, radius=1.0, Nnodes=20, burg_vec=np.array([1.0,0.0,0.0]), pbc=False):
    print("init_circular_loop: Lbox = %f, radius = %f, N = %d" % (Lbox, radius, Nnodes))
    cell = Cell(h=Lbox*np.eye(3), is_periodic=[pbc,pbc,pbc])
    theta = np.arange(Nnodes)*2.0*np.pi/Nnodes
    rn    = np.vstack([radius*np.cos(theta), radius*np.sin(theta), np.zeros_like(theta)]).T
    links = np.zeros((Nnodes, 8))
    for i in range(Nnodes):
        links[i,:] = np.concatenate(([i, (i+1)%Nnodes], burg_vec, np.zeros(3)))
    N = DisNetManager(DisNet(cell=cell, rn=rn, links=links))
    return N


def example1():
    """example1():
    Example of a script to compare line-tension nodal forces
    computed using pydis and pyexadis
    """
    
    print('EXAMPLE1')

    mu, nu, a = 50.0e9, 0.3, 1.0
    Ec = 1.0e6
    applied_stress = np.array([0.0, 0.0, 0.0, 0.0, -2.0e6, 0.0])

    N = init_circular_loop()
    
    state = {"burgmag": 1.0, "mu": mu, "nu": nu, "a": a, "maxseg": 0.3, "minseg": 0.1}
    state["applied_stress"] = applied_stress
    
    # pydis
    calforce = CalForce_pydis(state=state, Ec=Ec, force_mode='LineTension')
    state = calforce.NodeForce(N, state)
    f_pydis = state["nodeforces"]
    print('pydis',f_pydis)
    
    # exadis
    calforce = CalForce_pyexadis(state=state, Ec=Ec, force_mode='LineTension')
    state = calforce.NodeForce(N, state)
    f_pyexadis = state["nodeforces"]
    print('f_pyexadis',f_pyexadis)
    
    print('PASS' if np.allclose(f_pydis/mu, f_pyexadis/mu) else 'FAIL')

    print('EXAMPLE1 DONE')
    

def example2():
    """example2():
    Example of a script to compare N^2 nodal forces
    computed using pydis and pyexadis
    """
    import time
    from pydis.calforce.compute_stress_force_analytic_paradis import compute_segseg_force
    
    class CalForceN2():
        def __init__(self, mu, nu, a):
            self.mu = mu
            self.nu = nu
            self.a = a
            
        def NodeForce(self, G: DisNet):
            segs_data = G.get_segs_data_with_positions()
            source_tags = segs_data["tag1"]
            target_tags = segs_data["tag2"]
            R1 = segs_data["R1"]
            R2 = segs_data["R2"]
            burg_vecs = segs_data["burgers"]
            Nseg = segs_data["nodeids"].shape[0]
            Nnodes = len(G.all_nodes_tags())
            f = np.zeros((Nnodes,3))
            for i in range(Nseg):
                for j in range(i+1, Nseg):
                    p1 = R1[i,:].copy()
                    p2 = R2[i,:].copy()
                    p3 = R1[j,:].copy()
                    p4 = R2[j,:].copy()
                    b12 = burg_vecs[i,:].copy()
                    b34 = burg_vecs[j,:].copy()
                    # PBC
                    p2 = G.cell.closest_image(Rref=p1, R=p2)
                    p3 = G.cell.closest_image(Rref=p1, R=p3)
                    p4 = G.cell.closest_image(Rref=p3, R=p4)
                    f1, f2, f3, f4 = compute_segseg_force(p1, p2, p3, p4, b12, b34, self.mu, self.nu, self.a)
                    n1, n2 = source_tags[i][1], target_tags[i][1]
                    n3, n4 = source_tags[j][1], target_tags[j][1]
                    f[n1] += f1
                    f[n2] += f2
                    f[n3] += f3
                    f[n4] += f4     
            return f
    
    MU, NU, a = 50.0e9, 0.3, 1.0
    
    def compute_pydis(N):
        print('Compute pydis...')
        calforce = CalForceN2(mu=MU, nu=NU, a=a)
        G = N.get_disnet(DisNet)
        t0 = time.time()
        f = calforce.NodeForce(G)
        t1 = time.time()
        print(f'compute_pydis time: {t1-t0} sec')
        print('f_pydis',f)
        return f
    
    def compute_pyexadis(N):
        print('Compute pyexadis...')
        G = N.get_disnet(ExaDisNet)
        t0 = time.time()
        f = np.array(pyexadis.compute_force_n2(G.net, mu=MU, nu=NU, a=a))
        t1 = time.time()
        print(f'compute_pyexadis time: {t1-t0} sec')
        print('f_pyexadis',f)
        return f
    
    print('EXAMPLE2')
    
    G = ExaDisNet()
    G.generate_prismatic_config(crystal='bcc', Lbox=300.0, num_loops=24, radius=60.0)
    N = DisNetManager(G)
    
    f_pydis = compute_pydis(N)
    f_pyexadis = compute_pyexadis(N)
    
    print('PASS' if np.allclose(f_pydis/MU, f_pyexadis/MU) else 'FAIL')
    
    print('EXAMPLE2 DONE')
    
    
if __name__ == "__main__":
    
    pyexadis.initialize()
    
    example1()
    example2()
    
    pyexadis.finalize()

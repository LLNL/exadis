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


def init_circular_loop(Lbox=10.0, radius=1.0, N=20, burg_vec=np.array([1.0,0.0,0.0]), pbc=False):
    print("init_circular_loop: Lbox = %f, radius = %f, N = %d" % (Lbox, radius, N))
    cell = Cell(h=Lbox*np.eye(3), is_periodic=[pbc,pbc,pbc])
    G = DisNet(cell=cell)
    theta = np.arange(N)*2.0*np.pi/N
    rn    = np.vstack([radius*np.cos(theta), radius*np.sin(theta), np.zeros_like(theta)]).T
    links = np.zeros((N, 8))
    for i in range(N):
        links[i,:] = np.concatenate(([i, (i+1)%N], burg_vec, np.zeros(3)))
    G.add_nodes_links_from_list(rn, links)
    return G


'''
Example of a script to compare line-tension nodal forces
computed using pydis and pyexadis
'''
def example1():
    
    print('EXAMPLE1')
    pyexadis.initialize()

    mu, nu, a = 50.0e9, 0.3, 1.0
    Ec = 1.0e6
    applied_stress = np.array([0.0, 0.0, 0.0, 0.0, -2.0e6, 0.0])

    G = init_circular_loop()
    N = DisNetManager({type(G): G})
    
    # pydis
    calforce = CalForce_pydis(mu=mu, nu=nu, a=a, Ec=Ec, force_mode='LineTension',
                        applied_stress=applied_stress)
    nodeforce_dict, segforce_dict = calforce.NodeForce(N)
    f_pydis = np.array(list(nodeforce_dict.values()))
    print('pydis',f_pydis)
    
    # exadis
    params = {"burgmag": 1.0, "mu": mu, "nu": nu, "a": a, "maxseg": 0.3, "minseg": 0.1}
    calforce = CalForce_pyexadis(params=params, Ec=Ec, force_mode='LineTension')
    nodeforce_dict = calforce.NodeForce(N, applied_stress=applied_stress)
    f_pyexadis = np.array(list(nodeforce_dict.values()))
    print('f_pyexadis',f_pyexadis)
    
    print('PASS' if np.allclose(f_pydis/mu, f_pyexadis/mu) else 'FAIL')

    pyexadis.finalize()
    print('EXAMPLE1 DONE')
    
    
'''
Example of a script to compare N^2 nodal forces
computed using pydis and pyexadis
'''
def example2():
    import time
    from pydis.calforce.compute_stress_force_analytic_paradis import compute_segseg_force
    
    class CalForceN2():
        def __init__(self, mu, nu, a):
            self.mu = mu
            self.nu = nu
            self.a = a
            
        def NodeForce(self, G: DisNet):
            segments = G.seg_list()
            N = len(segments)
            nseg = len(segments)
            ntot = nseg*(nseg-1)//2
            k = 0
            f = np.zeros((G._G.number_of_nodes(),3))
            for i in range(nseg):
                for j in range(i+1, nseg):
                    seg1 = segments[i]
                    seg2 = segments[j]
                    p1 = np.array(seg1["R1"])
                    p2 = np.array(seg1["R2"])
                    p3 = np.array(seg2["R1"])
                    p4 = np.array(seg2["R2"])
                    b12 = np.array(seg1["burg_vec"])
                    b34 = np.array(seg2["burg_vec"])
                    # PBC
                    p2 = G.cell.map_to(p2, p1)
                    p3 = G.cell.map_to(p3, p1)
                    p4 = G.cell.map_to(p4, p3)
                    f1, f2, f3, f4 = compute_segseg_force(p1, p2, p3, p4, b12, b34, self.mu, self.nu, self.a)
                    n1 = seg1["edge"][0][1]
                    n2 = seg1["edge"][1][1]
                    n3 = seg2["edge"][0][1]
                    n4 = seg2["edge"][1][1]
                    f[n1] += f1
                    f[n2] += f2
                    f[n3] += f3
                    f[n4] += f4
                    k += 1        
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
    pyexadis.initialize()
    
    datafile = '../../tests/data/taneg_001_bcc0b_1.data'
    G = ExaDisNet()
    G.read_paradis(datafile)
    N = DisNetManager({type(G): G})
    
    f_pydis = compute_pydis(N)
    f_pyexadis = compute_pyexadis(N)
    
    print('PASS' if np.allclose(f_pydis/MU, f_pyexadis/MU) else 'FAIL')
    
    pyexadis.finalize()
    print('EXAMPLE2 DONE')
    
    
if __name__ == "__main__":
    
    #example1()
    example2()

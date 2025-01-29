#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include "vec.h"
#include "network.h"
#include "force_common.h"

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

typedef ExaDiS::Vec3 Vec3;
typedef ExaDiS::Mat33 Mat33;
namespace ExaDiS {
    SerialDisNet* read_paradis(const char* file);
}

extern std::string exadis_root_dir;

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<>
    struct reduction_identity<ExaDiS::Vec3> {
        KOKKOS_FORCEINLINE_FUNCTION static ExaDiS::Vec3 sum() {
            return ExaDiS::Vec3();
        }
    };
} // namespace Kokkos

struct Node {
    int flag;
    Vec3 pos;
    Vec3 f;
    Vec3 v;
    KOKKOS_FORCEINLINE_FUNCTION Node() {}
    KOKKOS_FORCEINLINE_FUNCTION Node(int _flag, Vec3 _pos) : flag(_flag), pos(_pos) {}
};

struct Seg {
    int n1, n2;
    Vec3 burg;
    Vec3 plane;
    KOKKOS_FORCEINLINE_FUNCTION Seg() {}
    KOKKOS_FORCEINLINE_FUNCTION Seg(int _n1, int _n2, Vec3 _b, Vec3 _p) : n1(_n1), n2(_n2), burg(_b), plane(_p) {}
};
#if 1
struct Conn {
    //static const int MAX_CONNS = 10; // from network.h
    int num;
    int node[MAX_CONN], seg[MAX_CONN], order[MAX_CONN];
    KOKKOS_INLINE_FUNCTION Conn() { num = 0; }
    KOKKOS_INLINE_FUNCTION Conn(const ExaDiS::Conn& conn) {
        num = conn.num;
        for (int i = 0; i < num; i++) {
            node[i] = conn.node[i];
            seg[i] = conn.seg[i];
            order[i] = conn.order[i];
        }
    }
    KOKKOS_FORCEINLINE_FUNCTION void add_connection(int n, int s, int o) {
        node[num] = n; seg[num] = s; order[num] = o; num++;
    }
};
#else
typedef ExaDiS::Conn Conn;
#endif

#define UNIFIED_MEMORY 1

#if UNIFIED_MEMORY
typedef typename Kokkos::SharedSpace T_memory_space;
#else
typedef typename Kokkos::DefaultExecutionSpace::memory_space T_memory_space;
#endif

template<typename T, typename... Args>
inline T* exadis_new(Args... args) {
#if 1 //UNIFIED_MEMORY
    void* p = Kokkos::kokkos_malloc<Kokkos::SharedSpace>(sizeof(T));
    return new(p) T(args...);
#else
    return new T(args...);
#endif
}

template<typename T>
inline void exadis_delete(T *p) {
#if 1 //UNIFIED_MEMORY
    if (p) p->~T();
    Kokkos::kokkos_free<Kokkos::SharedSpace>(p);
#else
    if (p) delete p;
#endif
}

struct System {
    ExaDiS::Cell cell;
    Kokkos::View<Node*, T_memory_space> nodes;
    Kokkos::View<Seg*, T_memory_space> segs;
    Kokkos::View<Conn*, T_memory_space> conn;
    int Nnodes, Nsegs;

    Kokkos::Timer timer;
    double time_force = 0.0;
    double time_mobility = 0.0;

    System(ExaDiS::SerialDisNet* network) {
        cell = network->cell;
        Kokkos::resize(nodes, network->number_of_nodes());
        Kokkos::resize(segs, network->number_of_segs());
        Kokkos::resize(conn, network->number_of_nodes());
        Nnodes = network->number_of_nodes();
        Nsegs = network->number_of_segs();

    #if UNIFIED_MEMORY
        for (int i = 0; i < network->number_of_nodes(); i++)
            nodes(i) = Node(0, network->nodes[i].pos);
        for (int i = 0; i < network->number_of_segs(); i++)
            segs(i) = Seg(network->segs[i].n1, network->segs[i].n2, network->segs[i].burg, network->segs[i].plane);
        for (int i = 0; i < network->number_of_nodes(); i++)
            conn(i) = network->conn[i];
    #else
        auto h_nodes = create_mirror_view(nodes);
        for (int i = 0; i < network->number_of_nodes(); i++)
            h_nodes(i) = Node(0, network->nodes[i].pos);
        Kokkos::deep_copy(nodes, h_nodes);
        
        auto h_segs = create_mirror_view(segs);
        for (int i = 0; i < network->number_of_segs(); i++)
            h_segs(i) = Seg(network->segs[i].n1, network->segs[i].n2, network->segs[i].burg, network->segs[i].plane);
        Kokkos::deep_copy(segs, h_segs);

        auto h_conn = create_mirror_view(conn);
        for (int i = 0; i < network->number_of_nodes(); i++)
            h_conn(i) = network->conn[i];
        Kokkos::deep_copy(conn, h_conn);
    #endif
    }
};


KOKKOS_INLINE_FUNCTION
void SegSegForceTest(const Vec3& r1, const Vec3& r2, const Vec3& r3, const Vec3& r4, 
                     const Vec3& b1, const Vec3& b2, double a, double MU, double NU,
                     Vec3& f1, Vec3& f2, Vec3& f3, Vec3& f4, int flag12, int flag34)
{
    f1 = f2 = f3 = f4 = MU*dot(b1, b2)*cross(r2-r1, r4-r3);
}

struct SegSeg {
    int s1, s2;
    KOKKOS_FORCEINLINE_FUNCTION SegSeg() {}
    KOKKOS_FORCEINLINE_FUNCTION SegSeg(int _s1, int _s2) : s1(_s1), s2(_s2) {}
};

struct SegSegForce {
    Vec3 f1, f2, f3, f4;
    KOKKOS_INLINE_FUNCTION SegSegForce() { f1 = f2 = f3 = f4 = Vec3(0.0); }
    KOKKOS_INLINE_FUNCTION SegSegForce(const Vec3& _f1, const Vec3& _f2, const Vec3& _f3, const Vec3& _f4) 
    : f1(_f1), f2(_f2), f3(_f3), f4(_f4) {}
};

struct SegSegIso
{
    double MU = 50.0e9;
    double NU = 0.3;
    double a = 1.0;
    
    SegSegIso() {}
    SegSegIso(System* system) {}

    KOKKOS_INLINE_FUNCTION
    SegSegForce segseg_force(System* system, const SegSeg& ss, 
                             int compute_seg12=1, int compute_seg34=1) const
    {
        int n1 = system->segs(ss.s1).n1;
        int n2 = system->segs(ss.s1).n2;
        Vec3 b1 = system->segs(ss.s1).burg;
        Vec3 r1 = system->nodes(n1).pos;
        Vec3 r2 = system->cell.pbc_position(r1, system->nodes(n2).pos);
        double l1 = (r2-r1).norm2();
            
        int n3 = system->segs(ss.s2).n1;
        int n4 = system->segs(ss.s2).n2;
        Vec3 b2 = system->segs(ss.s2).burg;
        Vec3 r3 = system->cell.pbc_position(r1, system->nodes(n3).pos);
        Vec3 r4 = system->cell.pbc_position(r3, system->nodes(n4).pos);
        double l2 = (r4-r3).norm2();

        Vec3 f1(0.0), f2(0.0), f3(0.0), f4(0.0);
        if (l1 >= 1.e-20 && l2 >= 1.e-20) {
            SegSegForceIsotropic(r1, r2, r3, r4, b1, b2, a, MU, NU, 
                                 f1, f2, f3, f4, compute_seg12, compute_seg34);
            //SegSegForceTest(r1, r2, r3, r4, b1, b2, a, MU, NU, f1, f2, f3, f4, 0, 0);
        }
        
        return SegSegForce(f1, f2, f3, f4);
    }
};

template<class F>
struct ForceBase {

    double MU = 50.0e9;
    double NU = 0.3;
    double a = 1.0;
    
    System* system;
    F force;

    ForceBase(System* _system) : system(_system) {}

    struct TagZero {};
    struct TagCompute1 {};
    struct TagCompute2 {};

    KOKKOS_INLINE_FUNCTION
    void operator() (TagZero, int i) const {
        system->nodes(i).f = Vec3(0.0);
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagCompute1, int i) const {
        
        int n1 = system->segs(i).n1;
        int n2 = system->segs(i).n2;

        Vec3 fs1(0.0), fs2(0.0);
        
        for (int j = 0; j < system->Nsegs; j++) {
            if (j == i) continue; // skip self-force
            
            SegSegForce fs = force.segseg_force(system, SegSeg(i, j), 1, 0);
            fs1 += fs.f1;
            fs2 += fs.f2;
        }
        
        // Increment nodal forces
        Kokkos::atomic_add(&system->nodes(n1).f, fs1);
        Kokkos::atomic_add(&system->nodes(n2).f, fs2);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (TagCompute2, int i) const {
        
        int n1 = system->segs(i).n1;
        int n2 = system->segs(i).n2;
        Vec3 b1 = system->segs(i).burg;
        Vec3 r1 = system->nodes(n1).pos;
        Vec3 r2 = system->cell.pbc_position(r1, system->nodes(n2).pos);
        double l1 = (r2-r1).norm2();

        Vec3 fs1(0.0), fs2(0.0);
        
        for (int j = 0; j < system->Nsegs; j++) {
            if (j == i) continue; // skip self-force
            
            int n3 = system->segs(j).n1;
            int n4 = system->segs(j).n2;
            Vec3 b2 = system->segs(j).burg;
            Vec3 r3 = system->cell.pbc_position(r1, system->nodes(n3).pos);
            Vec3 r4 = system->cell.pbc_position(r3, system->nodes(n4).pos);
            double l2 = (r4-r3).norm2();

            Vec3 f1(0.0), f2(0.0), f3(0.0), f4(0.0);
            if (l1 >= 1.e-20 && l2 >= 1.e-20) {
                SegSegForceIsotropic(r1, r2, r3, r4, b1, b2, a, MU, NU, f1, f2, f3, f4, 1, 0);
                //SegSegForceTest(r1, r2, r3, r4, b1, b2, a, MU, NU, f1, f2, f3, f4, 1, 0);
            }
            fs1 += f1;
            fs2 += f2;
        }
        
        // Increment nodal forces
        Kokkos::atomic_add(&system->nodes(n1).f, fs1);
        Kokkos::atomic_add(&system->nodes(n2).f, fs2);
    }
};

typedef ForceBase<SegSegIso> Force;

template<typename TagCompute, bool use_lauchbounds=false>
void force(System* system)
{
    system->timer.reset();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Force::TagZero>(0, system->Nnodes),
        Force(system)
    );
    Kokkos::fence();

    if (use_lauchbounds) {
        const unsigned int MaxThreads = 16;
        const unsigned int MinBlocks = 1;
        using policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<MaxThreads,MinBlocks>,TagCompute>;
        Kokkos::parallel_for("ComputeForce",
            policy(0, system->Nsegs),
            Force(system)
        );
    } else {
        using policy = Kokkos::RangePolicy<TagCompute>;
        Kokkos::parallel_for("ComputeForce",
            policy(0, system->Nsegs),
            Force(system)
        );
    }
    Kokkos::fence();

    system->time_force += system->timer.seconds();
}

struct Mobility {

    double Medge = 1.0;
    double Mscrew = 1.0;
    
    System* system;

    Mobility(System* _system) : system(_system) {}
        
    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const {
        system->nodes(i).v = node_velocity(system, i, system->nodes(i).f);
    }

    KOKKOS_INLINE_FUNCTION
    Mat33 glide_constraints(int nconn, Vec3* norm) const
    {
        Mat33 P = Mat33().eye();
        
        // Find independent glide constraints
        for (int j = 0; j < nconn; j++) {
            for (int k = 0; k < j; k++)
                norm[j] = norm[j].orthogonalize(norm[k]);
            if (norm[j].norm2() >= 0.05) {
                norm[j] = norm[j].normalized();
                Mat33 Q = Mat33().eye() - outer(norm[j], norm[j]);
                P = Q * P;
            }
        }
        
        // Zero-out tiny non-zero components due to round-off errors
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (fabs(P[i][j]) < 1e-10) P[i][j] = 0.0;
        
        return P;
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 node_velocity(System* system, const int& i, const Vec3& fi) const
    {
        auto nodes = system->nodes.data();
        auto segs = system->segs.data();
        auto conn = system->conn.data();
        auto cell = system->cell;
        
        Vec3 vi(0.0);
        
        int nconn = conn[i].num;
        if (nconn >= 2) {

            double eps = 1e-10;
            
            Vec3 r1 = nodes[i].pos;
            Vec3 norm[MAX_CONN];
            
            int numNonZeroLenSegs = 0;
            double LtimesB = 0.0;
            for (int j = 0; j < nconn; j++) {

                int k = conn[i].node[j];
                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
                Vec3 dr = r2-r1;
                double L = dr.norm();
                if (L < eps) continue;
                numNonZeroLenSegs++;
                dr = 1.0/L * dr;
                
                int s = conn[i].seg[j];
                int order = conn[i].order[j];
                Vec3 burg = order*segs[s].burg;
                double bMag = burg.norm();
                double invbMag = 1.0 / bMag;
                
                norm[j] = segs[s].plane.normalized();
                
                double dangle = invbMag * fabs(dot(burg, dr));
                double Mob = Medge+(Mscrew-Medge)*dangle;
                LtimesB += (L / Mob);
            }
            LtimesB /= 2.0;
            
            if (numNonZeroLenSegs > 0) {
                // Get glide constraints projection matrix
                Mat33 P = glide_constraints(nconn, norm);
                vi = P * (1.0/LtimesB * fi);
            }
        }
        
        return vi;
    }
};

void mobility(System* system)
{
    system->timer.reset();

    Kokkos::parallel_for(system->Nnodes, Mobility(system));
    Kokkos::fence();

    system->time_mobility += system->timer.seconds();
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

    std::string dirpath = exadis_root_dir + "/tests/test_perf/";
    std::vector<std::string> files;
    files.push_back("500.data");//0
    files.push_back("1000.data");//1
    files.push_back("2000.data");//2
    files.push_back("5000.data");//3
    files.push_back("10000.data");//4
    
    Kokkos::Timer timer;
    Kokkos::fence(); timer.reset();

    //std::string datafile = dirpath+files[3];
    std::string datafile = exadis_root_dir + "/examples/22_fcc_Cu_15um_1e3/180chains_16.10e.data";
    ExaDiS::SerialDisNet* network = ExaDiS::read_paradis(datafile.c_str());

    System* system = exadis_new<System>(network);

    //ForceN2* force0 = exadis_new<ForceN2>(system);
    
    double tinit = timer.seconds();
    Kokkos::fence();
    
    int nsteps = 100;
    timer.reset();
    for (int i = 0; i < nsteps; i++) {
        printf(" step %d / %d\n", i+1, nsteps);
        
        // Original kernel
        //force<Force::TagCompute1>(system);

        // Kernel with manual inlining
        //force<Force::TagCompute2>(system);

        // Original kernel with lauchbounds hint
        //force<Force::TagCompute1,true>(system);

        // Kernel with manual inlining and lauchbounds hint
        force<Force::TagCompute2,true>(system);

    }

    if (1) {
        std::string file = "force.dat";
        printf(" write_force: %s\n", file.c_str());
        FILE* fp = fopen(file.c_str(), "w");
        auto h_nodes = create_mirror_view(system->nodes);
        Kokkos::deep_copy(h_nodes, system->nodes);
        for (int i = 0; i < system->Nnodes; i++)
            fprintf(fp, "%e %e %e\n", h_nodes(i).f.x, h_nodes(i).f.y, h_nodes(i).f.z);
        fclose(fp);
    }

    printf("%20s %6.3f sec\n", "init time:", tinit);
    printf("%20s %6.3f sec\n", "force time:", system->time_force);
    //printf("%20s %6.3f sec\n", "mobility time:", system->time_mobility);
    printf("%20s %6.3f sec\n", "total time:", timer.seconds());
    
    exadis_delete(system);
    delete network;

    return 0;
}

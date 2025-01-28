#include <iostream>
#include "system.h"
#include "force.h"
#include "mobility.h"
#include "functions.h"

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

using namespace ExaDiS;

struct SegSegTest
{
    double MU, NU, a;
    struct Params {};
    SegSegTest(System *system, Params params) {
        MU = system->params.MU;
        NU = system->params.NU;
        a = system->params.a;
    }

    template<class N>
    KOKKOS_INLINE_FUNCTION
    SegSegForce segseg_force(System *system, N *net, const SegSeg &ss, 
                             int compute_seg12=1, int compute_seg34=1) 
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;
        
        int n1 = segs[ss.s1].n1;
        int n2 = segs[ss.s1].n2;
        Vec3 b1 = segs[ss.s1].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        double l1 = (r2-r1).norm2();
            
        int n3 = segs[ss.s2].n1;
        int n4 = segs[ss.s2].n2;
        Vec3 b2 = segs[ss.s2].burg;
        Vec3 r3 = cell.pbc_position(r1, nodes[n3].pos);
        Vec3 r4 = cell.pbc_position(r3, nodes[n4].pos);
        double l2 = (r4-r3).norm2();

        Vec3 f1(0.0), f2(0.0), f3(0.0), f4(0.0);
        if (l1 >= 1.e-20 && l2 >= 1.e-20) {
            f1 = f2 = f3 = f4 = MU*dot(b1, b2)*cross(r2-r1, r4-r3);
        }
        
        return SegSegForce(f1, f2, f3, f4);
    }
};
typedef ForceSeg<ForceSegN2<SegSegTest> > ForceN2Test;
typedef ForceSegSegList<SegSegTest,false> FORCE_SEGSEG_TEST;


template<int I>
class ForceTest : public Force {
public:
    CoreDefault* core;
    //double MU, NU, a;

    //T_nodes nodes;
    //T_segs segs;
    //T_conn conn;
    //Cell cell;
    //Mat33 extstress;

    //Kokkos::View<Vec3*[2]> fseg;
    Kokkos::View<Vec3*> fnode;

    ForceTest(System* system) {
        //MU = system->params.MU;
        //NU = system->params.NU;
        //a = system->params.a;
        CoreDefault::Params cparams;
        core = exadis_new<CoreDefault>(system, cparams);
    }

    template<class N>
    struct ComputeForce {
        System* system;
        ForceTest<I> force;
        N* net;
        ComputeForce(System* _system, ForceTest* _force, N* _net) : system(_system), force(*_force), net(_net) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const int &i) const {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto cell = net->cell;
            
            int n1 = segs[i].n1;
            int n2 = segs[i].n2;
            Vec3 b = segs[i].burg;
            Vec3 r1 = nodes[n1].pos;
            Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
            
            Vec3 t = r2-r1;
            double L = t.norm();
            if (L < 1e-10) return;
            t = t.normalized();

            // Core-force
            Vec3 fsf = force.core->core_force(b, t);
            
            // External PK force
            Vec3 fpk = pk_force(b, r1, r2, system->extstress);
            
            Vec3 f1 = +1.0*fsf + fpk;
            Vec3 f2 = -1.0*fsf + fpk;
            
            //Kokkos::atomic_add(&nodes[n1].f, f1);
            //Kokkos::atomic_add(&nodes[n2].f, f2);
            Kokkos::atomic_add(&force.fnode(n1), f1);
            Kokkos::atomic_add(&force.fnode(n2), f2);
        }
    };
    
    void compute(System* system, bool zero=true)
    {
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].start();

        DeviceDisNet* net = system->get_device_network();
        
        //extstress = system->extstress;
        if (zero) {
            //zero_force(net);
            Kokkos::resize(fnode, net->Nnodes_local);
            Kokkos::deep_copy(fnode, 0.0);
        }
        Kokkos::parallel_for(net->Nsegs_local, ComputeForce<DeviceDisNet>(system, this, net));
        
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].stop();
    }

    Vec3 node_force(System *system, const int &i) { return Vec3(0.0); }
};



int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

    Crystal crystal = Crystal(BCC_CRYSTAL);

#if 0
    std::string dirpath = exadis_root_dir + "/tests/test_perf/";
    //std::string datafile = dirpath + "500.data";
    //std::string datafile = dirpath + "1000.data";
    //std::string datafile = dirpath + "2000.data";
    std::string datafile = dirpath + "5000.data";
    //std::string datafile = dirpath + "10000.data";
    double maxseg = 15.0;
    double minseg = 3.0;
    SerialDisNet* config = read_paradis(datafile.c_str());
#else
    double Lbox = 300.0;
    double maxseg = 15.0;
    double minseg = 3.0;
    SerialDisNet* config = generate_prismatic_config(crystal, Lbox, 200, 0.2*Lbox, maxseg);
#endif
    
    double burgmag = 2.85e-10;
    double MU = 50.0e9;
    double NU = 0.3;
    double a = 1.0;
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    params.rtol = 0.3;
    params.nextdt = 5e-13;   
    
    System* system = exadis_new<System>(); 
    //System* system = new System(); 
    system->initialize(params, crystal, config);
    
    /*
    Force* force = new ForceCollection(system, {
        new ForceType::CORE_SELF_PKEXT(system),
        new ForceType::BRUTE_FORCE_N2(system)
        //new ForceN2Test(system)
        //new ForceType::LONG_FFT_SHORT_ISO(system, 128)
    });
    */
    //Force* force = exadis_new<ForceType::LINE_TENSION_MODEL>(system);
    

    //Force* force = new ForceType::CORE_SELF_PKEXT(system);
    Force* force = new ForceType::BRUTE_FORCE_N2(system);
    //Force* force = new ForceType::FORCE_SEGSEG_ISO(system, 37.5);
    //Force* force = new ForceFFT(system, 128);
    //Force* force = new ForceType::LONG_FFT_SHORT_ISO(system, 128);
    
    //Force* force = new ForceN2Test(system);
    //Force* force = new FORCE_SEGSEG_TEST(system, 37.5);
    //Force* force = exadis_new<ForceTest>(system);
    //Force* force = new ForceTest<0>(system);

    /*
    Force* force = exadis_new<ForceType::DDD_FFT_MODEL>(system,
        ForceType::CORE_SELF_PKEXT::Params(),
        ForceType::LONG_FFT_SHORT_ISO::Params(64)
    );
    */
    MobilityType::BCC_0B::Params mobparams(
        2600.0, //MobEdge
        20.0, //MobScrew
        1e-4, //MobClimb
        3400.0 //vmax
    );
    Mobility* mobility = new MobilityType::BCC_0B(system, mobparams);
    
    //Integrator* integrator = new IntegratorEuler(system);
    //Integrator* integrator = new IntegratorTrapezoid(system, force, mobility);
    //Integrator* integrator = new IntegratorMulti<IntegratorTrapezoid>(system, force, mobility, 10);
    
    //Collision* collision = new CollisionRetroactive(system);

    //Topology* topology = new TopologySerial(system, force, mobility);
    //Topology* topology = new TopologyParallel<ForceTypes,MobilityType::BCC_0B>(system, force, mobility);

    //Remesh* remesh = new RemeshSerial(system);
    
    Kokkos::Timer timer;
    timer.reset();
    
    system->params.check_params();
    printf("neighbor_cutoff = %e\n", system->neighbor_cutoff);
    
    int maxsteps = 100;
    for (int step = 0; step < maxsteps; step++) {

        #if 0
            DeviceDisNet* net = system->get_device_network();
            //NeighborBox* neighbox = exadis_new<NeighborBox>(system, net, 1.0, Neighbor::NeiSeg);
            //NeighborBoxSort* neighbox = exadis_new<NeighborBoxSort>(system, net, 1.0, Neighbor::NeiSeg);
            //exadis_delete(neighbox);
            
            //NeighborList* neilist = generate_neighbor_list(system, net, 1.0, Neighbor::NeiSeg);
            NeighborList* neilist = generate_neighbor_list_sort(system, net, 1.0, Neighbor::NeiNode);
            exadis_delete(neilist);
        #endif
        
        // Do some force pre-computation for the step if needed
        force->pre_compute(system);
        
        // Nodal force calculation
        force->compute(system);
        
        // Mobility calculation
        mobility->compute(system);
        
        // Time-integration
        //integrator->integrate(system);
        
        // Compute plastic strain
        //system->plastic_strain();
        
        // Reset glide planes
        //system->reset_glide_planes();
        
        // Collision
        //collision->handle(system);
        
        // Topology
        //topology->handle(system);
        
        // Remesh
        //remesh->remesh(system);
        
        // Update stress
        //update_mechanics(ctrl);
        
        if ((step+1) % MIN(maxsteps, 10) == 0) printf("Step = %6d: nodes = %d, dt = %e, elapsed = %.1f sec\n",
        step+1, system->Nnodes_total(), system->realdt, timer.seconds());
    }
    
    Kokkos::fence();
    double totaltime = timer.seconds();
    system->print_timers(1);
    ExaDiS_log("RUN TIME: %f sec\n", totaltime);

    if (1) {
        std::string file = "force0.dat";
        printf(" write_force: %s\n", file.c_str());
        auto net = system->get_serial_network();
        FILE* fp = fopen(file.c_str(), "w");
        for (int i = 0; i < net->number_of_nodes(); i++)
            fprintf(fp, "%e %e %e\n", net->nodes[i].f.x, net->nodes[i].f.y, net->nodes[i].f.z);
        fclose(fp);
    }
    
    //exadis_delete(system);
    //delete force;
    //exadis_delete(force);
    delete mobility;
    //delete integrator;
    //delete collision;
    //delete topology;
    //delete remesh;

    return 0;
}

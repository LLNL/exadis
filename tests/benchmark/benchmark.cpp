/*---------------------------------------------------------------------------
 *
 *  ExaDiS benchmark
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "system.h"
#include "functions.h"
#include "force.h"
#include "mobility.h"
#include "integrator.h"
#include "collision.h"
#include "topology.h"
#include "remesh.h"

using namespace ExaDiS;

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

/*---------------------------------------------------------------------------
 *
 *    Function:     make_benchmark_system
 *
 *-------------------------------------------------------------------------*/
System* make_benchmark_system(int system_id)
{
    SerialDisNet* config = nullptr;
    int crystal_type = -1;
    double burgmag, MU, NU, a, maxseg, minseg;
    double rtol, rann, nextdt;
    burgmag = MU = NU = a = maxseg = minseg = -1.0;
    rtol = rann = nextdt = -1.0;
    
    if (system_id == 0) {
        // BCC loops
        crystal_type = BCC_CRYSTAL;
        double Lbox = 300.0;
        maxseg = 15.0;
        minseg = 3.0;
        config = generate_prismatic_config(crystal_type, Lbox, 24, 0.21*Lbox, maxseg);
        
        burgmag = 2.85e-10;
        MU = 42.3e9;
        NU = 0.339;
        a = 1.0;
        rtol = 0.3;
        
    } else if (system_id == 1) {
        // FCC configuration
        std::string datafile = exadis_root_dir + "/examples/22_fcc_Cu_15um_1e3/180chains_16.10e.data";
        crystal_type = FCC_CRYSTAL;
        maxseg = 2000.0;
        minseg = 300.0;
        config = read_paradis(datafile.c_str());
        
        burgmag = 2.55e-10;
        MU = 54.6e9;
        NU = 0.324;
        a = 6.0;
        rtol = 10.0;
        rann = 10.0;
        nextdt = 1e-10;
        
    } else {
        ExaDiS_fatal("Error: Invalid system id = %d\n", system_id);
    }
    
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    params.rtol = rtol;
    if (rann > 0.0) params.rann = rann;
    if (nextdt > 0.0) params.nextdt = nextdt;
    
    return make_system(config, Crystal(crystal_type), params);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_force
 *
 *-------------------------------------------------------------------------*/
struct ForceBenchmark {
    System* system;
    Force* force;
    int Ncomp = 0;
    std::string label;
    ForceBenchmark(System* s, Force* f, int N, std::string l) :
    system(s), force(f), Ncomp(N), label(l) {}
};

void test_force(int Nmult)
{
    Kokkos::Timer timer;
    
    System* system0 = make_benchmark_system(0);
    System* system1 = make_benchmark_system(1);
    
    std::vector<ForceBenchmark> forces;
    for (int i = 0; i < 2; i++) {
        System* s = (i == 0) ? system0 : system1;
        forces.emplace_back(s, exadis_new<ForceType::LINE_TENSION_MODEL>(s), Nmult*100, "line tension model");
        forces.emplace_back(s, exadis_new<ForceType::BRUTE_FORCE_N2>(s), Nmult*((i==0)?10:1), "ForceN2");
        forces.emplace_back(s, exadis_new<ForceSegSegList<SegSegIso,false> >(s, 5000.0), Nmult*10, "ForceSegSegList");
        forces.emplace_back(s, exadis_new<ForceFFT>(s, ForceFFT::Params(64)), Nmult*10, "ForceFFT-64");
    }
    
    ExaDiS_log("-----------------------------------------------------------------\n");
    ExaDiS_log("test_force | Force type             Nsegs     Ncomp      Time\n");
    ExaDiS_log("-----------------------------------------------------------------\n");
    
    for (auto f : forces) {
        Kokkos::fence(); timer.reset();
        f.force->pre_compute(f.system);
        for (int i = 0; i < f.Ncomp; i++)
            f.force->compute(f.system);
        double t = timer.seconds();
        ExaDiS_log("test_force | %-18s %8d %8dx %10.4f sec\n", f.label.c_str(), f.system->Nsegs_total(), f.Ncomp, t);
    }
    
    ExaDiS_log("-----------------------------------------------------------------\n");
    
    for (auto f : forces)
        exadis_delete(f.force);
    
    exadis_delete(system0);
    exadis_delete(system1);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_integrator
 *
 *-------------------------------------------------------------------------*/
struct IntegratorBenchmark {
    System* system;
    Force* force;
    Mobility* mob;
    Integrator* igtr;
    int Nstep = 0;
    std::string label;
    IntegratorBenchmark(System* s, Force* f, Mobility* m, Integrator* i, int N, std::string l) :
    system(s), force(f), mob(m), igtr(i), Nstep(N), label(l) {}
};

void test_integrator(int Nmult)
{
    Kokkos::Timer timer;
    
    System* system0 = make_benchmark_system(0);
    System* system1 = make_benchmark_system(1);
    
    int Ngrid = 64;
    
    Force* force0 = exadis_new<ForceType::DDD_FFT_MODEL>(system0, 
        ForceType::CORE_SELF_PKEXT::Params(),
        ForceType::LONG_FFT_SHORT_ISO::Params(Ngrid)
    );
    Force* force1 = exadis_new<ForceType::DDD_FFT_MODEL>(system1, 
        ForceType::CORE_SELF_PKEXT::Params(),
        ForceType::LONG_FFT_SHORT_ISO::Params(Ngrid)
    );
    Force* forcesub0 = exadis_new<ForceType::SUBCYCLING_MODEL>(system0, ForceType::SUBCYCLING_MODEL::Params(Ngrid));
    Force* forcesub1 = exadis_new<ForceType::SUBCYCLING_MODEL>(system1, ForceType::SUBCYCLING_MODEL::Params(Ngrid));
    
    MobilityType::BCC_0B::Params mobparams0(2600.0, 20.0, 1e-4, 3400.0);
    Mobility* mobility0 = new MobilityType::BCC_0B(system0, mobparams0);
    MobilityType::FCC_0::Params mobparams1(64103.0, 64103.0, 4000.0);
    Mobility* mobility1 = new MobilityType::FCC_0(system1, mobparams1);
    
    std::vector<IntegratorBenchmark> integrator;
    integrator.emplace_back(system0, force0, mobility0, new IntegratorTrapezoid(system0, force0, mobility0), Nmult*20, "System0 trapezoid");
    integrator.emplace_back(system0, force0, mobility0, new IntegratorRKF(system0, force0, mobility0), Nmult*20, "System0 rkf");
    integrator.emplace_back(system0, forcesub0, mobility0, new IntegratorSubcycling(system0, forcesub0, mobility0), Nmult*10, "System0 subcycling");
    integrator.emplace_back(system1, force1, mobility1, new IntegratorTrapezoid(system1, force1, mobility1), Nmult*10, "System1 trapezoid");
    integrator.emplace_back(system1, force1, mobility1, new IntegratorRKF(system1, force1, mobility1), Nmult*10, "System1 rkf");
    integrator.emplace_back(system1, forcesub1, mobility1, new IntegratorSubcycling(system1, forcesub1, mobility1), Nmult*10, "System1 subcycling");
    
    ExaDiS_log("-----------------------------------------------------------------\n");
    ExaDiS_log("test_integrator | Integration type   Nsegs    Nstep      Time\n");
    ExaDiS_log("-----------------------------------------------------------------\n");
    
    for (auto itgr : integrator) {
        Kokkos::fence(); timer.reset();
        for (int i = 0; i < itgr.Nstep; i++) {
            itgr.force->pre_compute(itgr.system);
            itgr.force->compute(itgr.system);
            itgr.mob->compute(itgr.system);
            itgr.igtr->integrate(itgr.system);
            ExaDiS_log("   Step = %d / %d, Nsegs = %d, dt = %e, elapsed = %.2f sec\n",
            i+1, itgr.Nstep, itgr.system->Nsegs_total(), itgr.system->realdt, timer.seconds());
        }
        double t = timer.seconds();
        ExaDiS_log("test_integrator | %-14s %8d %7dx %10.4f sec\n", itgr.label.c_str(), itgr.system->Nsegs_total(), itgr.Nstep, t);
    }
    
    ExaDiS_log("-----------------------------------------------------------------\n");
    
    exadis_delete(system0);
    exadis_delete(system1);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_cycle
 *
 *-------------------------------------------------------------------------*/
void test_cycle(int Nmult, bool full)
{    
    {
        ExaDiS_log("-----------------------------------------------------------------\n");
        if (full) ExaDiS_log("test_cycle_full System0\n");
        else ExaDiS_log("test_cycle System0\n");
        ExaDiS_log("-----------------------------------------------------------------\n");
        
        int Nstep = Nmult*20;
        System* system = make_benchmark_system(1);
        system->params.nextdt = 1e-12;
        Force* force = exadis_new<ForceType::DDD_FFT_MODEL>(system, 
            ForceType::CORE_SELF_PKEXT::Params(),
            ForceType::LONG_FFT_SHORT_ISO::Params(64)
        );
        Mobility* mobility = new MobilityType::FCC_0(system, MobilityType::FCC_0::Params(64103.0, 64103.0, 4000.0));
        Integrator* integrator = new IntegratorTrapezoid(system, force, mobility);
        Collision* collision = new CollisionRetroactive(system);
        Topology* topology = new TopologyParallel<ForceType::DDD_FFT_MODEL,MobilityType::FCC_0>(system, force, mobility);
        Remesh* remesh = new RemeshSerial(system);
        
        system->params.check_params();

        Kokkos::Timer timer; timer.reset();
        
        for (int i = 0; i < 2; i++)
            remesh->remesh(system);
        system->timer[system->TIMER_REMESH].accumtime = 0.0;

        for (int i = 0; i < Nstep; i++) {
            force->pre_compute(system);
            force->compute(system);
            mobility->compute(system);
            integrator->integrate(system);
            if (full) {
                collision->handle(system);
                topology->handle(system);
                remesh->remesh(system);
            }
            ExaDiS_log("   Step = %d / %d, Nsegs = %d, dt = %e, elapsed = %.2f sec\n",
            i+1, Nstep, system->Nsegs_total(), system->realdt, timer.seconds());
        }

        system->print_timers();
        ExaDiS_log("System0 CYCLE (%dx) time: %f sec\n", Nstep, timer.seconds());
        
        delete remesh;
        delete topology;
        delete collision;
        delete integrator;
        delete mobility;
        exadis_delete(force);
        exadis_delete(system);
    }
    
    {
        ExaDiS_log("-----------------------------------------------------------------\n");
        if (full) ExaDiS_log("test_cycle_full System1\n");
        else ExaDiS_log("test_cycle System1\n");
        ExaDiS_log("-----------------------------------------------------------------\n");
        
        int Nstep = Nmult*10;
        System* system = make_benchmark_system(1);
        Force* force = exadis_new<ForceType::SUBCYCLING_MODEL>(system, ForceType::SUBCYCLING_MODEL::Params(64));
        Mobility* mobility = new MobilityType::FCC_0(system, MobilityType::FCC_0::Params(64103.0, 64103.0, 4000.0));
        Integrator* integrator = new IntegratorSubcycling(system, force, mobility,
            IntegratorSubcycling::Params({0.0, 100.0, 600.0, 1600.0})
        );
        Collision* collision = new CollisionRetroactive(system);
        Topology* topology = new TopologyParallel<ForceType::SUBCYCLING_MODEL,MobilityType::FCC_0>(system, force, mobility);
        Remesh* remesh = new RemeshSerial(system);
        
        system->params.check_params();

        Kokkos::Timer timer; timer.reset();
        
        for (int i = 0; i < 2; i++)
            remesh->remesh(system);
        system->timer[system->TIMER_REMESH].accumtime = 0.0;

        for (int i = 0; i < Nstep; i++) {
            force->pre_compute(system);
            force->compute(system);
            mobility->compute(system);
            integrator->integrate(system);
            if (full) {
                collision->handle(system);
                topology->handle(system);
                remesh->remesh(system);
            }
            ExaDiS_log("   Step = %d / %d, Nsegs = %d, dt = %e, elapsed = %.2f sec\n",
            i+1, Nstep, system->Nsegs_total(), system->realdt, timer.seconds());
        }

        system->print_timers();
        ExaDiS_log("System1 SUBCYCLING CYCLE (%dx) time: %f sec\n", Nstep, timer.seconds());
        
        delete remesh;
        delete topology;
        delete collision;
        delete integrator;
        delete mobility;
        exadis_delete(force);
        exadis_delete(system);
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    
    int Nmult = 1;
    bool force = true;
    bool integrator = true;
    bool cycle = true;
    bool cycle_full = true;
    
    if (argc > 1) {
        Nmult = MAX(atoi(argv[1]), 1);
    }
    if (argc > 2) {
        force = integrator = cycle = cycle_full = false;
        for (int i = 2; i < argc; i++) {
            std::string arg = std::string(argv[i]);
            if (arg == "-all") { force = integrator = cycle = cycle_full = true; }
            else if (arg == "-force") { force = true; }
            else if (arg == "-integrator") { integrator = true; }
            else if (arg == "-cycle") { cycle = true; }
            else if (arg == "-cycle_full") { cycle_full = true; }
        }
    }
    
    if (force) test_force(Nmult);
    if (integrator) test_integrator(Nmult);
    if (cycle) test_cycle(Nmult, false);
    if (cycle_full) test_cycle(Nmult, true);
    
    Kokkos::finalize();
}

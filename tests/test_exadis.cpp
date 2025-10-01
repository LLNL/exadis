/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include <iostream>

#include "exadis.h"
#include "driver.h"

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

using namespace ExaDiS;

/*---------------------------------------------------------------------------
 *
 *    Function:     tests
 *
 *-------------------------------------------------------------------------*/
void tests(ExaDiSApp *exadis, int test_id)
{
    System *system = exadis->system;
    
    SerialDisNet* config = nullptr;
    Crystal crystal;
    double burgmag, MU, NU, a;
    double maxseg = -1.0, minseg = -1.0;
    double rtol = 1.0;
    double rann = -1.0;
    double nextdt = 5e-13;
    double maxdt = -1.0;
    double Ecore_junc_fact = 1.0;
    std::vector<ExaDiSApp::Control> ctrls;
    std::string outputdir = "output";
    
    if (test_id == 0) {
    
        // Prismatic loop test case
        crystal = Crystal(BCC_CRYSTAL);
        double Lbox = 300.0;
        maxseg = 0.1 * Lbox;
        minseg = 0.2 * maxseg;
        double radius = 0.2 * Lbox;
        config = generate_prismatic_config(crystal, Lbox, 1, radius, maxseg);
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 1000;
        ctrl.loading = ExaDiSApp::STRESS_CONTROL;
        ctrl.appstress = Mat33().symmetric(0.0, 0.0, 2e9, 0.0, 0.0, 0.0);
        ctrl.printfreq = 10;
        ctrl.outfreq = 100;
        ctrls.push_back(ctrl);
        
        ctrl.appstress = Mat33().zero();
        //ctrls.push_back(ctrl);
        
    } else if (test_id == 1) {
        
        // Collision test case
        double Lbox = 300.0;
        maxseg = 0.1 * Lbox;
        minseg = 0.2 * maxseg;
        config = new SerialDisNet(Lbox);
        Mat33 R = Mat33().eye();
        
        Vec3 burg(1.0, 0.0, 0.0);
        Vec3 plane(0.0, 1.0, 0.0);
        double thetadeg = -90.0;
        double L = 0.5 * Lbox;
        Vec3 center = config->cell.center();
        insert_frs(config, burg, plane, thetadeg, L, center, R);
        
        burg = Vec3(2.0, 0.0, 0.0); // junction
        plane = Vec3(0.0, 0.0, 1.0);
        center += Vec3(0.2*Lbox, 0.0, 0.0);
        insert_frs(config, burg, plane, thetadeg, L, center, R);
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 200;
        ctrl.loading = ExaDiSApp::STRESS_CONTROL;
        ctrl.appstress = Mat33().symmetric(0.0, 0.0, 0.0, 5e8, 0.0, 0.0);
        ctrl.printfreq = 1;//10;
        ctrl.outfreq = 10;
        ctrls.push_back(ctrl);
        
    } else if (test_id == 2) {
        
        // Binary junction test case
        crystal = Crystal(BCC_CRYSTAL);
        double Lbox = 100.0;
        maxseg = 0.1 * Lbox;
        minseg = 0.2 * maxseg;
        config = new SerialDisNet(Lbox);
        Mat33 R = Mat33().eye();
        Vec3 center = config->cell.center();
        double L = 1.0 * Lbox;
        
        Vec3 b1 = Vec3(-1.0, 1.0, 1.0).normalized();
        Vec3 n1 = Vec3(1.0, -1.0, 0.0).normalized();
        double phi1 = 20.0; // degrees
        
        Vec3 b2 = Vec3(1.0, -1.0, 1.0).normalized();
        Vec3 n2 = Vec3(0.0, 1.0, 1.0).normalized();
        double phi2 = 20.0; // degrees
        
        Vec3 linter = cross(n1, n2).normalized();
        Vec3 y1 = cross(linter, n1).normalized();
        Vec3 ldir1 = cos(phi1*M_PI/180.0)*linter+sin(phi1*M_PI/180.0)*y1;
        Vec3 y2 = cross(linter, n2).normalized();
        Vec3 ldir2 = cos(phi2*M_PI/180.0)*linter+sin(phi2*M_PI/180.0)*y2;
        
        Vec3 delta = 1.0 * Vec3(1.0, 1.0, 0.0);
        
        insert_frs(config, b1, n1, ldir1, L, center+delta, R);
        insert_frs(config, b2, n2, ldir2, L, center-delta, R);
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 47;
        ctrl.loading = ExaDiSApp::STRESS_CONTROL;
        ctrl.appstress = Mat33().zero();
        ctrl.printfreq = 1;
        ctrl.outfreq = 1;
        ctrls.push_back(ctrl);
        
    } else if (test_id == 3) {
        
        // MD-like BCC simulation
        crystal = Crystal(BCC_CRYSTAL);
        double Lbox = 300.0;
        maxseg = 0.1 * Lbox;
        minseg = 0.2 * maxseg;
        double radius = 0.2 * Lbox;
        config = generate_prismatic_config(crystal, Lbox, 12, radius, maxseg);
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 100;//1000;
        //ctrl.nsteps = ExaDiSApp::NUM_STEPS(1000);
        //ctrl.nsteps = ExaDiSApp::MAX_STRAIN(0.1);
        ctrl.loading = ExaDiSApp::STRAIN_RATE_CONTROL;
        ctrl.erate = 2e8;
        ctrl.appstress = Mat33().zero();
        ctrl.printfreq = 1;
        ctrl.propfreq = 10;
        using Prop = ExaDiSApp::Prop;
        ctrl.props = {Prop::STEP, Prop::STRAIN, Prop::STRESS, Prop::DENSITY, Prop::WALLTIME};
        ctrl.outfreq = 10;
        ctrls.push_back(ctrl);
        
    } else if (test_id == 4) {
        
        // Nodal x-slip
        crystal = Crystal(BCC_CRYSTAL);
        maxseg = 10.0;
        minseg = 2.0;
        std::string datafile = exadis_root_dir + "/examples/06_nodal_xslip/ta-elemental-single.data";
        config = read_paradis(datafile.c_str());
        
        rtol = 0.3;
        nextdt = 1e-13;
        Ecore_junc_fact = 0.2;
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 500;
        ctrl.loading = ExaDiSApp::STRAIN_RATE_CONTROL;
        ctrl.erate = -2e8;
        ctrl.appstress = Mat33().zero();
        ctrl.printfreq = 1;
        ctrl.outfreq = 10;
        ctrls.push_back(ctrl);
        
    } else if (test_id == 5) {
        
        // Dislocation intersection in BCC
        crystal = Crystal(BCC_CRYSTAL);
        double Lbox = 300.0;
        Vec3 bmin = -0.5*Vec3(Lbox);
        Vec3 bmax = +0.5*Vec3(Lbox);
        config = new SerialDisNet(Cell(bmin, bmax));
        Mat33 R = Mat33().eye();
        Vec3 center = config->cell.center();
        double L = 0.7 * Lbox;
        maxseg = 0.1 * Lbox;
        minseg = 0.2 * maxseg;
        
        Vec3 b1 = Vec3(-1.0, 1.0, 1.0).normalized();
        Vec3 n1 = Vec3(1.0, -1.0, 0.0).normalized();
        double phi1 = 90.0; // degrees
        
        Vec3 b2 = Vec3(1.0, -1.0, 1.0).normalized();
        Vec3 n2 = Vec3(0.0, 1.0, 1.0).normalized();
        double phi2 = -90.0; // degrees
        
        Vec3 linter = cross(n1, n2).normalized();
        Vec3 y1 = cross(linter, n1).normalized();
        Vec3 ldir1 = cos(phi1*M_PI/180.0)*linter+sin(phi1*M_PI/180.0)*y1;
        Vec3 y2 = cross(linter, n2).normalized();
        Vec3 ldir2 = cos(phi2*M_PI/180.0)*linter+sin(phi2*M_PI/180.0)*y2;
        
        Vec3 delta = 0.05 * L * Vec3(1.0, 1.0, 0.0);
        
        insert_frs(config, b1, n1, ldir1, L, center+delta, R);
        insert_frs(config, b2, n2, ldir2, L, center-delta, R);
        
        maxdt = 5e-13;
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 200;
        ctrl.loading = ExaDiSApp::STRESS_CONTROL;
        ctrl.appstress = Mat33().symmetric(0.0, 0.0, 2e9, 0.0, 0.0, 0.0);
        ctrl.printfreq = 1;
        ctrl.outfreq = 1;
        ctrls.push_back(ctrl);
        
    } else if (test_id == 6) {
        
        // MD-like FCC simulation
        crystal = Crystal(FCC_CRYSTAL);
        double Lbox = 300.0;
        maxseg = 0.1 * Lbox;
        minseg = 0.2 * maxseg;
        double radius = 0.2 * Lbox;
        config = generate_prismatic_config(crystal, Lbox, 1, radius, maxseg);
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 1000;
        ctrl.loading = ExaDiSApp::STRAIN_RATE_CONTROL;
        //ctrl.loading = ExaDiSApp::STRESS_CONTROL;
        ctrl.erate = 2e8;
        ctrl.edir = Vec3(1.0,0.0,0.0);
        ctrl.appstress = Mat33().zero();
        ctrl.printfreq = 1;
        ctrl.propfreq = 10;
        using Prop = ExaDiSApp::Prop;
        ctrl.props = {Prop::STEP, Prop::STRAIN, Prop::STRESS, Prop::DENSITY, Prop::WALLTIME};
        ctrl.outfreq = 10;
        ctrls.push_back(ctrl);
        
    } else if (test_id == 7) {
        
        // Cu 15um
        crystal = Crystal(FCC_CRYSTAL);
        
        maxseg = 2000.0;
        minseg = 300.0;
        //std::string datafile = exadis_root_dir + "/tests/data/180chains_16.10e.data";
        //config = read_paradis(datafile.c_str());
        
        double Lbox = 58824.0;
        double radius = 0.3 * Lbox;
        config = generate_prismatic_config(crystal, Lbox, 12, radius, maxseg);
        
        nextdt = 5e-10;
        maxdt = 1e-9;
        rtol = 10.0;
        rann = 10.0;
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 100;
        ctrl.loading = ExaDiSApp::STRAIN_RATE_CONTROL;
        ctrl.erate = 1e3;
        ctrl.appstress = Mat33().zero();
        ctrl.printfreq = 1;
        ctrl.outfreq = 1;
        ctrls.push_back(ctrl);
        
    } else if (test_id == 8) {
        
        // Dislocation intersections/junctions in FCC
        crystal = Crystal(FCC_CRYSTAL);
        double Lbox = 10000.0;
        //Vec3 bmin = -0.5*Vec3(Lbox);
        //Vec3 bmax = +0.5*Vec3(Lbox);
        //config = new SerialDisNet(Cell(bmin, bmax));
        config = new SerialDisNet(Lbox);
        Mat33 R = Mat33().eye();
        Vec3 center = config->cell.center();
        double L = 0.7 * Lbox;
        maxseg = 0.1 * Lbox;
        minseg = 0.2 * maxseg;
        
        Vec3 b1 = Vec3(0.0, 1.0, 1.0).normalized();
        Vec3 n1 = Vec3(1.0, 1.0,-1.0).normalized();
        double phi1 = 30.0; // degrees
        
        int junction = 3;
        Vec3 b2, n2;
        if (junction == 0) { // Lomer
            b2 = Vec3(1.0, 0.0,-1.0).normalized();
            n2 = Vec3(1.0, 1.0, 1.0).normalized();
        } else if (junction == 1) { // Glissile
            b2 = Vec3(1.0, 0.0,-1.0).normalized();
            n2 = Vec3(1.0,-1.0, 1.0).normalized();
        } else if (junction == 2) { // Hirth
            b2 = Vec3(0.0,-1.0, 1.0).normalized();
            n2 = Vec3(1.0, 1.0, 1.0).normalized();
        } else if (junction == 3) { // Colinear
            b2 = Vec3(0.0,-1.0,-1.0).normalized();
            n2 = Vec3(1.0,-1.0, 1.0).normalized();
        }
        double phi2 = 30.0; // degrees
        
        Vec3 linter = cross(n1, n2).normalized();
        Vec3 y1 = cross(linter, n1).normalized();
        Vec3 ldir1 = cos(phi1*M_PI/180.0)*linter+sin(phi1*M_PI/180.0)*y1;
        Vec3 y2 = cross(linter, n2).normalized();
        Vec3 ldir2 = cos(phi2*M_PI/180.0)*linter+sin(phi2*M_PI/180.0)*y2;
        
        Vec3 delta = 0.01 * L * Vec3(1.0, 1.0, 0.0);
        
        insert_frs(config, b1, n1, ldir1, L, center+delta, R);
        insert_frs(config, b2, n2, ldir2, L, center-delta, R);
        
        rtol = 5.0;
        nextdt = 1e-12;
        maxdt = 1e-10;
        
        ExaDiSApp::Control ctrl;
        ctrl.nsteps = 100;
        ctrl.loading = ExaDiSApp::STRESS_CONTROL;
        ctrl.appstress = Mat33().zero();//Mat33().symmetric(0.0, 0.0, 2e9, 0.0, 0.0, 0.0);
        ctrl.printfreq = 1;
        ctrl.outfreq = 1;
        ctrls.push_back(ctrl);
    
    } else {
        ExaDiS_fatal("Error: invalid test_id = %d\n", test_id);
    }
    
    if (crystal.type == BCC_CRYSTAL) {
        burgmag = 2.85e-10;
        MU = 55.0e9;
        NU = 0.339;
        a = 1.0;
    } else {
        burgmag = 2.55e-10;
        MU = 54.6e9;
        NU = 0.324;
        a = 6.0;
    }
    
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    params.nextdt = nextdt;
    if (maxdt > 0.0) params.maxdt = maxdt;
    params.rtol = rtol;
    if (rann > 0.0) params.rann = rann;
    
    system->initialize(params, crystal, config);
    
    // Core model
    CoreDefault::Params CoreDefaultParams(-1.0, Ecore_junc_fact);
    CoreMD::Params CoreMD_Ta_Li03( // CoreMD fit for Ta_Li03 potential
        1.0, // rc used in the fit
        4, {-1.47709126, 2.87723871, -2.05414287, -0.0548932, 1.01028565}, // b<111>
        4, {-1.97070195, 3.78687972, -1.59401082, 0.08581447, 0.97687589} // b<100>
    );
    
    // Subcycling
    bool subcycling = 0;
    
    // Force
    if (!subcycling) {
        //exadis->force = exadis_new<ForceType::LINE_TENSION_MODEL>(system);
        exadis->force = new ForceCollection(system, {
            //exadis_new<ForceType::COREMD_SELF_PKEXT>(system, CoreMD_Ta_Li03),
            //exadis_new<ForceType::CORE_SELF_PKEXT>(system),
            exadis_new<ForceType::CORE_SELF_PKEXT>(system, CoreDefaultParams),
            //exadis_new<ForceType::BRUTE_FORCE_N2>(system)
            //exadis_new<ForceFFT>(system, ForceFFT::Params(64))
            //exadis_new<ForceSegSegList<SegSegIso>>(system, /*50.0*/ 1000.0)
            exadis_new<ForceType::LONG_FFT_SHORT_ISO>(system, 32)
        });
    } else {
        exadis->force = exadis_new<ForceType::SUBCYCLING_MODEL>(system, ForceType::SUBCYCLING_MODEL::Params(64));
    }
    
    // Mobility
    if (crystal.type == BCC_CRYSTAL) {
        MobilityType::BCC_0B::Params mobparams(
            2600.0, //MobEdge
            20.0, //MobScrew
            1e-4, //MobClimb
            3400.0 //vmax
        );
        exadis->mobility = new MobilityType::BCC_0B(system, mobparams);
    } else if (crystal.type == FCC_CRYSTAL) {
        MobilityType::FCC_0::Params mobparams(
            64103.0, //MobEdge
            64103.0, //MobScrew
            4000.0 //vmax
        );
        exadis->mobility = new MobilityType::FCC_0(system, mobparams);
    } else {
        MobilityType::GLIDE::Params mobparams(1000.0);
        exadis->mobility = new MobilityType::GLIDE(system, mobparams);
    }
    
    // Intergrator
    if (!subcycling) {
        exadis->integrator = new IntegratorTrapezoid(system, exadis->force, exadis->mobility);
        //exadis->integrator = new IntegratorRKF(system, exadis->force, exadis->mobility);
        //exadis->integrator = new IntegratorMulti<IntegratorTrapezoid>(system, exadis->force, exadis->mobility, 10);
    } else {
        exadis->integrator = new IntegratorSubcycling(system, exadis->force, exadis->mobility);
    }
    
    // Collision
    //exadis->collision = new Collision(system);
    exadis->collision = new CollisionRetroactive(system);
    
    // Topology
    if (!subcycling) {
        //exadis->topology = new Topology(system);
        exadis->topology = new TopologySerial(system, exadis->force, exadis->mobility);
    } else {
        exadis->topology = new TopologyParallel<ForceType::SUBCYCLING_MODEL,MobilityType::FCC_0>(system, exadis->force, exadis->mobility);
    }
    
    // Remesh
    //exadis->remesh = new Remesh(system);
    exadis->remesh = new RemeshSerial(system);
    
    // Output
    exadis->outputdir = outputdir;
    
    // Simulation setup
    exadis->set_simulation();
    
    
    // Simulation
    for (auto ctrl : ctrls)
        exadis->run(ctrl);
        
    if (subcycling) {
        exadis_delete(exadis->force);
        exadis->force = nullptr;
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
#ifdef MPI
    MPI_Init(&argc, &argv);
#endif
    Kokkos::ScopeGuard guard(argc, argv);
    
    int test_id = 3;
    if (argc >= 2) test_id = atoi(argv[1]);

    ExaDiS::ExaDiSApp exadis(argc, argv);
    tests(&exadis, test_id);

#ifdef MPI
    MPI_Finalize();
#endif
    return 0;
}

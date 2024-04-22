/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Author: Nicolas Bertin
 *	bertin1@llnl.gov
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
 *    Function:     example_fcc_Cu_15um_1e3
 *                  Example of a 15um simulation of fcc Cu loaded at a
 *                  strain rate of 1e3/s using the subcycling integrator.
 *                  E.g. see Bertin et al., MSMSE 27 (7), 075014 (2019)
 *
 *-------------------------------------------------------------------------*/
void example_fcc_Cu_15um_1e3(ExaDiSApp* exadis)
{
    // Simulation parameters
    Crystal crystal(FCC_CRYSTAL);
    double burgmag = 2.55e-10;
    double MU = 54.6e9;
    double NU = 0.324;
    double a = 6.0;

    double maxseg = 2000.0;
    double minseg = 300.0;
    double rtol = 10.0;
    double rann = 10.0;
    double nextdt = 1e-10;
    double maxdt = 1e-9;
    
    MobilityType::FCC_0::Params mobparams(
        64103.0, // MobEdge
        64103.0, // MobScrew
        4000.0 // vmax (m/s)
    );

    // Loading and options
    ExaDiSApp::Control ctrl;
    ctrl.nsteps = ExaDiSApp::MAX_STRAIN(0.01); // 1% strain
    ctrl.loading = ExaDiSApp::STRAIN_RATE_CONTROL;
    ctrl.erate = 1e3;
    ctrl.appstress = Mat33().zero();
    ctrl.printfreq = 1;
    ctrl.propfreq = 10;
    using Prop = ExaDiSApp::Prop;
    ctrl.props = {Prop::STEP, Prop::STRAIN, Prop::STRESS, Prop::DENSITY, Prop::NSEGS, Prop::WALLTIME};
    ctrl.outfreq = 100;
    std::string outputdir = "output_fcc_Cu_15um_1e3";
    
    
    // Initialization
    std::string datafile = exadis_root_dir + "/examples/22_fcc_Cu_15um_1e3/180chains_16.10e.data";
    SerialDisNet* config = read_paradis(datafile.c_str());
    
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    params.nextdt = nextdt;
    params.maxdt = maxdt;
    params.rtol = rtol;
    params.rann = rann;
    
    System* system = exadis->system;
    system->initialize(params, crystal, config);
    
    // Modules
    exadis->force = exadis_new<ForceType::SUBCYCLING_MODEL>(system, ForceType::SUBCYCLING_MODEL::Params(64));
    exadis->mobility = new MobilityType::FCC_0(system, mobparams);
    exadis->integrator = new IntegratorSubcycling(system, exadis->force, exadis->mobility,
        IntegratorSubcycling::Params({0.0, 100.0, 600.0, 1600.0})
    );
    exadis->collision = new CollisionRetroactive(system);
    exadis->topology = new TopologyParallel<ForceType::SUBCYCLING_MODEL,MobilityType::FCC_0>(system, exadis->force, exadis->mobility);
    exadis->remesh = new RemeshSerial(system);

    // Simulation setup
    exadis->outputdir = outputdir;
    exadis->set_simulation();
    
    // Run simulation
    exadis->run(ctrl);
    
    // Safely free force
    exadis_delete(exadis->force);
    exadis->force = nullptr;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard guard(argc, argv);
    
    ExaDiS::ExaDiSApp exadis(argc, argv);
    example_fcc_Cu_15um_1e3(&exadis);

    return 0;
}

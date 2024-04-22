/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Author: Nicolas Bertin
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
 *    Function:     example_bcc_Ta_100nm_2e8
 *                  Example of a 100nm MD-like simulation of bcc Ta loaded
 *                  at a strain rate of 2e8/s.
 *                  E.g. see Bertin et al., arXiv preprint arXiv:2210.14343
 *
 *-------------------------------------------------------------------------*/
void example_bcc_Ta_100nm_2e8(ExaDiSApp* exadis)
{
    // Simulation parameters
    Crystal crystal(BCC_CRYSTAL);
    double burgmag = 2.85e-10;
    double MU = 55.0e9;
    double NU = 0.339;
    double a = 1.0;

    double Lbox = 300.0;
    double maxseg = 15.0;
    double minseg = 3.0;
    double rtol = 0.3;
    double nextdt = 5e-13;
    
    MobilityType::BCC_0B::Params mobparams(
        2600.0, // MobEdge
        20.0, // MobScrew
        1e-4, // MobClimb
        3400.0 // vmax (m/s)
    );

    // Loading and options
    ExaDiSApp::Control ctrl;
    ctrl.nsteps = ExaDiSApp::MAX_STRAIN(1.0);
    ctrl.loading = ExaDiSApp::STRAIN_RATE_CONTROL;
    ctrl.erate = 2e8;
    ctrl.appstress = Mat33().zero();
    ctrl.printfreq = 1;
    ctrl.propfreq = 10;
    using Prop = ExaDiSApp::Prop;
    ctrl.props = {Prop::STEP, Prop::STRAIN, Prop::STRESS, Prop::DENSITY, Prop::NSEGS, Prop::WALLTIME};
    ctrl.outfreq = 100;
    std::string outputdir = "output_bcc_Ta_100nm_2e8";
    
    
    // Initialization
    SerialDisNet* config = generate_prismatic_config(crystal, Lbox, 12, 0.21*Lbox, maxseg);
    
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    params.nextdt = nextdt;
    params.rtol = rtol;
    
    System* system = exadis->system;
    system->initialize(params, crystal, config);
    
    // Modules
    exadis->force = exadis_new<ForceType::DDD_FFT_MODEL>(system, 
        ForceType::CORE_SELF_PKEXT::Params(),
        ForceType::LONG_FFT_SHORT_ISO::Params(64)
    );
    exadis->mobility = new MobilityType::BCC_0B(system, mobparams);
    exadis->integrator = new IntegratorMulti<IntegratorTrapezoid>(system, exadis->force, exadis->mobility, 10);
    exadis->collision = new CollisionRetroactive(system);
    exadis->topology = new TopologyParallel<ForceType::DDD_FFT_MODEL,MobilityType::BCC_0B>(system, exadis->force, exadis->mobility);
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
    example_bcc_Ta_100nm_2e8(&exadis);

    return 0;
}

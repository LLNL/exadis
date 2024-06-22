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

using namespace ExaDiS;

/*---------------------------------------------------------------------------
 *
 *    Function:     init_frank_read_src_loop
 *
 *-------------------------------------------------------------------------*/
SerialDisNet* init_frank_read_src_loop(double Lbox, Vec3& b, double Lsource, bool pbc=false)
{
    SerialDisNet* network = new SerialDisNet(Lbox);
    network->cell.xpbc = network->cell.ypbc = network->cell.zpbc = pbc;
    Vec3 c = network->cell.center();
    
    std::vector<DisNode> nodes = {
        DisNode(c+Vec3(0.0, -Lsource/2.0, 0.0),      PINNED_NODE),
        DisNode(c+Vec3(0.0,  0.0,         0.0),      UNCONSTRAINED),
        DisNode(c+Vec3(0.0,  Lsource/2.0, 0.0),      PINNED_NODE),
        DisNode(c+Vec3(0.0,  Lsource/2.0, -Lsource), PINNED_NODE),
        DisNode(c+Vec3(0.0, -Lsource/2.0, -Lsource), PINNED_NODE)
    };
    
    int N = nodes.size();
    for (int i = 0; i < N; i++) {
        Vec3 p = cross(b, nodes[(i+1)%N].pos-nodes[i].pos).normalized();
        network->add_node(nodes[i].pos, nodes[i].constraint);
        network->add_seg(i, (i+1)%N, b, p);
    }
    
    return network;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_frank_read_src
 *
 *-------------------------------------------------------------------------*/
void test_frank_read_src(ExaDiSApp* exadis)
{
    // Simulation parameters
    double burgmag = 3e-10;
    double MU = 50e9;
    double NU = 0.3;
    double a = 1.0;
    double Mob = 1.0;
    Mat33 applied_stress = Mat33().symmetric(0.0, 0.0, 0.0, 0.0, -4.0e8, 0.0);
    
    double Lbox = 1000.0;
    double maxseg = 0.04*Lbox;
    double minseg = 0.01*Lbox;
    double dt = 1.0e-8;
    double rann = 2.0;
    
    ExaDiSApp::Control ctrl;
    ctrl.nsteps = 200;
    ctrl.loading = ExaDiSApp::STRESS_CONTROL;
    ctrl.appstress = applied_stress;
    ctrl.printfreq = 10;
    ctrl.outfreq = 10;
    std::string outputdir = "output_test_frank_read_src";
    
    // Initialization
    Vec3 b = Vec3(1.0, 0.0, 0.0);
    double Lsource = 0.125*Lbox;
    bool pbc = 0;
    SerialDisNet* config = init_frank_read_src_loop(Lbox, b, Lsource, pbc);
    
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    params.nextdt = dt;
    params.rann = rann;
    
    System* system = exadis->system;
    system->initialize(params, Crystal(), config);
    
    // Modules
    exadis->force = new ForceType::LINE_TENSION_MODEL(system);
    exadis->mobility = new MobilityType::GLIDE(system, MobilityType::GLIDE::Params(Mob));
    exadis->integrator = new IntegratorEuler(system);
    exadis->collision = new CollisionRetroactive(system);
    exadis->topology = new TopologySerial(system, exadis->force, exadis->mobility);
    exadis->remesh = new RemeshSerial(system);

    // Simulation setup
    exadis->outputdir = outputdir;
    exadis->set_simulation();
    
    // Simulation
    exadis->run(ctrl);
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
    test_frank_read_src(&exadis);

    return 0;
}

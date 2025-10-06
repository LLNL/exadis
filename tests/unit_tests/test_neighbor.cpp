#include <iostream>
#include "system.h"
#include "force.h"
#include "mobility.h"
#include "integrator.h"
#include "functions.h"
#include "../debug.h"

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

using namespace ExaDiS;

System* get_system()
{
    // Simulation parameters
    int crystal_type = FCC_CRYSTAL;
    Crystal crystal(crystal_type);
    
    double maxseg = 2000.0;
    double minseg = 300.0;
    std::string datafile = exadis_root_dir + "/examples/22_fcc_Cu_15um_1e3/180chains_16.10e.data";
    SerialDisNet* config = read_paradis(datafile.c_str(), false);
    
    double burgmag = 2.55e-10;
    double MU = 54.6e9;
    double NU = 0.324;
    double a = 6.0;
    Params params(burgmag, MU, NU, a, maxseg, minseg);
    
    return make_system(config, crystal, params);
}

void test_neighborlist()
{
    System* system = get_system();
    DeviceDisNet* net = system->get_device_network();
    
    NeighborList* neilist = exadis_new<NeighborList>();
    
    std::vector<double> cutoffs = {100.0, 1000.0, 2000.0, 5000.0, 7500.0, 15000.0};
    for (double cutoff : cutoffs) {
        generate_neighbor_list(system, net, neilist, cutoff, Neighbor::NeiSeg);
        printf("%d\n", neilist->Ntotnei);
    }
    
    exadis_delete(neilist);
    exadis_delete(system);
}

void test_segseglist()
{
    System* system = get_system();
    DeviceDisNet* net = system->get_device_network();
    
    SegSegList* segseglist = exadis_new<SegSegList>(system, 0.0, false);
    
    std::vector<double> cutoffs = {100.0, 1000.0, 2000.0, 5000.0, 7500.0, 15000.0};
    for (double cutoff : cutoffs) {
        segseglist->set_cutoff(system, cutoff);
        segseglist->build_list<DeviceDisNet>(system, net);
        printf("%d\n", segseglist->Nsegseg);
    }
    
    exadis_delete(segseglist);
    exadis_delete(system);
}

void test_subgroups()
{
    System* system = get_system();
    system->params.rtol = 1.0;
    system->params.nextdt = 1e-12;
    
    Force* force = exadis_new<ForceType::SUBCYCLING_MODEL>(system, ForceType::SUBCYCLING_MODEL::Params(64));
    Mobility* mobility = new MobilityType::FCC_0(system, MobilityType::FCC_0::Params(64103.0, 64103.0, 4000.0));
    IntegratorSubcycling* integrator = new IntegratorSubcycling(system, force, mobility,
        IntegratorSubcycling::Params({0.0, 100.0, 600.0, 1600.0})
    );
    
    force->pre_compute(system);
    integrator->integrate(system);
    
    double cutoff = static_cast<ForceType::SUBCYCLING_MODEL*>(force)->fsegseg->get_cutoff();
    SegSegGroups* subgroups = integrator->get_subgroups();
    printf("%d\n", subgroups->Nsegseg_tot);
    for (int i = 0; i < subgroups->Ngroups; i++)
        printf("%d\n", subgroups->Nsegseg[i]);
    
    delete integrator;
    delete mobility;
    exadis_delete(force);
    exadis_delete(system);
}

int main(int argc, char* argv[])
{
    ExaDiS::Initialize init(argc, argv);
    
    std::string name = "";
    if (argc == 2) name = std::string(argv[1]);
    
    if (name == "test_neighborlist")
        test_neighborlist();
    else if (name == "test_segseglist")
        test_segseglist();
    else if (name == "test_subgroups")
        test_subgroups();
    else
        std::cerr << "Error: invalid test name = '" << name << "'" << std::endl;
    
    return 0;
}

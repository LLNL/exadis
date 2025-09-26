#include <iostream>
#include "system.h"
#include "force.h"
#include "functions.h"
#include "../debug.h"

#define STR(str) #str
#define STRING(str) STR(str)
std::string exadis_root_dir = STRING(EXADIS_ROOT_DIR);

using namespace ExaDiS;

template<class F>
void test_force(std::string name="")
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

    System* system = make_system(config, crystal, params);
    
    system->extstress = Mat33().voigt(10e6, 5e6, 20e6, 3e6, 7e6, 1e6);

    Force* force;
    if constexpr (std::is_same<F,ForceType::LINE_TENSION_MODEL>::value) {
        force = exadis_new<ForceType::LINE_TENSION_MODEL>(system);
    }
    else if constexpr (std::is_same<F,ForceType::CUTOFF_MODEL>::value) {
        force = exadis_new<ForceType::CUTOFF_MODEL>(system, 
            ForceType::CORE_SELF_PKEXT::Params(),
            ForceType::FORCE_SEGSEG_ISO::Params(7500.0)
        );
    }
    else if constexpr (std::is_same<F,ForceType::DDD_FFT_MODEL>::value) {
        force = exadis_new<ForceType::DDD_FFT_MODEL>(system, 
            ForceType::CORE_SELF_PKEXT::Params(),
            ForceType::LONG_FFT_SHORT_ISO::Params(64)
        );
    }
    else if constexpr (std::is_same<F,ForceFFT>::value) {
        force = exadis_new<ForceFFT>(system,
            ForceFFT::Params(64)
        );
    }

    force->pre_compute(system);
    force->compute(system);
    
    if (name == "fft_serialdisnet") {
        // Test synchronization of FFT grid data
        ForceFFT* forcefft = static_cast<ForceFFT*>(force);
        forcefft->synchronize_stress_gridval();
        SerialDisNet* network = system->get_serial_network();
        forcefft->zero_force(network);
        auto nodes = network->get_nodes();
        auto segs = network->get_segs();
        for (int i = 0; i < network->Nsegs_local; i++) {
            int n1 = segs[i].n1;
            int n2 = segs[i].n2;
            SegForce fseg = forcefft->segment_force(system, network, i);
            nodes[n1].f += fseg.f1;
            nodes[n2].f += fseg.f2;
        }
    }
    
    if (0) {
        // write reference results in a file
        debug::write_force(system, "test_force_"+name+".dat");
    } else {
        // print current results in the console
        SerialDisNet* network = system->get_serial_network();
        for (int i = 0; i < network->Nnodes_local; i++) {
            Vec3 f = network->nodes[i].f;
            printf("%e %e %e\n", f.x, f.y, f.z);
        }
    }

    exadis_delete(force);
    exadis_delete(system);
}

int main(int argc, char* argv[])
{
    ExaDiS::Initialize init(argc, argv);
    
    std::string name = "";
    if (argc == 2) name = std::string(argv[1]);
        
    if (name == "lt")
        test_force<ForceType::LINE_TENSION_MODEL>("lt");
    else if (name == "cutoff")
        test_force<ForceType::CUTOFF_MODEL>("cutoff");
    else if (name == "ddd_fft")
        test_force<ForceType::DDD_FFT_MODEL>("ddd_fft");
    else if (name == "fft" || name == "fft_serialdisnet")
        test_force<ForceFFT>(name);
    else
        std::cerr << "Error: invalid force type = '" << name << "'" << std::endl;

    return 0;
}

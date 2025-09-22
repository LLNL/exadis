/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "types.h"

using namespace ExaDiS;

/*---------------------------------------------------------------------------
 *
 *    Struct:       SystemTest
 *
 *-------------------------------------------------------------------------*/
struct SystemTest {
    int Nnodes;
    Kokkos::View<Vec3*> nodes;
    
    SystemTest() {
        Nnodes = 4;
        Kokkos::resize(nodes, Nnodes);
        auto h_nodes = Kokkos::create_mirror_view(nodes);
        for (int i = 0; i < Nnodes; i++)
            h_nodes(i) = Vec3(0.0);
        Kokkos::deep_copy(nodes, h_nodes);
    }
    
    bool check_results(bool print=0) {
        auto h_nodes = Kokkos::create_mirror_view(nodes);
        Kokkos::deep_copy(h_nodes, nodes);
        if (print) printf(" Results:\n");
        for (int i = 0; i < Nnodes; i++) {
            if ((h_nodes(i)-Vec3(1.0*i)).norm2() > 1e-10) return false;
            if (print) printf("  nodes(%d) = %f %f %f\n", i, h_nodes(i).x, h_nodes(i).y, h_nodes(i).z);
        }
        return true;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     test_system
 *
 *-------------------------------------------------------------------------*/
struct FunctorSystem {
    SystemTest system;
    FunctorSystem(SystemTest& _system) : system(_system) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const {
        system.nodes(i) = Vec3(1.0*i);
    }
};

void test_system()
{
    ExaDiS_log("test_system()\n");
    
    SystemTest* system = new SystemTest();
    try
    {
        Kokkos::parallel_for("FunctorSystem",
            system->Nnodes, FunctorSystem(*system)
        );
        Kokkos::fence();
        ExaDiS_log(" %s\n", system->check_results() ? "PASS" : "FAIL");
    }
    catch(const std::runtime_error& re) {
        ExaDiS_log("Runtime error: %s\n", re.what());
        ExaDiS_log(" FAIL\n");
    } catch(const std::exception& ex) {
        ExaDiS_log("Error occurred: %s\n", ex.what());
        ExaDiS_log(" FAIL\n");
    } catch(...) {
        ExaDiS_log("Unknown error occurred\n");
        ExaDiS_log(" FAIL\n");
    }
    delete system;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_system_unified
 *
 *-------------------------------------------------------------------------*/
struct FunctorSystemUnified {
    SystemTest* system;
    FunctorSystemUnified(SystemTest* _system) : system(_system) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const int& i) const {
        system->nodes(i) = Vec3(1.0*i);
    }
};

void test_system_unified()
{
    ExaDiS_log("test_system_unified()\n");
    
    SystemTest* system = exadis_new<SystemTest>();
    try
    {
        Kokkos::parallel_for("FunctorSystemUnified",
            system->Nnodes, FunctorSystemUnified(system)
        );
        Kokkos::fence();
        ExaDiS_log(" %s\n", system->check_results() ? "PASS" : "FAIL");
    }
    catch(const std::runtime_error& re) {
        ExaDiS_log("Runtime error: %s\n", re.what());
        ExaDiS_log(" FAIL\n");
    } catch(const std::exception& ex) {
        ExaDiS_log("Error occurred: %s\n", ex.what());
        ExaDiS_log(" FAIL\n");
    } catch(...) {
        ExaDiS_log("Unknown error occurred\n");
        ExaDiS_log(" FAIL\n");
    }
    exadis_delete(system);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    ExaDiS::Initialize init(argc, argv);
    
    std::string test_name = "";
    if (argc > 1)
        test_name = std::string(argv[1]);
    
    if (test_name == "test_system" || test_name.empty())
        test_system();
    if (test_name == "test_system_unified" || test_name.empty())
        test_system_unified();
    
    return 0;
}

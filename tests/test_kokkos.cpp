/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include <Kokkos_Core.hpp>

/*---------------------------------------------------------------------------
 *
 *    Function:     test_memory
 *
 *-------------------------------------------------------------------------*/
void test_memory()
{
    printf("test_memory()\n");
    
    int N = 5;
    Kokkos::View<int*> array("array", N);
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int& i) {
        array(i) = 2*i;
    });
    Kokkos::fence();
    
    auto h_array = Kokkos::create_mirror_view(array);
    Kokkos::deep_copy(h_array, array);
    int error = 0;
    for (int i = 0; i < N; i++)
        error += (h_array(i) != 2*i);
    
    if (error) printf(" FAIL\n");
    else printf(" PASS\n");
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_unified_memory
 *
 *-------------------------------------------------------------------------*/
void test_unified_memory()
{
    printf("test_unified_memory()\n");
    
    int N = 5;
    Kokkos::View<int*, Kokkos::SharedSpace> array("array", N);
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int& i) {
        array(i) = 2*i;
    });
    Kokkos::fence();
    
    int error = 0;
    for (int i = 0; i < N; i++)
        error += (array(i) != 2*i);
    
    if (error) printf(" FAIL\n");
    else printf(" PASS\n");
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    std::string test_name = "";
    if (argc > 1)
        test_name = std::string(argv[1]);
    
    if (test_name == "test_initialize" || test_name.empty())
        printf("test_initialize()\n");
    Kokkos::ScopeGuard guard(argc, argv);
    if (test_name == "test_initialize" || test_name.empty())
        printf(" PASS\n");
    
    if (test_name == "test_memory" || test_name.empty())
        test_memory();
    if (test_name == "test_unified_memory" || test_name.empty())
        test_unified_memory();
    
    return 0;
}

/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Nicolas Bertin
 *    bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_TYPES_H
#define EXADIS_TYPES_H

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include "vec.h"
#include "network.h"
#include <new>

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)
#define SIGN(a) (a>0?1:(-1*(a<0)))

namespace ExaDiS {

template <unsigned int error> void print_(const char *format, ...);
#define ExaDiS_log print_<0>
#define ExaDiS_fatal print_<1>

#define EXADIS_NOT_IMPLEMENTED() { \
    printf("Error: function %s not implemented at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
    exit(1); \
}

/*---------------------------------------------------------------------------
 *
 *    Class:        Initialize
 *                  This class acts as an alias for the Kokkos ScopeGuard
 *                  to initialize and finalize both Kokkos and MPI.
 *
 *-------------------------------------------------------------------------*/
class Initialize {
public:
    Initialize(int argc, char* argv[]) {
#ifdef MPI
        MPI_Init(&argc, &argv);
#endif
        if (Kokkos::is_initialized()) {
            Kokkos::abort(Kokkos::Impl::scopeguard_create_while_initialized_warning().c_str());
        }
        if (Kokkos::is_finalized()) {
            Kokkos::abort(Kokkos::Impl::scopeguard_create_after_finalize_warning().c_str());
        }
        Kokkos::initialize(argc, argv);
    }
    ~Initialize() {
        if (Kokkos::is_finalized()) {
            Kokkos::abort(Kokkos::Impl::scopeguard_destruct_after_finalize_warning().c_str());
        }
        Kokkos::finalize();
#ifdef MPI
        MPI_Finalize();
#endif
    }
};

/*---------------------------------------------------------------------------
 *
 *    Functions:    Memory management and defintions
 *
 *-------------------------------------------------------------------------*/
#ifndef EXADIS_FULL_UNIFIED_MEMORY
#define EXADIS_FULL_UNIFIED_MEMORY 1
#endif

template<typename T, typename... Args>
inline T* exadis_new(Args... args) {
#if 1 //EXADIS_UNIFIED_MEMORY
    void* p = Kokkos::kokkos_malloc<Kokkos::SharedSpace>(sizeof(T));
    return new(p) T(args...);
#else
    return new T(args...);
#endif
}

template<typename T>
inline void exadis_delete(T *p) {
#if 1 //EXADIS_UNIFIED_MEMORY
    if (p) p->~T();
    Kokkos::kokkos_free<Kokkos::SharedSpace>(p);
#else
    if (p) delete p;
#endif
}

#if EXADIS_FULL_UNIFIED_MEMORY
typedef typename Kokkos::SharedSpace T_memory_space;
#else
typedef typename Kokkos::DefaultExecutionSpace::memory_space T_memory_space;
#endif
typedef typename Kokkos::SharedSpace T_memory_shared;

typedef Kokkos::View<DisNode*, T_memory_space> T_nodes;
typedef Kokkos::View<DisSeg*, T_memory_space> T_segs;
typedef Kokkos::View<Conn*, T_memory_space> T_conn;
typedef Kokkos::View<Vec3*, T_memory_space> T_x;
typedef Kokkos::View<Vec3*, T_memory_space> T_v;

typedef Kokkos::TeamPolicy<>::member_type team_handle;

/*---------------------------------------------------------------------------
 *
 *    Functions:    Resize view
 *                  Allocate views with a buffer factor so that resizing
 *                  only triggers a memory reallocation infrequently
 *
 *-------------------------------------------------------------------------*/
#define ALLOC_SIZE_FACT 1.5
static const float ALLOC_MIN_FACT = (2.0-ALLOC_SIZE_FACT)/ALLOC_SIZE_FACT;

template <class T, class... P>
inline std::enable_if_t<
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,Kokkos::LayoutLeft> ||
    std::is_same_v<typename Kokkos::View<T, P...>::array_layout,Kokkos::LayoutRight>>
resize_view(Kokkos::View<T, P...>& v, const size_t n) {
    const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG;
    if (n > v.extent(0) || n < static_cast<size_t>(v.extent(0) * ALLOC_MIN_FACT)) {
        size_t nalloc = static_cast<size_t>(n * ALLOC_SIZE_FACT);
        Kokkos::impl_resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                            v, nalloc, n0, n0, n0, n0, n0, n0, n0);
    }
}

template <class... Properties>
void resize_view(Kokkos::DualView<Properties...>& dv, const size_t n) {
    if (n > dv.extent(0) || n < static_cast<size_t>(dv.extent(0) * ALLOC_MIN_FACT)) {
        size_t nalloc = static_cast<size_t>(n * ALLOC_SIZE_FACT);
        dv.resize(nalloc);
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     get_team_sizes()
 *                  Helper function to return the default team size and
 *                  number of teams to use to distribute Ntot operations
 *
 *-------------------------------------------------------------------------*/
struct TeamSize {
    int team_size, num_teams;
    TeamSize(int _ts, int _nt) : team_size(_ts), num_teams(_nt) {}
};
template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
static TeamSize get_team_sizes(int Ntot, int max_size=128) {
#if defined(KOKKOS_ENABLE_OPENMP)
    if (std::is_same<ExecutionSpace, Kokkos::OpenMP>::value)
        max_size = 64; // hard-coded Kokkos limit
#endif
    int concurrency = ExecutionSpace::concurrency();
    int team_size = MIN(concurrency, max_size);
    int num_teams = (Ntot + team_size - 1) / team_size;
    return TeamSize(team_size, num_teams);
}


/*---------------------------------------------------------------------------
 *
 *    Class:        DeviceDisNet
 *                  This class implements a Kokkos-based network data structure 
 *                  for use in a device (e.g. GPU) execution space.
 *
 *-------------------------------------------------------------------------*/
class DeviceDisNet {
public:
    typedef typename Kokkos::DefaultExecutionSpace ExecutionSpace;
    static const char* name() { return "DeviceDisNet"; }
    
    Cell cell;
    int Nnodes_local, Nsegs_local;
    T_nodes nodes;
    T_segs segs;
    T_conn conn;
    
    DeviceDisNet(const Cell &_cell) : cell(_cell) {}
    
    KOKKOS_FORCEINLINE_FUNCTION T_nodes::pointer_type get_nodes() { return nodes.data(); }
    KOKKOS_FORCEINLINE_FUNCTION T_segs::pointer_type get_segs() { return segs.data(); }
    KOKKOS_FORCEINLINE_FUNCTION T_conn::pointer_type get_conn() { return conn.data(); }
    
    inline void update_ptr() {}
};

/*---------------------------------------------------------------------------
 *
 *    Class:        DisNetManager
 *                  This class implements the DisNetManager that handles and
 *                  synchronizes the dislocation network object between the
 *                  different execution spaces.
 *
 *-------------------------------------------------------------------------*/
class DisNetManager {
public:
    enum Active {SERIAL_ACTIVE, DEVICE_ACTIVE};
    
    DisNetManager(SerialDisNet *n) {
        set_network(n);
        d_network = exadis_new<DeviceDisNet>(n->cell);
    }
    DisNetManager(DeviceDisNet *n) {
        set_network(n);
        s_network = new SerialDisNet(n->cell);
    }
    
    SerialDisNet *get_serial_network()
    {
        if (active != SERIAL_ACTIVE) {
            // Transfer memory from device to serial networks
            //printf("Network transfer device->serial\n");
            
            s_network->nodes.resize(d_network->Nnodes_local);
            s_network->segs.resize(d_network->Nsegs_local);
            s_network->conn.resize(d_network->Nnodes_local);
            
        #if EXADIS_FULL_UNIFIED_MEMORY
            for (int i = 0; i < d_network->Nnodes_local; i++)
                s_network->nodes[i] = d_network->nodes(i);
            for (int i = 0; i < d_network->Nsegs_local; i++)
                s_network->segs[i] = d_network->segs(i);
            for (int i = 0; i < d_network->Nnodes_local; i++)
                s_network->conn[i] = d_network->conn(i);
        #else
            T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(d_network->nodes);
            T_segs::HostMirror h_segs = Kokkos::create_mirror_view(d_network->segs);
            T_conn::HostMirror h_conn = Kokkos::create_mirror_view(d_network->conn);
            
            Kokkos::deep_copy(h_nodes, d_network->nodes);
            Kokkos::deep_copy(h_segs, d_network->segs);
            Kokkos::deep_copy(h_conn, d_network->conn);
            
            for (int i = 0; i < d_network->Nnodes_local; i++)
                s_network->nodes[i] = h_nodes(i);
            for (int i = 0; i < d_network->Nsegs_local; i++)
                s_network->segs[i] = h_segs(i);
            for (int i = 0; i < d_network->Nnodes_local; i++)
                s_network->conn[i] = h_conn(i);
        #endif
            
            // Copy cell in case it has changed
            s_network->cell = d_network->cell;
        }
        s_network->update_ptr();
        set_active(SERIAL_ACTIVE);
        return s_network;
    };
    
    DeviceDisNet *get_device_network()
    {
        if (active != DEVICE_ACTIVE) {
            // Transfer memory from device to serial networks
            //printf("Network transfer serial->device\n");
            
            Kokkos::resize(d_network->nodes, s_network->number_of_nodes());
            Kokkos::resize(d_network->segs, s_network->number_of_segs());
            Kokkos::resize(d_network->conn, s_network->number_of_nodes());
            
        #if EXADIS_FULL_UNIFIED_MEMORY
            for (int i = 0; i < s_network->number_of_nodes(); i++)
                d_network->nodes(i) = s_network->nodes[i];
            for (int i = 0; i < s_network->number_of_segs(); i++)
                d_network->segs(i) = s_network->segs[i];
            for (int i = 0; i < s_network->number_of_nodes(); i++)
                d_network->conn(i) = s_network->conn[i];
        #else
            T_nodes::HostMirror h_nodes = Kokkos::create_mirror_view(d_network->nodes);
            T_segs::HostMirror h_segs = Kokkos::create_mirror_view(d_network->segs);
            T_conn::HostMirror h_conn = Kokkos::create_mirror_view(d_network->conn);
            
            for (int i = 0; i < s_network->number_of_nodes(); i++)
                h_nodes(i) = s_network->nodes[i];
            for (int i = 0; i < s_network->number_of_segs(); i++)
                h_segs(i) = s_network->segs[i];
            for (int i = 0; i < s_network->number_of_nodes(); i++)
                h_conn(i) = s_network->conn[i];
            
            Kokkos::deep_copy(d_network->nodes, h_nodes);
            Kokkos::deep_copy(d_network->segs, h_segs);
            Kokkos::deep_copy(d_network->conn, h_conn);
        #endif
            
            d_network->Nnodes_local = s_network->number_of_nodes();
            d_network->Nsegs_local = s_network->number_of_segs();
            
            // Copy cell in case it has changed
            d_network->cell = s_network->cell;
        }
        set_active(DEVICE_ACTIVE);
        return d_network;
    };
    
    void set_network(SerialDisNet *n) {
        s_network = n;
        set_active(SERIAL_ACTIVE);
        set_need_sync();
    }
    
    void set_network(DeviceDisNet *n) {
        d_network = n;
        set_active(DEVICE_ACTIVE);
        set_need_sync();
    }
    
    inline int get_active() { return active; }
    inline void set_active(int a) { active = a; }
    inline void set_need_sync() { need_sync = true; }
    inline void clear_sync() { need_sync = false; }
    
    inline int Nnodes_local() { 
        if (active == SERIAL_ACTIVE) return s_network->number_of_nodes();
        else return d_network->Nnodes_local;
    }
    inline int Nsegs_local() {
        if (active == SERIAL_ACTIVE) return s_network->number_of_segs();
        else return d_network->Nsegs_local;
    }
    
    ~DisNetManager() {
        if (s_network) delete s_network;
        exadis_delete(d_network);
    }
    
private:
    int active; // serial or device
    bool need_sync;
    SerialDisNet *s_network;
    DeviceDisNet *d_network;
};

} // namespace ExaDiS


/*---------------------------------------------------------------------------
 *
 *    Struct:       reduction_identity
 *
 *-------------------------------------------------------------------------*/
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<>
    struct reduction_identity<ExaDiS::Vec3> {
        KOKKOS_FORCEINLINE_FUNCTION static ExaDiS::Vec3 sum() {
            return ExaDiS::Vec3();
        }
    };
    
    template<>
    struct reduction_identity<ExaDiS::Mat33> {
        KOKKOS_FORCEINLINE_FUNCTION static ExaDiS::Mat33 sum() {
            return ExaDiS::Mat33();
        }
    };
} // namespace Kokkos

/*---------------------------------------------------------------------------
 *
 *    Struct:       RandomGenerator
 *
 *-------------------------------------------------------------------------*/
#include <Kokkos_Random.hpp>

struct RandomGenerator {
    Kokkos::Random_XorShift64_Pool<Kokkos::Serial> random_pool_serial;
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> random_pool_device;
    
    RandomGenerator(int seed=12345) {
        random_pool_serial = Kokkos::Random_XorShift64_Pool<Kokkos::Serial>(seed);
        random_pool_device = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>(seed);
    }
    
    template<class ExecutionSpace>
    KOKKOS_FORCEINLINE_FUNCTION
    int rand(int min, int max) const {
        if constexpr (std::is_same<ExecutionSpace,Kokkos::Serial>::value) {
            auto generator = random_pool_serial.get_state();
            int val = generator.rand(min, max);
            random_pool_serial.free_state(generator);
            return val;
        } else {
            auto generator = random_pool_device.get_state();
            int val = generator.rand(min, max);
            random_pool_device.free_state(generator);
            return val;
        }
    }
    
    template<class ExecutionSpace>
    KOKKOS_FORCEINLINE_FUNCTION
    double drand(double min, double max) const {
        if constexpr (std::is_same<ExecutionSpace,Kokkos::Serial>::value) {
            auto generator = random_pool_serial.get_state();
            double val = generator.drand(min, max);
            random_pool_serial.free_state(generator);
            return val;
        } else {
            auto generator = random_pool_device.get_state();
            double val = generator.drand(min, max);
            random_pool_device.free_state(generator);
            return val;
        }
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:       SortView
 *
 *-------------------------------------------------------------------------*/
#include <algorithm>
#include <numeric>
#ifdef KOKKOS_ENABLE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct SortView
{
    template <typename ViewType>
    SortView(ViewType& val, int beg, int end) {
        std::sort(val.data() + beg, val.data() + end);
    }
};

template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct SortViewByKey
{
    template <typename ViewTypeKey, typename ViewTypeVal>
    SortViewByKey(ViewTypeKey& key, ViewTypeVal& val, int beg, int end) {
        std::vector<size_t> idx(end-beg);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&key,beg](size_t i1, size_t i2) {
            return key[i1+beg] < key[i2+beg];
        });
        for (size_t i = 0; i < idx.size(); i++) {
            size_t curr = i;
            size_t next = idx[curr];
            while (next != i) {
                std::swap(key[curr+beg], key[next+beg]);
                std::swap(val[curr+beg], val[next+beg]);
                idx[curr] = curr;
                curr = next;
                next = idx[next];
            }
            idx[curr] = curr;
        }
    }
};

#ifdef KOKKOS_ENABLE_CUDA
template<>
struct SortView<Kokkos::Cuda>
{
    template <typename ViewType>
    SortView(const ViewType& val, int beg, int end) {
        typedef typename ViewType::value_type val_type;
        thrust::sort(thrust::device_ptr<val_type>(val.data() + beg),
                     thrust::device_ptr<val_type>(val.data() + end));
    }
};

template<>
struct SortViewByKey<Kokkos::Cuda>
{
    template <typename ViewTypeKey, typename ViewTypeVal>
    SortViewByKey(const ViewTypeKey& key, const ViewTypeVal& val, int beg, int end) {
        typedef typename ViewTypeKey::value_type key_type;
        typedef typename ViewTypeVal::value_type val_type;
        thrust::sort_by_key(thrust::device_ptr<key_type>(key.data() + beg),
                            thrust::device_ptr<key_type>(key.data() + end),
                            thrust::device_ptr<val_type>(val.data() + beg));
    }
};
#endif

#endif

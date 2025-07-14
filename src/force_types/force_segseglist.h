/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_SEGSEGLIST_H
#define EXADIS_FORCE_SEGSEGLIST_H

#include "force.h"
#include "neighbor.h"
#include "functions.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        SegSegList
 *                  Class to build and manage a list of segment pairs
 *                  across the different execution/memory spaces.
 *
 *-------------------------------------------------------------------------*/
class SegSegList {
public:
    struct Params {
        double cutoff;
        Params() { cutoff = -1.0; }
        Params(double _cutoff) : cutoff(_cutoff) {}
    };
    
    double cutoff;
    int Nsegseg;
    bool use_compute_map;
    
    SegSegList(System *system, double _cutoff, bool _use_compute_map) {
        set_cutoff(system, _cutoff);
        use_compute_map = _use_compute_map;
        initialize();
    }
    
    SegSegList(System *system, Params params, bool _use_compute_map) {
        set_cutoff(system, params.cutoff);
        use_compute_map = _use_compute_map;
        initialize();
    }
    
    void set_cutoff(System* system, double _cutoff) {
        cutoff = _cutoff;
        if (cutoff < 0)
            ExaDiS_fatal("Error: undefined cutoff parameter in SegSegList\n");
        // Set the general neighbor cutoff for the simulation
        system->register_neighbor_cutoff(cutoff);
    }
    
    void initialize() {
        Nsegseg = 0;
        Kokkos::resize(gcount, 1);
        neilist = exadis_new<NeighborList>();
        need_sync = false;
        map_built = false;
        a_ssl.ssl = this;
        use_flag = false;
    }
    
    template<class N>
    struct BuildSegSegList {
        N* net;
        SegSegList* segseglist;
        NeighborList* neilist;
        double cutoff2;
        bool count_only;
        
        BuildSegSegList(N* _net, SegSegList* _segseglist, NeighborList* _neilist, 
                        double _cutoff, bool _count_only) : 
        net(_net), segseglist(_segseglist), neilist(_neilist), count_only(_count_only) {
            cutoff2 = _cutoff * _cutoff;
            segseglist->reset_gcount(net);
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const team_handle& team) const {
            if (cutoff2 <= 0.0) return;
            
            int tid = team.team_rank();
            int i = team.league_rank();
            
            auto gcount = segseglist->get_gcount(net);
            auto list = segseglist->get_list(net);
            
            auto count = neilist->get_count();
            auto nei = neilist->get_nei();
            
            int Nnei = 0;
            Kokkos::single(Kokkos::PerTeam(team), [=] (int& nei) {
                nei = count[i];
            }, Nnei);
            
            if (count_only) {
                
                int Nneitot;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, Nnei), [&] (const int& l, int& nsum) {
                    int j = nei(i,l); // neighbor seg
                    if (i < j) { // avoid double-counting
                        // Compute distance
                        double dist2 = get_min_dist2_segseg(net, i, j);
                        if (dist2 >= 0.0 && dist2 < cutoff2) {
                            nsum++;
                        }
                    }
                }, Nneitot);
                
                Kokkos::single(Kokkos::PerTeam(team), [=] () {
                    Kokkos::atomic_add(&gcount[0], Nneitot);
                });
                Kokkos::single(Kokkos::PerTeam(team), [=] () {
                    segseglist->segneicount(i) = Nneitot;
                });
                
            } else {
                
                int idx;
                int* shared_count = (int*)team.team_scratch(0).get_shmem(sizeof(int));
                Kokkos::single(Kokkos::PerTeam(team), [=] (int& tidx) {
                    tidx = Kokkos::atomic_fetch_add(&gcount[0], segseglist->segneicount(i));
                    shared_count[0] = 0;
                }, idx);
                
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, Nnei), [&] (const int& l) {
                    int j = nei(i,l); // neighbor seg
                    if (i < j) { // avoid double-counting
                        // Compute distance
                        double dist2 = get_min_dist2_segseg(net, i, j);
                        if (dist2 >= 0.0 && dist2 < cutoff2) {
                            int k = Kokkos::atomic_fetch_add(&shared_count[0], 1);
                            list[idx+k] = SegSeg(i, j);
                        }
                    }
                });
            }
        }
    };

    template<class N>
    void build_list(System* system, N* net)
    {    
        generate_neighbor_list(system, net, neilist, cutoff, Neighbor::NeiSeg);
        
        Kokkos::resize(segneicount, net->Nsegs_local);
        
        Kokkos::parallel_for("ForceSegSeg::BuildSegSegList",
            Kokkos::TeamPolicy<typename N::ExecutionSpace>(net->Nsegs_local, Kokkos::AUTO),
            BuildSegSegList<N>(net, this, neilist, cutoff, true)
        );
        Kokkos::fence();
        
        init_list<N>(net);
        //printf("SegSegList: cutoff = %e, Nsegseg = %d\n", cutoff, Nsegseg);
        
        Kokkos::parallel_for("ForceSegSeg::BuildSegSegList",
            Kokkos::TeamPolicy<typename N::ExecutionSpace>(net->Nsegs_local, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(sizeof(int))),
            BuildSegSegList<N>(net, this, neilist, cutoff, false)
        );
        Kokkos::fence();
        
        if (use_compute_map) {
            auto compute_map = get_compute_map(net);
            build_compute_map<N>(this, compute_map, net);
        }
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::DualView<SegSeg*>::t_dev::pointer_type get_list(DeviceDisNet *n) { 
        return segseglist.d_view.data();
    }
    /*
    KOKKOS_INLINE_FUNCTION
    Kokkos::DualView<SegSeg*>::t_host::pointer_type get_list(SerialDisNet *n) { 
        //if (need_sync) error...
        return segseglist.h_view.data();
    }
    */
    KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::DualView<SegSegForce*>::t_dev::pointer_type get_fsegseglist(DeviceDisNet *n) { 
        return fsegseglist.d_view.data();
    }
    /*
    KOKKOS_INLINE_FUNCTION
    Kokkos::DualView<SegSegForce*>::t_host::pointer_type get_fsegseglist(SerialDisNet *n) { 
        //if (need_sync) error...
        return fsegseglist.h_view.data();
    }
    */
    
    NeighborList* neilist;
    Kokkos::DualView<int*> gcount;
    Kokkos::DualView<SegSeg*> segseglist;
    Kokkos::DualView<SegSegForce*> fsegseglist;
    Kokkos::View<int*> segneicount;
    
    bool use_flag;
    Kokkos::View<bool*> segsegflag;
    
    struct SegSegFlagAccessor {
        SegSegList* ssl;
        KOKKOS_FORCEINLINE_FUNCTION
        bool operator[](int i) {
            if (!ssl->use_flag) return 1;
            return ssl->segsegflag(i);
        }
    };
    SegSegFlagAccessor a_ssl;
    
    KOKKOS_FORCEINLINE_FUNCTION SegSegFlagAccessor get_flag(DeviceDisNet *n) { return a_ssl; }

private:
    bool need_sync;
    bool map_built;
    
    template<class N>
    void init_list(N *net)
    {    
        if constexpr (std::is_same<N, DeviceDisNet>::value) {
            need_sync = true;
            Kokkos::deep_copy(gcount.h_view, gcount.d_view);
        } else {
            need_sync = false;
        }
        
        Nsegseg = gcount.h_view(0);
        resize_view(segseglist, Nsegseg);
        map_built = false;
        
        if (use_compute_map)
            resize_view(fsegseglist, Nsegseg);
    };
    
    inline void reset_gcount(DeviceDisNet *n) { Kokkos::deep_copy(gcount.d_view, 0); }
    inline void reset_gcount(SerialDisNet *n) { Kokkos::deep_copy(gcount.h_view, 0); }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::DualView<int*>::t_dev::pointer_type get_gcount(DeviceDisNet *n) { 
        return gcount.d_view.data();
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::DualView<int*>::t_host::pointer_type get_gcount(SerialDisNet *n) { 
        return gcount.h_view.data();
    }
    
public:
    template<class N>
    struct NodeComputeMap {
        typedef typename N::ExecutionSpace::memory_space memory_space;
        Kokkos::View<int*, memory_space> beg, end;
        Kokkos::View<int*, memory_space> segseg;
        Kokkos::View<int*, memory_space> fpos;
        
        inline void resize(int Nnodes, int Nsegseg) {
            Kokkos::resize(beg, Nnodes);
            Kokkos::resize(end, Nnodes);
            resize_view(segseg, 4*Nsegseg);
            resize_view(fpos, 4*Nsegseg);
        }
    };
    NodeComputeMap<DeviceDisNet> d_compute_map;
    NodeComputeMap<SerialDisNet> s_compute_map;
    
    KOKKOS_FORCEINLINE_FUNCTION NodeComputeMap<DeviceDisNet> *get_compute_map(DeviceDisNet *n) { return &d_compute_map; }
    KOKKOS_FORCEINLINE_FUNCTION NodeComputeMap<SerialDisNet> *get_compute_map(SerialDisNet *n) { return &s_compute_map; }
    
    template<class N>
    struct BuildComputeMap {
        N *net;
        SegSegList *segseglist;
        NodeComputeMap<N> *compute_map;
        
        typedef typename N::ExecutionSpace::memory_space memory_space;
        Kokkos::View<int*, memory_space> nsize;
        
        BuildComputeMap(N *_net, SegSegList *_segseglist, 
                        Kokkos::View<int*, memory_space> &_nsize, 
                        NodeComputeMap<N> *_compute_map) : 
        net(_net), segseglist(_segseglist), nsize(_nsize), compute_map(_compute_map) {}
        
        struct TagCount {};
        struct TagScan {};
        struct TagFill {};
        
        KOKKOS_INLINE_FUNCTION
        void operator() (TagCount, const int &i) const {
            auto segs = net->get_segs();
            auto list = segseglist->get_list(net);
            SegSeg segseg = list[i];
            int n1 = segs[segseg.s1].n1;
            int n2 = segs[segseg.s1].n2;
            int n3 = segs[segseg.s2].n1;
            int n4 = segs[segseg.s2].n2;
            Kokkos::atomic_inc(&nsize(n1));
            Kokkos::atomic_inc(&nsize(n2));
            Kokkos::atomic_inc(&nsize(n3));
            Kokkos::atomic_inc(&nsize(n4));
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (TagScan, int i, int &partial_sum, bool is_final) const {
            if (is_final) compute_map->beg(i) = partial_sum;
            partial_sum += nsize(i);
            if (is_final) compute_map->end(i) = partial_sum;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (TagFill, const int &i) const {
            auto segs = net->get_segs();
            auto list = segseglist->get_list(net);
            SegSeg segseg = list[i];
            int n1 = segs[segseg.s1].n1;
            int n2 = segs[segseg.s1].n2;
            int n3 = segs[segseg.s2].n1;
            int n4 = segs[segseg.s2].n2;
            
            int idx, beg;
            
            beg = compute_map->beg(n1);
            idx = Kokkos::atomic_fetch_add(&nsize(n1), 1);
            compute_map->segseg(beg+idx) = i;
            compute_map->fpos(beg+idx) = 0;
            
            beg = compute_map->beg(n2);
            idx = Kokkos::atomic_fetch_add(&nsize(n2), 1);
            compute_map->segseg(beg+idx) = i;
            compute_map->fpos(beg+idx) = 1;
            
            beg = compute_map->beg(n3);
            idx = Kokkos::atomic_fetch_add(&nsize(n3), 1);
            compute_map->segseg(beg+idx) = i;
            compute_map->fpos(beg+idx) = 2;
            
            beg = compute_map->beg(n4);
            idx = Kokkos::atomic_fetch_add(&nsize(n4), 1);
            compute_map->segseg(beg+idx) = i;
            compute_map->fpos(beg+idx) = 3;
        }
    };
    
    template<class N>
    static void build_compute_map(SegSegList* ssl, NodeComputeMap<N>* compute_map, N *net)
    {    
        compute_map->resize(net->Nnodes_local, ssl->Nsegseg);
        
        typedef typename N::ExecutionSpace::memory_space memory_space;
        Kokkos::View<int*, memory_space> nsize("nsize", net->Nnodes_local);
        
        BuildComputeMap f(net, ssl, nsize, compute_map);
        
        Kokkos::parallel_for("BuildComputeMap::Count",
            Kokkos::RangePolicy<typename N::ExecutionSpace,
            typename BuildComputeMap<N>::TagCount>(0, ssl->Nsegseg), f
        );
        Kokkos::fence();
        
        Kokkos::parallel_scan("BuildComputeMap::Scan",
            Kokkos::RangePolicy<typename N::ExecutionSpace,
            typename BuildComputeMap<N>::TagScan>(0, net->Nnodes_local), f
        );
        Kokkos::fence();
        
        Kokkos::deep_copy(nsize, 0);
        Kokkos::parallel_for("BuildComputeMap::Fill",
            Kokkos::RangePolicy<typename N::ExecutionSpace,
            typename BuildComputeMap<N>::TagFill>(0, ssl->Nsegseg), f
        );
        Kokkos::fence();
    }
    
    ~SegSegList() {
        exadis_delete(neilist);
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceSegSegList
 *                  Base class for force contributions between segment pairs
 *                  that are computed based on a segseglist build at 
 *                  pre-compute time given a pair cutoff.
 *                  The class is instantiated with a segsegforce kernel <F> in
 *                  which a segseg_force() function must be defined.
 *
 *                  Option use_compute_map implements an approach to compute
 *                  and store pairwise forces in a dedicated array, and then
 *                  assemble nodal forces using a node_map. This is to avoid 
 *                  heavy usage of atomic operations on devices (e.g. GPU),
 *                  which can significantly hurt performance.
 *
 *-------------------------------------------------------------------------*/
template<class F>
class ForceSegSegList : public Force {
private:
    SegSegList* segseglist;
    F* force; // segsegforce kernel
    
#if defined(EXADIS_USE_COMPUTE_MAPS)
    static const bool _use_compute_map = true;
#else
    static const bool _use_compute_map = false;
#endif
    
    int TIMER_SEGSEGLIST, TIMER_COMPUTE;
    
public:
    //typedef typename SegSegList::Params Sparams;
    typedef typename F::Params Fparams;
    
    struct Params {
        double cutoff;
        bool use_compute_map = _use_compute_map;
        Fparams fparams = Fparams();
        Params(double _cutoff) : cutoff(_cutoff) {}
        Params(double _cutoff, Fparams _fparams) : cutoff(_cutoff), fparams(_fparams) {}
    };
    
    ForceSegSegList(System* system, Params params) {
        force = exadis_new<F>(system, params.fparams);
        initialize(system, params.cutoff, params.use_compute_map);
    }
    
    ForceSegSegList(System* system, double cutoff, Fparams fparams=Fparams()) {
        force = exadis_new<F>(system, fparams);
        initialize(system, cutoff, _use_compute_map);
    }
    
    template<class FLong>
    ForceSegSegList(System* system, FLong* flong) {
        force = exadis_new<F>(system, flong);
        double cutoff = flong->get_neighbor_cutoff();
        initialize(system, cutoff, _use_compute_map);
    }
    
    void initialize(System* system, double cutoff, bool use_compute_map) {
        segseglist = exadis_new<SegSegList>(system, cutoff, use_compute_map);
        TIMER_SEGSEGLIST = system->add_timer("ForceSegSegList build list");
        TIMER_COMPUTE = system->add_timer("ForceSegSegList compute forces");
    }
    
    double get_cutoff() { return segseglist->cutoff; }
    SegSegList* get_segseglist() { return segseglist; }

    template<class N>
    struct AddSegSegForce {
        System *system;
        F *force;
        N *net;
        SegSegList *segseglist;
        
        AddSegSegForce(System *_system, F *_force, N *_net, SegSegList *_segseglist) : 
        system(_system), force(_force), net(_net), segseglist(_segseglist) {}
        
        KOKKOS_FORCEINLINE_FUNCTION
        void operator()(const int &i) const {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            
            auto list = segseglist->get_list(net);
            auto flag = segseglist->get_flag(net);
            if (!flag[i]) return;
            SegSeg segseg = list[i];
            
            SegSegForce fsegseg = force->segseg_force(system, net, segseg);
            
            if (segseglist->use_compute_map) {
                auto fsegseglist = segseglist->get_fsegseglist(net);
                fsegseglist[i] = fsegseg;
            } else {
                int n1 = segs[segseg.s1].n1;
                int n2 = segs[segseg.s1].n2;
                int n3 = segs[segseg.s2].n1;
                int n4 = segs[segseg.s2].n2;
                Kokkos::atomic_add(&nodes[n1].f, fsegseg.f1);
                Kokkos::atomic_add(&nodes[n2].f, fsegseg.f2);
                Kokkos::atomic_add(&nodes[n3].f, fsegseg.f3);
                Kokkos::atomic_add(&nodes[n4].f, fsegseg.f4);
            }
        }
    };
    
    template<class N>
    struct AddSegSegForceMap {
        System *system;
        N *net;
        SegSegList *segseglist;
        
        AddSegSegForceMap(System *_system, N *_net, SegSegList *_segseglist) : 
        system(_system), net(_net), segseglist(_segseglist) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const int &i) const {
            auto nodes = net->get_nodes();
            auto fsegseglist = segseglist->get_fsegseglist(net);
            auto map = segseglist->get_compute_map(net);
            auto flag = segseglist->get_flag(net);
            
            Vec3 fn(0.0);
            for (int j = map->beg(i); j < map->end(i); j++) {
                int k = map->segseg(j);
                if (!flag[k]) continue;
                SegSegForce fsegseg = fsegseglist[k];
                int fpos = map->fpos(j);
                fn += fsegseg[fpos];
            }
            nodes[i].f += fn;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const team_handle& team) const {
            int i = team.league_rank();
            
            auto nodes = net->get_nodes();
            auto fsegseglist = segseglist->get_fsegseglist(net);
            auto map = segseglist->get_compute_map(net);
            auto flag = segseglist->get_flag(net);
            
            Vec3 fn(0.0);
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, map->beg(i), map->end(i)), [&] (const int& j, Vec3& fsum) {
                int k = map->segseg(j);
                if (!flag[k]) return;
                SegSegForce fsegseg = fsegseglist[k];
                int fpos = map->fpos(j);
                fsum += fsegseg[fpos];
            }, fn);
            
            Kokkos::single(Kokkos::PerTeam(team), [=] () {
                nodes[i].f += fn;
            });
        }
    };
    
    void pre_compute(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].start();
        system->devtimer[TIMER_SEGSEGLIST].start();
        
        DeviceDisNet *net = system->get_device_network();
        segseglist->build_list<DeviceDisNet>(system, net);
        
        Kokkos::fence();
        system->devtimer[TIMER_SEGSEGLIST].stop();
        system->timer[system->TIMER_FORCE].stop();
    }
    
    void compute(System *system, bool zero=true)
    {
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].start();
        system->devtimer[TIMER_COMPUTE].start();
        
        DeviceDisNet *net = system->get_device_network();
        if (zero) zero_force(net);
        
        using policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<64,1>>;
        Kokkos::parallel_for("ForceSegSeg::compute", policy(0, segseglist->Nsegseg),
            AddSegSegForce<DeviceDisNet>(system, force, net, segseglist)
        );
        
        if (segseglist->use_compute_map) {
            // Assemble at nodes using the compute map
            Kokkos::fence();
            if (segseglist->Nsegseg < 20000) {
                Kokkos::parallel_for("ForceSegSeg::AddSegSegForceMap", net->Nnodes_local,
                    AddSegSegForceMap<DeviceDisNet>(system, net, segseglist)
                );
            } else {
            #if defined(KOKKOS_ENABLE_HIP)
                auto policy = Kokkos::TeamPolicy<>(net->Nnodes_local, 16);
            #else
                auto policy = Kokkos::TeamPolicy<>(net->Nnodes_local, Kokkos::AUTO);
            #endif
                Kokkos::parallel_for("ForceSegSeg::AddSegSegForceMap", policy,
                    AddSegSegForceMap<DeviceDisNet>(system, net, segseglist)
                );
            }
        }
        
        Kokkos::fence();
        system->devtimer[TIMER_COMPUTE].stop();
        system->timer[system->TIMER_FORCE].stop();
    }
    
    Vec3 node_force(System* system, const int& i)
    {
        SerialDisNet* net = system->get_serial_network();
        auto nodes = net->get_nodes();
        auto conn = net->get_conn();
        
        // Recompute neighbor list for now...
        NeighborBin* neighbor = generate_neighbor_segs(net, segseglist->cutoff, system->params.maxseg);
        
        auto neilist = neighbor->query(nodes[i].pos);
        
        Vec3 f(0.0);
        for (int j = 0; j < conn[i].num; j++) {
            int k = conn[i].seg[j];
            for (int l = 0; l < neilist.size(); l++) {
                int n = neilist[l];
                if (n == k) continue;
                SegSegForce fs = force->segseg_force(system, net, SegSeg(k, n), 1, 0);
                f += ((conn[i].order[j] == 1) ? fs.f1 : fs.f2);
            }
        }
        
        delete neighbor;
        
        return f;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& i, const team_handle& team)
    {
        Vec3 f(0.0);
        if (segseglist->cutoff <= 0.0) return f;
        
        auto conn = net->get_conn();
        int nconn = conn[i].num;
        
        // Here we need to get the neighbor list arrays from net
        // They contain the list of neighbor segments to a position
        auto count = net->get_count();
        auto nei = net->get_nei();
        int Nnei = count[i];
        
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nconn*Nnei), [&] (const int& t, Vec3& fsum) {
            int j = t / Nnei; // conn id
            int k = conn[i].seg[j];
            int l = t % Nnei; // nei id
            int n = nei(i,l); // neighbor seg
            
            if (n != k) {
                SegSegForce fs = force->segseg_force(system, net, SegSeg(k, n), 1, 0);
                fsum += ((conn[i].order[j] == 1) ? fs.f1 : fs.f2);
            }
        }, f);
        team.team_barrier();
        
        return f;
    }
    
    ~ForceSegSegList() {
        exadis_delete(segseglist);
        exadis_delete(force);
    }

    const char* name() { return "ForceSegSegList"; }
};

namespace ForceType {
    typedef ForceSegSegList<SegSegIso> FORCE_SEGSEG_ISO;
    typedef ForceCollection2<CORE_SELF_PKEXT,FORCE_SEGSEG_ISO> CUTOFF_MODEL;
}

} // namespace ExaDiS

#endif

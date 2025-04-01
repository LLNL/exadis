/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_H
#define EXADIS_FORCE_H

#include "system.h"

namespace ExaDiS {

struct SegForce {
    Vec3 f1, f2;
    KOKKOS_INLINE_FUNCTION SegForce() { f1 = f2 = Vec3(0.0); }
    KOKKOS_INLINE_FUNCTION SegForce(const Vec3 &_f1, const Vec3 &_f2) : f1(_f1), f2(_f2) {}
};

struct SegSeg {
    int s1, s2;
    KOKKOS_FORCEINLINE_FUNCTION SegSeg() {}
    KOKKOS_FORCEINLINE_FUNCTION SegSeg(int _s1, int _s2) : s1(_s1), s2(_s2) {}
};

struct SegSegForce {
    Vec3 f1, f2, f3, f4;
    KOKKOS_INLINE_FUNCTION SegSegForce() { f1 = f2 = f3 = f4 = Vec3(0.0); }
    KOKKOS_INLINE_FUNCTION SegSegForce(const Vec3 &_f1, const Vec3 &_f2, const Vec3 &_f3, const Vec3 &_f4) 
    : f1(_f1), f2(_f2), f3(_f3), f4(_f4) {}
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3& operator[](const int i) {
        if      (i == 0) return f1;
        else if (i == 1) return f2;
        else if (i == 2) return f3;
        else if (i == 3) return f4;
        else             return f1;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        Force
 *
 *-------------------------------------------------------------------------*/
class Force {
public:
    Force() {}
    Force(System *system) {}
    virtual void pre_compute(System *system) {}
    virtual void compute(System *system, bool zero=true) = 0;
    virtual Vec3 node_force(System *system, const int &i) = 0;
    virtual ~Force() {}
    virtual const char* name() { return "ForceNone"; }
    
    template<class N>
    void zero_force(N *net) {
        using policy = Kokkos::RangePolicy<typename N::ExecutionSpace>;
        Kokkos::parallel_for(policy(0, net->Nnodes_local), KOKKOS_LAMBDA(const int i) {
            auto nodes = net->get_nodes();
            nodes[i].f = Vec3(0.0);
        });
        Kokkos::fence();
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceCollection
 *                  Base class to define a collection (list) of several
 *                  force contributions that need to be summed up
 *                  to define the total force.
 *
 *-------------------------------------------------------------------------*/
class ForceCollection : public Force {
private:
    std::vector<Force*> forces;

public:
    ForceCollection(System *system, std::vector<Force*> forcelist) {
        forces = forcelist;
    }
    
    void pre_compute(System *system) {
        for (auto force : forces)
            force->pre_compute(system);
    }
    
    void compute(System *system, bool zero=true) {
        auto net = system->get_device_network();
        if (zero) zero_force(net);
        for (auto force : forces)
            force->compute(system, false);
    }
    
    Vec3 node_force(System *system, const int &i) {
        Vec3 f(0.0);
        for (auto force : forces)
            f += force->node_force(system, i);
        return f;
    }
    
    ~ForceCollection() {
        for (auto force : forces)
            if (force) delete force;
    }
    
    const char* name() {
        std::string name = "ForceCollection = {";
        std::vector<std::string> a;
        for (int i = 0; i < forces.size(); i++) {
            name += forces[i]->name();
            name += (i < forces.size()-1) ? "," : "}";
        }
        //return name.c_str();
        return "ForceCollection";
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceCollection2
 *                  Templated version of the ForceCollection base class
 *                  with two force contributions. This is mainly for use
 *                  with the TopologyParallel module which cannot support
 *                  the base ForceCollection in the current implementation.
 *
 *-------------------------------------------------------------------------*/
template<class F1, class F2>
class ForceCollection2 : public Force {
public:
    F1* force1;
    F2* force2;
    
    typedef typename F1::Params F1params;
    typedef typename F2::Params F2params;

public:
    struct Params {
        F1params f1params;
        F2params f2params;
        Params() { f1params=F1params(); f2params=F2params(); }
        Params(F1params _f1params, F2params _f2params) : f1params(_f1params), f2params(_f2params) {}
    };
    
    ForceCollection2(System* system, Params params) {
        force1 = exadis_new<F1>(system, params.f1params);
        force2 = exadis_new<F2>(system, params.f2params);
    }
    
    ForceCollection2(System* system, F1params f1params=F1params(), F2params f2params=F2params()) {
        force1 = exadis_new<F1>(system, f1params);
        force2 = exadis_new<F2>(system, f2params);
    }
    
    void pre_compute(System *system) {
        force1->pre_compute(system);
        force2->pre_compute(system);
    }
    
    void compute(System *system, bool zero=true) {
        auto net = system->get_device_network();
        if (zero) zero_force(net);
        force1->compute(system, false);
        force2->compute(system, false);
    }
    
    Vec3 node_force(System *system, const int &i) {
        Vec3 f(0.0);
        f += static_cast<Force*>(force1)->node_force(system, i);
        f += static_cast<Force*>(force2)->node_force(system, i);
        return f;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& i, const team_handle& team)
    {
        Vec3 f(0.0);
        f += force1->node_force(system, net, i, team);
        f += force2->node_force(system, net, i, team);
        return f;
    }
    
    ~ForceCollection2() {
        exadis_delete(force1);
        exadis_delete(force2);
    }
    
    const char* name() { return "ForceCollection2"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceLongShort
 *                  Base class for elastic for contributions partitionned 
 *                  between a long-range (computed with a grid method) and
 *                  a short-range (explicit seg/seg sum) contributions.
 *
 *-------------------------------------------------------------------------*/
template<class FLong, class FShort>
class ForceLongShort : public Force {
protected:
    FLong *flong;
    FShort *fshort;

public:
    typedef typename FLong::Params Params;
    
    ForceLongShort(System *system, Params params) {
        // Long-range
        flong = exadis_new<FLong>(system, params);
        // Short-range
        fshort = exadis_new<FShort>(system, flong);
    }
    
    ForceLongShort(System *system, int Ngrid) {
        // Long-range
        flong = exadis_new<FLong>(system, Params(Ngrid));
        // Short-range
        fshort = exadis_new<FShort>(system, flong);
    }
    
    ForceLongShort(System *system, int Nx, int Ny, int Nz) {
        // Long-range
        flong = exadis_new<FLong>(system, Params(Nx, Ny, Nz));
        // Short-range
        fshort = exadis_new<FShort>(system, flong);
    }
    
    virtual void pre_compute(System *system) {
        flong->pre_compute(system);
        fshort->pre_compute(system);
    }
    
    virtual void compute(System *system, bool zero=true) {
        auto net = system->get_device_network();
        if (zero) zero_force(net);
        flong->compute(system, false);
        fshort->compute(system, false);
    }
    
    virtual Vec3 node_force(System *system, const int &i) {
        Vec3 f(0.0);
        f += flong->node_force(system, i);
        f += fshort->node_force(system, i);
        return f;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& i, const team_handle& team)
    {
        Vec3 f(0.0);
        f += flong->node_force(system, net, i, team);
        f += fshort->node_force(system, net, i, team);
        return f;
    }
    
    virtual ~ForceLongShort() {
        exadis_delete(flong);
        exadis_delete(fshort);
    }
    
    const char* name() { return "ForceLongShort"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        ForceSeg
 *                  Base class for force contributions that are computed by
 *                  assembling nodal forces from (arm) segment forces.
 *                  The class is instantiated with a force kernel <F> in
 *                  which a segment_force() function must be defined.
 *
 *-------------------------------------------------------------------------*/
template <class F>
class ForceSeg : public Force {
public:
    typedef typename F::Params Params;
    F *force; // force kernel
    
    ForceSeg(System *system, Params params=Params()) {
        force = exadis_new<F>(system, params);
    }
    
    template<class N>
    struct AddSegmentForce {
        System *system;
        F *force;
        N *net;
        AddSegmentForce(System *_system, F *_force, N *_net) : system(_system), force(_force), net(_net) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const int &i) const {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            int n1 = segs[i].n1;
            int n2 = segs[i].n2;
            
            SegForce fseg = force->segment_force(system, net, i);
            
            Kokkos::atomic_add(&nodes[n1].f, fseg.f1);
            Kokkos::atomic_add(&nodes[n2].f, fseg.f2);
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const team_handle& team) const {
            int i = team.league_rank(); // seg id
            
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            int n1 = segs[i].n1;
            int n2 = segs[i].n2;
            
            SegForce fseg = force->segment_force(system, net, i, team);
            
            Kokkos::atomic_add(&nodes[n1].f, fseg.f1);
            Kokkos::atomic_add(&nodes[n2].f, fseg.f2);
        }
    };
    
    void pre_compute(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].start();
        
        if constexpr (F::has_pre_compute)
            force->pre_compute(system);
        
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].stop();
    }
    
    void compute(System *system, bool zero=true)
    {
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].start();
        
        DeviceDisNet *net = system->get_device_network();
        if (zero) zero_force(net);
        
        if constexpr (F::has_compute_team) {
            Kokkos::parallel_for(Kokkos::TeamPolicy<>(net->Nsegs_local, Kokkos::AUTO),
                AddSegmentForce<DeviceDisNet>(system, force, net)
            );
        } else {
            using policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<64,1>>;
            Kokkos::parallel_for(policy(0, net->Nsegs_local), AddSegmentForce<DeviceDisNet>(system, force, net));
        }
        
        Kokkos::fence();
        system->timer[system->TIMER_FORCE].stop();
    }
    
    Vec3 node_force(System *system, const int &i)
    {
        SerialDisNet *network = system->get_serial_network();
        
        Vec3 f(0.0);
        
        if constexpr (F::has_node_force) {
            f = force->node_force(system, network, i);
        } else {
            auto conn = network->get_conn();
            
            for (int j = 0; j < conn[i].num; j++) {
                int k = conn[i].seg[j];
                SegForce fs = force->segment_force(system, network, k);
                f += ((conn[i].order[j] == 1) ? fs.f1 : fs.f2);
            }
        }
        
        return f;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& i, const team_handle& team)
    {
        Vec3 f(0.0);
        
        if constexpr (F::has_node_force) {
            f = force->node_force(system, net, i, team);
        } else {
            auto nodes = net->get_nodes();
            auto conn = net->get_conn();
            
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, conn[i].num), [&] (const int& j, Vec3& fsum) {
                int k = conn[i].seg[j];
                SegForce fs = force->segment_force(system, net, k);
                fsum += ((conn[i].order[j] == 1) ? fs.f1 : fs.f2);
            }, f);
            team.team_barrier();
        }
        
        return f;
    }
    
    ~ForceSeg() {
        exadis_delete(force);
    }
    
    const char* name() { return F::name; }
};

} // namespace ExaDiS


// Available force types
#include "force_common.h"
#include "force_iso.h"
#include "force_core.h"
#include "force_lt.h"
#include "force_n2.h"
#include "force_segseglist.h"
#include "force_fft.h"

#endif

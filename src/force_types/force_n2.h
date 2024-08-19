/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_N2_H
#define EXADIS_FORCE_N2_H

#include <Kokkos_ScatterView.hpp>
#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:   ForceSegN2
 *              Brute-force N^2 (pair interactions) force kernel where
 *              the force on each segment is computed by summing the elastic
 *              interactions with all other segments in the primary volume
 *              (closest PBC images).
 *              Does not include the self-stress.
 *
 *-------------------------------------------------------------------------*/
template<class F>
struct ForceSegN2
{
    static const bool has_pre_compute = false;
    static const bool has_compute_team = true;
    static const bool has_node_force = true;
    
    typedef typename F::Params Params;
    
    F *force;
    double MU, NU, a;
    
    ForceSegN2(System *system, Params &params)
    {
        MU = system->params.MU;
        NU = system->params.NU;
        a = system->params.a;
        
        force = exadis_new<F>(system, params);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    SegForce segment_force(System *system, N *net, const int &i) 
    {
        int Nsegs = net->Nsegs_local;

        Vec3 fs1(0.0), fs2(0.0);
        for (int j = 0; j < Nsegs; j++) {
            if (j == i) continue; // skip self-force
            
            SegSegForce fs = force->segseg_force(system, net, SegSeg(i, j), 1, 0);
            fs1 += fs.f1;
            fs2 += fs.f2;
        }
        
        return SegForce(fs1, fs2);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    SegForce segment_force(System* system, N* net, const int& i, const team_handle& team)
    {
        int Nsegs = net->Nsegs_local;
        
        Vec3 fs1(0.0), fs2(0.0);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, Nsegs), [&] (const int& j, Vec3& fs1sum, Vec3& fs2sum) {
            if (j != i) { // skip self-force
                SegSegForce fs = force->segseg_force(system, net, SegSeg(i, j), 1, 0);
                fs1sum += fs.f1;
                fs2sum += fs.f2;
            }
        }, fs1, fs2);
        team.team_barrier();
        
        return SegForce(fs1, fs2);
    }
    
    template<class N, typename ScatterViewType>
    struct NodeForce {
        System *system;
        F *force;
        N *net;
        ScatterViewType s_f;
        int i;
        
        NodeForce(System *_system, F *_force, N *_net, 
                  ScatterViewType &_s_f, int _i) : 
        system(_system), force(_force), net(_net), s_f(_s_f), i(_i) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const int &p) const {
            auto access = s_f.access();
            
            auto conn = net->get_conn();
            
            int j = p / net->Nsegs_local;
            int o = conn[i].order[j];
            int s1 = conn[i].seg[j];
            int s2 = p % net->Nsegs_local;
            
            if (s1 != s2) {
                SegSegForce fs = force->segseg_force(system, net, SegSeg(s1, s2), 1, 0);
                Vec3 fl = (o == 1) ? fs.f1 : fs.f2;
                access(0) += fl;
            }
        }
    };
    
    Vec3 node_force(System *system, SerialDisNet *net, const int &i) 
    {
        auto conn = net->get_conn();
        
        if (1) {
            
            // Use host execution space
            Kokkos::View<Vec3*, Kokkos::HostSpace> f("f", 1);
            Kokkos::deep_copy(f, 0.0);
            
            typedef typename Kokkos::View<Vec3*, Kokkos::HostSpace>::array_layout layout;
            typedef Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace> device;
            typedef Kokkos::Experimental::ScatterView<Vec3*, layout, device> Sview;
            Sview s_f(f);
            
            using policy = Kokkos::RangePolicy<typename Kokkos::DefaultHostExecutionSpace>;
            Kokkos::parallel_for("ForceSegN2::NodeForce", policy(0, conn[i].num*net->Nsegs_local),
                NodeForce<SerialDisNet,Sview>(system, force, net, s_f, i)
            );
            
            Kokkos::Experimental::contribute(f, s_f);
            
            return f(0);
            
        } else {
            
            // Use device execution space
            DeviceDisNet *d_net = system->get_device_network();
            system->net_mngr->set_active(DisNetManager::SERIAL_ACTIVE);
            
            Kokkos::View<Vec3*> f("f", 1);
            Kokkos::deep_copy(f, 0.0);
            typedef Kokkos::Experimental::ScatterView<Vec3*> Sview;
            Sview s_f(f);
            
            Kokkos::parallel_for("ForceSegN2::NodeForce", conn[i].num*net->Nsegs_local,
                NodeForce<DeviceDisNet,Sview>(system, force, d_net, s_f, i)
            );
            
            Kokkos::Experimental::contribute(f, s_f);
            
            auto h_f = Kokkos::create_mirror_view(f);
            Kokkos::deep_copy(h_f, f);
            
            return h_f(0);
        }
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_force(System* system, N* net, const int& i, const team_handle& team)
    {
        auto nodes = net->get_nodes();
        auto conn = net->get_conn();
        
        int nconn = conn[i].num;
        int Nsegs = net->Nsegs_local;
        
        Vec3 f;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nconn*Nsegs), [&] (const int& t, Vec3& fsum) {
            int n = t % Nsegs; // seg id
            int j = t / Nsegs; // conn id
            int k = conn[i].seg[j];
            if (k != n) {
                SegSegForce fs = force->segseg_force(system, net, SegSeg(k, n), 1, 0);
                fsum += ((conn[i].order[j] == 1) ? fs.f1 : fs.f2);
            }
        }, f);
        team.team_barrier();
        
        return f;
    }
    
    ~ForceSegN2() {
        exadis_delete(force);
    }
    
    static constexpr const char* name = "ForceSegN2";
};

namespace ForceType {
    typedef ForceSeg<ForceSegN2<SegSegIso> > BRUTE_FORCE_N2;
    typedef ForceCollection2<CORE_SELF_PKEXT,BRUTE_FORCE_N2> N2_MODEL;
}

} // namespace ExaDiS

#endif

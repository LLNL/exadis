/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_H
#define EXADIS_MOBILITY_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Mobility
 *
 *-------------------------------------------------------------------------*/
class Mobility {
public:
    bool non_linear = false;
    Mobility() {}
    Mobility(System *system) {}
    virtual void compute(System *system) = 0;
    virtual Vec3 node_velocity(System *system, const int &i, const Vec3 &fi) = 0;
    virtual ~Mobility() {}
    virtual const char* name() { return "MobilityNone"; }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityLocal
 *                  Base class for local types of mobilities in which the
 *                  node velocity is computed by looping over its arm and
 *                  summing a drag contribution.
 *
 *-------------------------------------------------------------------------*/
template <class M>
class MobilityLocal : public Mobility {
public:
    typedef M Mob;
    typedef typename M::Params Params;
    M *mob; // mobility kernel
    
    MobilityLocal(System *system, Params params) {
        mob = exadis_new<M>(system, params);
        non_linear = mob->non_linear;
    }
    
    template<class N>
    struct NodeMobility {
        System *system;
        M *mob;
        N *net;
        NodeMobility(System *_system, M *_mob, N *_net) : system(_system), mob(_mob), net(_net) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const int &i) const {
            auto nodes = net->get_nodes();
            nodes[i].v = mob->node_velocity(system, net, i, nodes[i].f);
        }
    };
    
    void compute(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_MOBILITY].start();
        
        DeviceDisNet *net = system->get_device_network();
        using policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<32,1>>;
        Kokkos::parallel_for(policy(0, net->Nnodes_local), NodeMobility<DeviceDisNet>(system, mob, net));
        
        Kokkos::fence();
        system->timer[system->TIMER_MOBILITY].stop();
    }
    
    Vec3 node_velocity(System *system, const int &i, const Vec3 &fi)
    {
        SerialDisNet *network = system->get_serial_network();
        return mob->node_velocity(system, network, i, fi);
    }
    
    ~MobilityLocal() {
        exadis_delete(mob);
    }
    
    const char* name() { return M::name; }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     apply_velocity_cap
 *
 *-------------------------------------------------------------------------*/
KOKKOS_FORCEINLINE_FUNCTION
void apply_velocity_cap(const double &vmax, const double &vscale, Vec3 &v)
{
    if (vmax <= 0.0) return;
    double vmag = v.norm() * vscale; // m/s
    if (vmag < 1e-5) return;
    double alpha = 10.0;
    double vcap = vmag / pow(1.0 + pow(vmag/vmax, alpha), 1.0/alpha);
    v = (vcap / vmag) * v;
}

} // namespace ExaDiS


// Available mobility types
#include "mobility_glide.h"
#include "mobility_bcc0b.h"
#include "mobility_fcc0.h"
#include "mobility_fcc0_fric.h"
#include "mobility_fcc0b.h"
#include "mobility_bcc_nl.h"

#endif

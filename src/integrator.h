/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INTEGRATOR_H
#define EXADIS_INTEGRATOR_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Integrator
 *
 *-------------------------------------------------------------------------*/
class Integrator {
public:
    double nextdt;
public:
    Integrator() {}
    Integrator(System* system) {}
    virtual void integrate(System* system) {}
    virtual KOKKOS_FUNCTION ~Integrator() {}
    virtual const char* name() { return "IntegratorNone"; }
    
    // Restart
    virtual void write_restart(FILE* fp) {
        fprintf(fp, "nextdt %.17g\n", nextdt);
    }
    virtual void read_restart(FILE* fp) {
        fscanf(fp, "nextdt %lf\n", &nextdt);
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorEuler
 *
 *-------------------------------------------------------------------------*/
class IntegratorEuler : public Integrator {    
public:
    IntegratorEuler(System* system) {
        nextdt = system->params.nextdt;
    }
    
    void integrate(System* system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].start();
        
        double dt = nextdt;
        
        DeviceDisNet* net = system->get_device_network();
        
        Kokkos::resize(system->xold, net->Nnodes_local);
        
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int i) {
            auto nodes = net->get_nodes();
            auto cell = net->cell;
            
            system->xold(i) = nodes[i].pos;
            Vec3 rnew = nodes[i].pos + dt*nodes[i].v;
            nodes[i].pos = cell.pbc_fold(rnew);
        });
        
        system->realdt = dt;
        
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].stop();
    }
    
    const char* name() { return "IntegratorEuler"; }
};

} // namespace ExaDiS


#include "integrator_trapezoid.h"
#include "integrator_rkf.h"
#include "integrator_multi.h"
#include "integrator_subcycling.h"

#endif

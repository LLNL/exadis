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
protected:
    double nextdt;
public:
    Integrator() {}
    Integrator(System *system) {}
    virtual void integrate(System *system) {}
    virtual ~Integrator() {}
    virtual const char* name() { return "IntegratorNone"; }
    
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
    IntegratorEuler(System *system) {
        nextdt = system->params.nextdt;
    }
    
    void integrate(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].start();
        
        double dt = nextdt;
        
        auto network = system->get_device_network();
        
        Kokkos::resize(system->xold, network->Nnodes_local);
        
        Kokkos::parallel_for(network->Nnodes_local, KOKKOS_LAMBDA(const int i)
        {
            system->xold(i) = network->nodes(i).pos;
            Vec3 rnew = network->nodes(i).pos + dt*network->nodes(i).v;
            network->nodes(i).pos = network->cell.pbc_fold(rnew);
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

/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INTEGRATOR_TRAPEZOID_H
#define EXADIS_INTEGRATOR_TRAPEZOID_H

#include "integrator.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorTrapezoid
 *
 *-------------------------------------------------------------------------*/
class IntegratorTrapezoid : public Integrator {
private:
    Force *force;
    Mobility *mobility;
    
    double newdt, currdt, maxdt;
    double rtol, errormax;
    double dtIncrementFact, dtDecrementFact, dtVariableAdjustment, dtExponent;
    
    System *s;
    DeviceDisNet *network;
    T_v vcurr;
    Kokkos::View<double*> errmax;
    
public:
    struct Params {};
    
    IntegratorTrapezoid(System *system, Force *_force, Mobility *_mobility, Params params=Params()) : 
    force(_force), mobility(_mobility)
    {
        nextdt = system->params.nextdt;
        maxdt = system->params.maxdt;
        rtol = system->params.rtol;
        
        dtIncrementFact = 1.2;
        dtDecrementFact = 0.5;
        dtVariableAdjustment = 0;
        dtExponent = 4.0;
    }
    
    struct TagPreserveData {};
    struct TagAdvanceNodes {};
    struct TagErrorNodes {};
    struct TagRestoreNodes {};
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagPreserveData, const int &i) const {
        auto nodes = network->get_nodes();
        s->xold(i) = nodes[i].pos;
        vcurr(i) = nodes[i].v;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagAdvanceNodes, const int &i) const {
        auto nodes = network->get_nodes();
        auto cell = network->cell;
        
        Vec3 pos = s->xold(i) + 0.5 * currdt * (vcurr(i) + vcurr(i));
        nodes[i].pos = cell.pbc_fold(pos);
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagErrorNodes, const int &i) const {
        auto nodes = network->get_nodes();
        auto cell = network->cell;
        
        Vec3 xold = s->xold(i);
        xold = cell.pbc_position(nodes[i].pos, xold);
        
        Vec3 verr = nodes[i].pos - xold - (0.5 * newdt * (nodes[i].v + vcurr(i)));
        double err = 0.0;
        err = fmax(err, fabs(verr.x));
        err = fmax(err, fabs(verr.y));
        err = fmax(err, fabs(verr.z));
        Kokkos::atomic_max(&errmax(0), err);
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagRestoreNodes, const int &i) const {
        auto nodes = network->get_nodes();
        auto cell = network->cell;
        
        Vec3 x = nodes[i].pos;
        Vec3 xold = s->xold(i);
        xold = cell.pbc_position(x, xold);
        Vec3 dx = x - xold - (0.5 * newdt * (nodes[i].v + vcurr(i)));
        Vec3 pos = x - dx;
        nodes[i].pos = cell.pbc_fold(pos);
    }
    
    void integrate(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].start();
        
        newdt = fmin(maxdt, nextdt);
        if (newdt <= 0.0) newdt = maxdt;
        
        s = system;
        network = system->get_device_network();
        
        // Save nodal data
        Kokkos::resize(system->xold, network->Nnodes_local);
        Kokkos::resize(vcurr, network->Nnodes_local);
        Kokkos::parallel_for("IntegratorTrapezoid::PreserveData",
            Kokkos::RangePolicy<TagPreserveData>(0, network->Nnodes_local), *this
        );
        Kokkos::fence();

        int convergent = 0;
        int maxIterations = 2;
        int incrDelta = 1;
        
        while (!convergent) {
            
            // Advance nodes
            currdt = newdt;
            Kokkos::parallel_for("IntegratorTrapezoid::AdvanceNodes",
                Kokkos::RangePolicy<TagAdvanceNodes>(0, network->Nnodes_local), *this
            );
            Kokkos::fence();
            
            for (int iter = 0; iter < maxIterations; iter++) {
                
                force->compute(system);
                mobility->compute(system);
                
                errmax = Kokkos::View<double*>("IntegratorTrapezoid:errmax", 1);
                Kokkos::parallel_for("IntegratorTrapezoid::ErrorNodes",
                    Kokkos::RangePolicy<TagErrorNodes>(0, network->Nnodes_local), *this
                );
                Kokkos::fence();
                auto h_errmax = Kokkos::create_mirror_view(errmax);
                Kokkos::deep_copy(h_errmax, errmax);
                errormax = h_errmax(0);
                
                if (errormax < rtol) {
                    convergent = 1;
                    break;
                } else {
                    incrDelta = 0;
                    if (iter == maxIterations-1) continue;
                    
                    Kokkos::parallel_for("IntegratorTrapezoid::RestoreNodes",
                        Kokkos::RangePolicy<TagRestoreNodes>(0, network->Nnodes_local), *this
                    );
                    Kokkos::fence();
                }
                
            } // for (iter = 0; ...)
            
            if (!convergent) {
                newdt *= dtDecrementFact;
                
                if ((newdt < 1.0e-20) /*&& (system->proc_rank == 0)*/)
                    ExaDiS_fatal("IntegratorTrapezoid(): Timestep has dropped below\n"
                                 "minimal threshold to %e. Aborting!", newdt);
            }
            
        } // while (!convergent)
        
        system->realdt = newdt;
        
        if (incrDelta) {
            if (dtVariableAdjustment) {
                double tmp1, tmp2, tmp3, tmp4, factor;
                tmp1 = pow(dtIncrementFact, dtExponent);
                tmp2 = errormax/rtol;
                tmp3 = 1.0 / dtExponent;
                tmp4 = pow(1.0/(1.0+(tmp1-1.0)*tmp2), tmp3);
                factor = dtIncrementFact * tmp4;
                nextdt = fmin(maxdt, newdt*factor);
            } else {
                nextdt = fmin(maxdt, newdt*dtIncrementFact);
            }
        } else {
            nextdt = newdt;
        }
        
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].stop();
    }
    
    const char* name() { return "IntegratorTrapezoid"; }
};
    
} // namespace ExaDiS

#endif

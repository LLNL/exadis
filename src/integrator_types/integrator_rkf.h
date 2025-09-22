/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INTEGRATOR_RKF_H
#define EXADIS_INTEGRATOR_RKF_H

#include "integrator.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorRKF
 *
 *-------------------------------------------------------------------------*/
class IntegratorRKF : public Integrator {
protected:
    Force* force;
    Mobility* mobility;
    
    double newdt, maxdt;
    double rtol, rtolth, rtolrel;
    double dtIncrementFact, dtDecrementFact, dtVariableAdjustment, dtExponent;
    
    System* s;
    DeviceDisNet* network;
    T_v rkf[6];
    T_v vcurr;
    double errmax[2];
    int incrDelta, iTry;
    int step;
    
    // Coefficients for error calculation
    const double er[6] = {
        1.0/360, 0.0, -128.0/4275, -2197.0/75240, 1.0/50, 2.0/55
    };
    
    // Array of coefficients used by the method.
    const double f[6][6] = {
        {1.0/4      ,  0.0        ,  0.0         ,  0.0          ,  0.0    , 0.0   },
        {3.0/32     ,  9.0/32     ,  0.0         ,  0.0          ,  0.0    , 0.0   },
        {1932.0/2197, -7200.0/2197,  7296.0/2197 ,  0.0          ,  0.0    , 0.0   },
        {439.0/216  , -8.0        ,  3680.0/513  , -845.0/4104   ,  0.0    , 0.0   },
        {-8.0/27    ,  2.0        , -3544.0/2565 ,  1859.0/4104  , -11.0/40, 0.0   },
        {16.0/135   ,  0.0        ,  6656.0/12825,  28561.0/56430, -9.0/50 , 2.0/55}
    };
    
public:
    struct Params {
        double rtolth, rtolrel;
        Params() { rtolth = 0.1; rtolrel = 0.01; }
        Params(double _rtolth, double _rtolrel) : rtolth(_rtolth), rtolrel(_rtolrel) {}
    };
    
    IntegratorRKF(System *system, Force *_force, Mobility *_mobility, Params params=Params()) : 
    force(_force), mobility(_mobility)
    {
        nextdt = system->params.nextdt;
        maxdt = system->params.maxdt;
        rtol = system->params.rtol;
        rtolth = params.rtolth;
        rtolrel = params.rtolrel;
        
        if (rtol < 0.0 || rtolth < 0.0 || rtolrel < 0.0)
            ExaDiS_fatal("Error: invalid rtol values in IntegratorRKF\n");
        
        dtIncrementFact = 1.2;
        dtDecrementFact = 0.5;
        dtVariableAdjustment = 0;
        dtExponent = 4.0;
    }
    
    struct TagPreserveData {};
    struct TagRKFStep {};
    struct TagErrorNodes {};
    struct TagRestoreVels {};
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagPreserveData, const int &i) const {
        auto nodes = network->get_nodes();
        s->xold(i) = nodes[i].pos;
        vcurr(i) = nodes[i].v;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagRKFStep, const int &i) const {
        auto nodes = network->get_nodes();
        auto cell = network->cell;
        
        if (step < 5) rkf[step](i) = nodes[i].v;
        Vec3 pos(0.0);
        for (int j = 0; j < step+1; j++)
            pos += f[step][j] * rkf[j](i);
        pos = s->xold(i) + newdt*pos;
        nodes[i].pos = cell.pbc_fold(pos);
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagErrorNodes, const int& i, double& emax0, double& emax1, int& errnans) const {
        auto nodes = network->get_nodes();
        auto cell = network->cell;
        
        rkf[5](i) = nodes[i].v;
        
        Vec3 err(0.0);
        for (int j = 0; j < 6; j++)
            err += er[j] * rkf[j](i);
        err = newdt * err;
        double errnet = err.norm();
        if (errnet > emax0) emax0 = errnet;
        
        Vec3 xold = s->xold(i);
        xold = cell.pbc_position(nodes[i].pos, xold);
        Vec3 dr = nodes[i].pos - xold;
        double drn = dr.norm();
        double relerr = 0.0;
        if (errnet > rtolth) {
            if (drn > rtolth/rtolrel) {
                relerr = errnet/drn;
            } else {
                relerr = 2*rtolrel;
            }
        }
        if (relerr > emax1) emax1 = relerr;
        if (std::isnan(drn)) errnans++;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (TagRestoreVels, const int &i) const {
        auto nodes = network->get_nodes();
        nodes[i].v = vcurr(i);
    }
    
    virtual inline void rkf_step(int i)
    {
        step = i;
        Kokkos::parallel_for("IntegratorRKF::RKFStep",
            Kokkos::RangePolicy<TagRKFStep>(0, network->Nnodes_local), *this
        );
        Kokkos::fence();
        
        if (i < 5) {
            force->compute(s);
            mobility->compute(s);
        }
    }
    
    virtual inline void compute_error()
    {
        int errnans = 0;
        Kokkos::parallel_reduce("IntegratorRKF::ErrorNodes",
            Kokkos::RangePolicy<TagErrorNodes>(0, network->Nnodes_local), *this,
            Kokkos::Max<double>(errmax[0]), Kokkos::Max<double>(errmax[1]), errnans
        );
        Kokkos::fence();
        
        if (errnans > 0)
            ExaDiS_fatal("Error: %d NaNs found during integration\n", errnans);
    }
    
    virtual inline void non_convergent()
    {
        // We need to start from the old velocities. So, first,
        // substitute them with the old ones.
        Kokkos::parallel_for("IntegratorRKF::RestoreVels",
            Kokkos::RangePolicy<TagRestoreVels>(0, network->Nnodes_local), *this
        );
        Kokkos::fence();
        
        incrDelta = 0;
        newdt *= dtDecrementFact;
        
        if ((newdt < 1.0e-20) /*&& (system->proc_rank == 0)*/)
            ExaDiS_fatal("IntegratorRKF(): Timestep has dropped below\n"
                         "minimal threshold to %e. Aborting!\n", newdt);
    }
    
    virtual void integrate(System* system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].start();
        
        newdt = fmin(maxdt, nextdt);
        if (newdt <= 0.0) newdt = maxdt;
        
        s = system;
        network = system->get_device_network();
        
        for (int i = 0; i < 6; i++)
            Kokkos::resize(rkf[i], network->Nnodes_local);
        
        // Save nodal data
        Kokkos::resize(system->xold, network->Nnodes_local);
        Kokkos::resize(vcurr, network->Nnodes_local);
        Kokkos::parallel_for("IntegratorRKF::PreserveData",
            Kokkos::RangePolicy<TagPreserveData>(0, network->Nnodes_local), *this
        );
        Kokkos::fence();

        int convergent = 0;
        incrDelta = 1;
        iTry = -1;
        
        while (!convergent) {
            iTry++;
            
            // Apply the Runge-Kutta-Fehlberg integrator one step at a time
            for (int i = 0; i < 5; i++) {
                rkf_step(i);
            }
            
            // Calculate the error
            compute_error();
            
            // If the error is within the tolerance, we've reached
            // convergence so we can accept this dt. Otherwise
            // reposition the nodes and try again.
            if (errmax[0] < rtol && errmax[1] < rtolrel) {
                // Calculate final positions
                rkf_step(5);
                convergent = 1;
            }
            
            if (!convergent) {
                non_convergent();
            }
            
        } // while (!convergent)
        
        system->realdt = newdt;
        
        if (incrDelta) {
            if (dtVariableAdjustment) {
                double tmp1, tmp2, tmp3, tmp4, factor;
                tmp1 = pow(dtIncrementFact, dtExponent);
                tmp2 = errmax[0]/rtol;
                tmp3 = 1.0 / dtExponent;
                tmp4 = pow(1.0/(1.0+(tmp1-1.0)*tmp2), tmp3);
                factor = dtIncrementFact * tmp4;
                newdt = fmin(maxdt, newdt*factor);
            } else {
                newdt = fmin(maxdt, newdt*dtIncrementFact);
            }
        }
        nextdt = newdt;
        
        Kokkos::fence();
        system->timer[system->TIMER_INTEGRATION].stop();
    }
    
    KOKKOS_FUNCTION ~IntegratorRKF() {}
    
    const char* name() { return "IntegratorRKF"; }
};
    
} // namespace ExaDiS

#endif

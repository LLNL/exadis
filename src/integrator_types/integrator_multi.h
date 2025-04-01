/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INTEGRATOR_MULTI_H
#define EXADIS_INTEGRATOR_MULTI_H

#include "integrator.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        IntegratorMulti
 *                  Base class to integrate the system by calling a base
 *                  integrator <I> multiple times in a row for a number 
 *                  of subcycles <maxsubcyle> or until a maximum time step
 *                  size <maxdt> is reached.
 *
 *-------------------------------------------------------------------------*/
template<class I>
class IntegratorMulti : public Integrator {
private:
    I* integrator;
    Force* force;
    Mobility* mobility;
    
    int maxsubcyle;
    double maxdt;
    
    T_x xold;
    
    typedef typename I::Params IParams;

public:
    struct Params {
        int maxsubcyle;
        IParams Iparams = IParams();
        Params() { maxsubcyle = 10; }
        Params(int _maxsubcyle) : maxsubcyle(_maxsubcyle) {}
        Params(IParams _Iparams, int _maxsubcyle) : Iparams(_Iparams), maxsubcyle(_maxsubcyle) {}
    };
    
    IntegratorMulti(System* system, Force* _force, Mobility* _mobility, int _maxsubcyle=10) :
    force(_force), mobility(_mobility) {
        maxdt = system->params.maxdt;
        maxsubcyle = _maxsubcyle;
        integrator = exadis_new<I>(system, force, mobility);
    }
    
    IntegratorMulti(System* system, Force* _force, Mobility* _mobility, Params params=Params()) :
    force(_force), mobility(_mobility) {
        maxdt = system->params.maxdt;
        maxsubcyle = params.maxsubcyle;
        integrator = exadis_new<I>(system, force, mobility, params.Iparams);
    }
    
    IntegratorMulti(const IntegratorMulti&) = delete;
    
    // Use a functor so that we can safely call the destructor
    // to free the base integrator at the end of the run
    struct SavePositions {
        DeviceDisNet* net;
        T_x xold;
        SavePositions(DeviceDisNet* _net, T_x& _xold) : net(_net), xold(_xold) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const {
            auto nodes = net->get_nodes();
            xold(i) = nodes[i].pos;
        }
    };
    
    void integrate(System* system)
    {
        // Save initial positions
        DeviceDisNet* network = system->get_device_network();
        Kokkos::resize(xold, network->Nnodes_local);
        Kokkos::parallel_for(network->Nnodes_local, SavePositions(network, xold));
        Kokkos::fence();
        
        // Integrate the system by calling the base 
        // integrator multiple times in a row
        double totdt = 0.0;
        int isubcycle = 0;
        while (totdt < maxdt) {
            
            if constexpr(!std::is_same<I,IntegratorTrapezoid>::value) {
                if (isubcycle > 0) {
                    force->compute(system);
                    mobility->compute(system);
                }
            }
            integrator->integrate(system);
            
            totdt += system->realdt;
            isubcycle++;
            if (isubcycle >= maxsubcyle) break;
        }
        system->realdt = totdt;
        
        // Restore initial positions as old positions
        // for plastic strain calculation and collisions
        Kokkos::deep_copy(system->xold, xold);
    }
    
    void write_restart(FILE* fp) { integrator->write_restart(fp); }
    void read_restart(FILE* fp) { integrator->read_restart(fp); }
    
    KOKKOS_FUNCTION ~IntegratorMulti() {
        KOKKOS_IF_ON_HOST((
            exadis_delete(integrator);
        ))
    }
    
    const char* name() { return "IntegratorMulti"; }
};

} // namespace ExaDiS

#endif

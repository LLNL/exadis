/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_COLLISION_RETROACTIVE_H
#define EXADIS_COLLISION_RETROACTIVE_H

#include "collision.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        CollisionRetroactive
 *
 *-------------------------------------------------------------------------*/
class CollisionRetroactive : public Collision {
public:
    CollisionRetroactive(System *system) {}
    
    void retroactive_collision(System* system);
    void retroactive_collision_parallel(System* system);
    
    void handle(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].start();
        
        //retroactive_collision(system);
        retroactive_collision_parallel(system);
        
        Kokkos::fence();
        system->timer[system->TIMER_COLLISION].stop();
    }
    
    const char* name() { return "CollisionRetroactive"; }
};

} // namespace ExaDiS

#endif

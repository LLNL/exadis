/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_COLLISION_H
#define EXADIS_COLLISION_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Collision
 *
 *-------------------------------------------------------------------------*/
class Collision {
public:
    Collision() {}
    Collision(System *system) {}
    virtual void handle(System *system) {}
    virtual const char* name() { return "CollisionNone"; }
};

} // namespace ExaDiS


// Available collision types
#include "collision_retroactive.h"

#endif

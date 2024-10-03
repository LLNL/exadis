/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_H
#define EXADIS_H

#define EXADIS_VERSION "0.1"

#include <Kokkos_Core.hpp>

#ifdef MPI
#include <mpi.h>
#error "Does not support MPI yet"
#endif

#include <vector>

#include "types.h"
#include "params.h"
#include "system.h"
#include "crystal.h"
#include "neighbor.h"
#include "force.h"
#include "mobility.h"
#include "integrator.h"
#include "collision.h"
#include "topology.h"
#include "remesh.h"
#include "cross_slip.h"

#include "functions.h"

#endif

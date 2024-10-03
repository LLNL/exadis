/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_H
#define EXADIS_CROSS_SLIP_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        CrossSlip
 *
 *-------------------------------------------------------------------------*/
class CrossSlip {
public:
    CrossSlip() {}
    CrossSlip(System* system) {}
    virtual void handle(System* system) {}
    virtual ~CrossSlip() {}
    virtual const char* name() { return "CrossSlipNone"; }
};

} // namespace ExaDiS


// Available cross-slip types
#include "cross_slip_serial.h"

#endif

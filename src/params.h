/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_PARAMS_H
#define EXADIS_PARAMS_H

#include "types.h"
#include "crystal.h"

namespace ExaDiS {

struct Params {
    double burgmag;
    double MU, NU;
    double a;
    double maxseg, minseg;
    double rann;
    double rtol;
    double maxdt, nextdt;
    int split3node;
    
    Params() { set_default_params(); };
    Params(double _burgmag, double _MU, double _NU, double _a, 
           double _maxseg, double _minseg)
    {
        set_default_params();
        burgmag = _burgmag;
        MU = _MU;
        NU = _NU;
        a = _a;
        maxseg = _maxseg;
        minseg = _minseg;
    }
    
    void set_default_params() {
        burgmag = -1.0;
        MU = -1.0;
        MU = -1.0;
        a = -1.0;
        maxseg = -1.0;
        minseg = -1.0;
        
        rtol = -1.0;
        maxdt = 1e-7;
        nextdt = 1e-12;
        rann = -1.0;
        split3node = 1;
    }
    
    void check_params() {
        if (burgmag <= 0.0)
            ExaDiS_fatal("Error: invalid parameter value of burgmag (%f)\n", burgmag);
        if (MU <= 0.0 || NU < 0.0 || a <= 0.0)
            ExaDiS_fatal("Error: invalid parameter values of MU (%f) / NU (%f) / a (%f)\n", MU, NU, a);
        if (maxseg <= 0.0)
            ExaDiS_fatal("Error: invalid parameter values of maxseg (%f)\n", maxseg);
        if (nextdt <= 0.0)
            ExaDiS_fatal("Error: invalid parameter value of nextdt (%f)\n", nextdt);
        
        if (rtol <= 0.0) {
            rtol = 0.25 * a;
            ExaDiS_log("Setting rtol to %f\n", rtol);
        }
        if (rann <= 0.0) {
            rann = 2.0 * rtol;
            ExaDiS_log("Setting rann to %f\n", rann);
        }
        if (minseg <= 0.0) {
            minseg = sqrt(8 * rtol * maxseg / sqrt(3.0));
            ExaDiS_log("Setting minseg to %f\n", minseg);
        }
    }
    
    // Python binding
    Params(std::string, double, double, double, double, double, double, double, double, double, double, int);
    CrystalParams crystal;
};

} // namespace ExaDiS

#endif

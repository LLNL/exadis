/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_CORE_H
#define EXADIS_FORCE_CORE_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:   CoreDefault
 *              Default core energy/force model
 *
 *-------------------------------------------------------------------------*/
struct CoreDefault
{
    double NU;
    double Ecore, Ecore_junc_fact;
    
    struct Params {
        double Ecore, Ecore_junc_fact;
        Params() { Ecore = -1.0; Ecore_junc_fact = 1.0; }
        Params(double _Ecore) { Ecore = _Ecore; Ecore_junc_fact = 1.0; }
        Params(double _Ecore, double _Ecore_junc_fact) { Ecore = _Ecore; Ecore_junc_fact = _Ecore_junc_fact; }
    };
    
    CoreDefault(System *system, Params &params) {
        NU = system->params.NU;
        Ecore = params.Ecore;
        Ecore_junc_fact = params.Ecore_junc_fact;
        
        if (Ecore < 0.0)
            Ecore = system->params.MU / (4.0*M_PI) * log(system->params.a/0.1);
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 core_force(const Vec3 &b, const Vec3 &t)
    {
        double bs = dot(b, t);
        Vec3 be = b - bs*t;
        double be2 = be.norm2();
        double bs2 = bs*bs;
        
        double Ecore_seg = Ecore;
        double bmag = b.norm();
        if (bmag > 1.01 && bmag < 1.5) Ecore_seg *= Ecore_junc_fact;
        
        double fL = -Ecore_seg*(bs2+be2/(1-NU));
        double ft =  Ecore_seg*2*bs*NU/(1-NU);
        
        Vec3 f = ft*be + fL*t;
        return -1.0*f; // return f1, f2=-f1
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:   CoreConstant
 *              Core model with constant orientation line-tension
 *
 *-------------------------------------------------------------------------*/
struct CoreConstant
{
    double Ecore, Ecore_junc_fact;
    
    struct Params {
        double Ecore, Ecore_junc_fact;
        Params() { Ecore = -1.0; Ecore_junc_fact = 1.0; }
        Params(double _Ecore) { Ecore = _Ecore; Ecore_junc_fact = 1.0; }
        Params(double _Ecore, double _Ecore_junc_fact) { Ecore = _Ecore; Ecore_junc_fact = _Ecore_junc_fact; }
    };
    
    CoreConstant(System *system, Params &params) {
        Ecore = params.Ecore;
        Ecore_junc_fact = params.Ecore_junc_fact;
        
        if (Ecore < 0.0)
            Ecore = system->params.MU / (4.0*M_PI) * log(system->params.a/0.1);
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 core_force(const Vec3 &b, const Vec3 &t)
    {
        double Ecore_seg = Ecore;
        double bmag = b.norm();
        if (bmag > 1.01 && bmag < 1.5) Ecore_seg *= Ecore_junc_fact;
        
        double b2 = b.norm2();
        double fL = -Ecore_seg*b2;
        Vec3 f = fL*t;
        return -1.0*f; // return f1, f2=-f1
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:   CoreMD
 *              Core energy model E(cos(theta)^2) fitted to atomistic data
 *
 *-------------------------------------------------------------------------*/
struct CoreMD
{
    static const int MAX_COEFFS = 10;
    double convert;
    int porder[2];
    double pcoeffs[2][MAX_COEFFS];
    
    struct Params {
        double rc;
        int porder0, porder1;
        std::vector<double> pcoeffs0, pcoeffs1;
        Params() { rc = -1.0; }
        Params(double _rc, 
               int _porder0, std::vector<double> _pcoeffs0,
               int _porder1, std::vector<double> _pcoeffs1)
        {
            rc = _rc;
            porder0 = _porder0;
            pcoeffs0 = _pcoeffs0;
            porder1 = _porder1;
            pcoeffs1 = _pcoeffs1;
        }
    };
    
    CoreMD(System *system, Params &params) {
        // Units conversion from eV/A to J/b^2
        double burgmag = system->params.burgmag;
        convert = 1.6022e-19 * 1e10 / burgmag / burgmag;
        
        if (params.rc < 0)
            ExaDiS_fatal("Error: undefined CoreMD model\n");
        
        if (params.rc != system->params.a)
            ExaDiS_fatal("Error: Cutoff (rc = %f) used to fit CoreMD values must equal the core radius (a = %f)\n", params.rc, system->params.a);
        
        if (params.porder0 >= MAX_COEFFS || params.porder1 >= MAX_COEFFS)
            ExaDiS_fatal("Error: the number of coeffs cannot exceed %d in CoreMD\n", MAX_COEFFS);
        
        if (params.pcoeffs0.size() != params.porder0+1 ||
            params.pcoeffs1.size() != params.porder1+1)
            ExaDiS_fatal("Error: the number of coeffs should be equal to porder+1 in CoreMD\n");
        
        porder[0] = params.porder0;
        for (int i = 0; i <= params.porder0; i++)
            pcoeffs[0][i] = params.pcoeffs0[i];
        porder[1] = params.porder1;
        for (int i = 0; i <= params.porder1; i++)
            pcoeffs[1][i] = params.pcoeffs1[i];
    }
    
    KOKKOS_INLINE_FUNCTION
    Vec3 core_force(const Vec3 &b, const Vec3 &t)
    {
        double bmag = b.norm();
        double b2 = b.norm2();
        double bs = dot(b, t);
        Vec3 be = b - bs*t;
        
        // Dislocation type
        int btype; // Burgers type
        if (bmag < 1.01) btype = 0; // <110>(FCC) or <111>(BCC) Burgers
        else if (bmag < 1.5) btype = 1; // Binary junction
        else btype = 2; // Other junction
        
        // Character angle
        double cost = bs / bmag;
		double cos2t = cost*cost;
        
        double bscale = 1.0;
        if (btype == 2) bscale = b2 / pow(2.0/sqrt(3.0), 2.0); // scale b^2 wrt [100] junction
        
        int ptype = (btype == 0) ? 0 : 1;
        int order = porder[ptype];
        double E = bscale*pcoeffs[ptype][order];
		double dEdcos2 = 0.0;
		double x = 1.0;
		for (int i = order-1; i >= 0; i--) {
			dEdcos2 += bscale*pcoeffs[ptype][i]*x;
			x *= cos2t;
			E += bscale*pcoeffs[ptype][i]*x;
		}
        
        // Units conversion from eV/A to J/b^2
        E *= convert;
        dEdcos2 *= convert;
        
        Vec3 f = -dEdcos2*2.0*bs/b2*be-E*t;
        return -1.0*f; // return f1, f2=-f1
    }
};

} // namespace ExaDiS

#endif

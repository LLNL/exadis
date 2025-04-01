/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_ISO_H
#define EXADIS_FORCE_ISO_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:       SegSegIso
 *                  Isotropic non-singular seg/seg force kernel
 *
 *-------------------------------------------------------------------------*/
struct SegSegIso
{
    struct Params {};
    
    double MU, NU, a;
    
    SegSegIso(System *system, Params params) {
        MU = system->params.MU;
        NU = system->params.NU;
        a = system->params.a;
    }

    template<class N>
    KOKKOS_FORCEINLINE_FUNCTION
    SegSegForce segseg_force(System *system, N *net, const SegSeg &ss, 
                             int compute_seg12=1, int compute_seg34=1) 
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;
        
        int n1 = segs[ss.s1].n1;
        int n2 = segs[ss.s1].n2;
        Vec3 b1 = segs[ss.s1].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        double l1 = (r2-r1).norm2();
            
        int n3 = segs[ss.s2].n1;
        int n4 = segs[ss.s2].n2;
        Vec3 b2 = segs[ss.s2].burg;
        Vec3 r3 = cell.pbc_position(r1, nodes[n3].pos);
        Vec3 r4 = cell.pbc_position(r3, nodes[n4].pos);
        double l2 = (r4-r3).norm2();

        Vec3 f1(0.0), f2(0.0), f3(0.0), f4(0.0);
        if (l1 >= 1.e-20 && l2 >= 1.e-20) {
            SegSegForceIsotropic(r1, r2, r3, r4, b1, b2, a, MU, NU, 
                                 f1, f2, f3, f4, compute_seg12, compute_seg34);
        }
        
        return SegSegForce(f1, f2, f3, f4);
    }
};

/*---------------------------------------------------------------------------
 *
 *    Struct:       SegSegIsoFFT
 *                  Isotropic non-singular seg/seg force kernel to be used
 *                  as the short-range force contribution with the long-range
 *                  ForceFFT method.
 *                  It includes two calls of the SegSegForceIsotropic function,
 *                  one with the physical core radius <a>, and one with the 
 *                  numerical grid size <rcgrid> (force correction).
 *
 *-------------------------------------------------------------------------*/
struct SegSegIsoFFT
{
    struct Params {
        double rcgrid;
        Params() { rcgrid = -1.0; }
        Params(double _rcgrid) { rcgrid = _rcgrid; }
    };
    
    double MU, NU, a, rcgrid;
    
    SegSegIsoFFT(System *system, Params params) {
        rcgrid = params.rcgrid;
        initialize(system);
    }
    
    template<class FLong>
    SegSegIsoFFT(System *system, FLong *flong) {
        rcgrid = flong->get_rcgrid();
        initialize(system);
    }
    
    void initialize(System *system) {
        if (rcgrid < 0.0)
            ExaDiS_fatal("Error: undefined rcgrid parameter in ForceSegSegIsoFFT\n");
        MU = system->params.MU;
        NU = system->params.NU;
        a = system->params.a;
    }

    template<class N>
    KOKKOS_FORCEINLINE_FUNCTION
    SegSegForce segseg_force(System *system, N *net, const SegSeg &ss,
                             int compute_seg12=1, int compute_seg34=1) 
    {
        // No need to compute anything if all 
        // forces are accounted for by the grid
        if (rcgrid <= a) return SegSegForce();
        
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;
        
        int n1 = segs[ss.s1].n1;
        int n2 = segs[ss.s1].n2;
        Vec3 b1 = segs[ss.s1].burg;
        Vec3 r1 = nodes[n1].pos;
        Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
        double l1 = (r2-r1).norm2();
            
        int n3 = segs[ss.s2].n1;
        int n4 = segs[ss.s2].n2;
        Vec3 b2 = segs[ss.s2].burg;
        Vec3 r3 = cell.pbc_position(r1, nodes[n3].pos);
        Vec3 r4 = cell.pbc_position(r3, nodes[n4].pos);
        double l2 = (r4-r3).norm2();

        Vec3 f1(0.0), f2(0.0), f3(0.0), f4(0.0);
        Vec3 f1c(0.0), f2c(0.0), f3c(0.0), f4c(0.0);
        if (l1 >= 1.e-20 && l2 >= 1.e-20) {
            SegSegForceIsotropic(r1, r2, r3, r4, b1, b2, a, MU, NU, 
                                 f1, f2, f3, f4, compute_seg12, compute_seg34);
            SegSegForceIsotropic(r1, r2, r3, r4, b1, b2, rcgrid, MU, NU,
                                 f1c, f2c, f3c, f4c, compute_seg12, compute_seg34);
        }
        
        return SegSegForce(f1-f1c, f2-f2c, f3-f3c, f4-f4c);
    }
};

} // namespace ExaDiS

#endif

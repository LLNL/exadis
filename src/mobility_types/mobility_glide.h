/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_GLIDE_H
#define EXADIS_MOBILITY_GLIDE_H

#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityGlide
 *
 *-------------------------------------------------------------------------*/
struct MobilityGlide
{
    const bool non_linear = false;
    double Medge, Mscrew;
    
    struct Params {
        double Medge, Mscrew;
        Params() { Medge = Mscrew = -1.0; }
        Params(double Mglide) {
            Medge = Mglide;
            Mscrew = Mglide;
        }
        Params(double _Medge, double _Mscrew) {
            Medge = _Medge;
            Mscrew = _Mscrew;
        }
    };
    
    MobilityGlide(System* system, Params& params)
    {
        if (params.Medge < 0 || params.Mscrew < 0.0)
            ExaDiS_fatal("Error: invalid MobilityGlide parameter values\n");
        
        Medge  = params.Medge;
        Mscrew = params.Mscrew;
    }
    
    KOKKOS_INLINE_FUNCTION
    Mat33 glide_constraints(int nconn, Vec3* norm)
    {
        Mat33 P = Mat33().eye();
        
        // Find independent glide constraints
        for (int j = 0; j < nconn; j++) {
            for (int k = 0; k < j; k++)
                norm[j] = norm[j].orthogonalize(norm[k]);
            if (norm[j].norm2() >= 0.05) {
                norm[j] = norm[j].normalized();
                Mat33 Q = Mat33().eye() - outer(norm[j], norm[j]);
                P = Q * P;
            }
        }
        
        // Zero-out tiny non-zero components due to round-off errors
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (fabs(P[i][j]) < 1e-10) P[i][j] = 0.0;
        
        return P;
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_velocity(System* system, N* net, const int& i, const Vec3& fi)
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto conn = net->get_conn();
        auto cell = net->cell;
        
        Vec3 vi(0.0);
        
        int nconn = conn[i].num;
        if (nconn >= 2 && nodes[i].constraint != PINNED_NODE) {

            double eps = 1e-10;
            
            Vec3 r1 = nodes[i].pos;
            Vec3 norm[MAX_CONN];
            
            int numNonZeroLenSegs = 0;
            double LtimesB = 0.0;
            for (int j = 0; j < nconn; j++) {

                int k = conn[i].node[j];
                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
                Vec3 dr = r2-r1;
                double L = dr.norm();
                if (L < eps) continue;
                numNonZeroLenSegs++;
                dr = 1.0/L * dr;
                
                int s = conn[i].seg[j];
                int order = conn[i].order[j];
                Vec3 burg = order*segs[s].burg;
                double bMag = burg.norm();
                double invbMag = 1.0 / bMag;
                
                norm[j] = segs[s].plane.normalized();
                
                double dangle = invbMag * fabs(dot(burg, dr));
                double Mob = Medge+(Mscrew-Medge)*dangle;
                LtimesB += (L / Mob);
            }
            LtimesB /= 2.0;
            
            if (numNonZeroLenSegs > 0) {
                // Get glide constraints projection matrix
                Mat33 P = glide_constraints(nconn, norm);
                vi = P * (1.0/LtimesB * fi);
            }
        }
        
        return vi;
    }
    
    static constexpr const char* name = "MobilityGlide";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityGlide> GLIDE;
}

} // namespace ExaDiS

#endif

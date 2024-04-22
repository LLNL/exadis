/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_FCC0_H
#define EXADIS_MOBILITY_FCC0_H

#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityFCC0
 *
 *-------------------------------------------------------------------------*/
struct MobilityFCC0
{
    double Medge, Mscrew;
    double vmax, vscale;
    
    struct Params {
        double Medge, Mscrew;
        double vmax;
        Params() { Medge = Mscrew = vmax = -1.0; }
        Params(double _Medge, double _Mscrew) {
            Medge = _Medge;
            Mscrew = _Mscrew;
            vmax = -1.0;
        }
        Params(double _Medge, double _Mscrew, double _vmax) {
            Medge = _Medge;
            Mscrew = _Mscrew;
            vmax = _vmax;
        }
    };
    
    MobilityFCC0(System* system, Params& params)
    {
        if (system->crystal.type != FCC_CRYSTAL)
            ExaDiS_fatal("Error: MobilityFCC0 must be used with FCC crystal type\n");
            
        if (params.Medge < 0 || params.Mscrew < 0.0)
            ExaDiS_fatal("Error: invalid MobilityFCC0 parameter values\n");
        
        Medge  = params.Medge;
        Mscrew = params.Mscrew;
        vmax   = params.vmax;
        vscale = system->params.burgmag; //vscale (convert factor from m/s)
    }
    
    KOKKOS_INLINE_FUNCTION
    Mat33 glide_constraints(int nconn, Vec3* norm, Vec3* line)
    {
        Mat33 P = Mat33().eye();
        
        // Find independent glide constraints
        int ngc = 0;
        for (int j = 0; j < nconn; j++) {
            for (int k = 0; k < j; k++)
                norm[j] = norm[j].orthogonalize(norm[k]);
            if (norm[j].norm2() >= 0.05) {
                norm[j] = norm[j].normalized();
                Mat33 Q = Mat33().eye() - outer(norm[j], norm[j]);
                P = Q * P;
                ngc++;
            }
        }
        
        // Find independent line constraints
        int nlc = 0;
        int jlc = -1;
        for (int j = 0; j < nconn; j++) {
            for (int k = 0; k < j; k++)
                line[j] = line[j].orthogonalize(line[k]);
            if (line[j].norm2() >= 0.05) {
                jlc = j;
                nlc++;
            }
        }
        
        if (nlc == 1) {
            P = outer(line[jlc], line[jlc]) * P;
            for (int j = 0; j < ngc; j++)
                if (fabs(dot(norm[j], line[jlc])) > 0.05) P.zero();
        } else if (nlc >= 2) {
            P.zero();
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
            Vec3 line[MAX_CONN];
            
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
                // Need crystal rotation here
                Vec3 n = system->crystal.Rinv * norm[j];
                if ((fabs(fabs(n.x) - fabs(n.y)) > 1e-4) ||
                    (fabs(fabs(n.y) - fabs(n.z)) > 1e-4)) {
                    // not a {111} plane
                    line[j] = dr.normalized();
                } else {
                    line[j] = Vec3(0.0);
                }
                
                double dangle = invbMag * fabs(dot(burg, dr));
                double Mob = Medge+(Mscrew-Medge)*dangle;
                LtimesB += (L / Mob);
            }
            LtimesB /= 2.0;
            
            if (numNonZeroLenSegs > 0) {
                // Get glide constraints projection matrix
                Mat33 P = glide_constraints(nconn, norm, line);
                
                vi = P * (1.0/LtimesB * fi);
                if (vmax > 0.0) 
                    apply_velocity_cap(vmax, vscale, vi);
            }
        }
        
        return vi;
    }
    
    static constexpr const char* name = "MobilityFCC0";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityFCC0> FCC_0;
}

} // namespace ExaDiS

#endif

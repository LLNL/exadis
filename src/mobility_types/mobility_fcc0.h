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
    bool non_linear = false;
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
        
        if (!system->crystal.use_glide_planes)
            ExaDiS_fatal("Error: MobilityFCC0 requires the use of glide planes\n");
        
        if (params.Medge < 0 || params.Mscrew < 0.0)
            ExaDiS_fatal("Error: invalid MobilityFCC0 parameter values\n");
        
        Medge  = params.Medge;
        Mscrew = params.Mscrew;
        vmax   = params.vmax;
        vscale = system->params.burgmag; //vscale (convert factor from m/s)
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
            
            Vec3 norm[3], line;
            int ngc = 0;
            int nlc = 0;
            Mat33 P = Mat33().eye();
            
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
                double dangle = 1.0 / bMag * fabs(dot(burg, dr));
                
                double Mob = Medge+(Mscrew-Medge)*dangle;
                LtimesB += (L / Mob);
                
                // Glide constraints
                Vec3 plane = segs[s].plane.normalized();
                Vec3 n = system->crystal.Rinv * plane;
                Vec3 l(0.0);
                if ((fabs(fabs(n.x) - fabs(n.y)) > 1e-4) ||
                    (fabs(fabs(n.y) - fabs(n.z)) > 1e-4)) {
                    l = dr; // not a {111} plane
                }
                // Find independent glide constraints
                if (ngc < 3) {
                    for (int k = 0; k < ngc; k++)
                        plane = plane.orthogonalize(norm[k]);
                    if (plane.norm2() >= 0.05) {
                        plane = plane.normalized();
                        Mat33 Q = Mat33().eye() - outer(plane, plane);
                        P = Q * P;
                        norm[ngc++] = plane;
                    }
                }
                // Find independent line constraints
                if (nlc < 2) {
                    if (nlc == 1)
                        l = l.orthogonalize(line);
                    if (l.norm2() >= 0.05) {
                        line = l.normalized();
                        nlc++;
                    }
                }
            }
            LtimesB /= 2.0;
            
            if (numNonZeroLenSegs > 0) {
                
                // Apply glide constraints
                if (nlc == 1) {
                    P = outer(line, line) * P;
                    for (int j = 0; j < ngc; j++)
                        if (fabs(dot(norm[j], line)) > 0.05) P.zero();
                } else if (nlc >= 2) {
                    P.zero();
                }
                
                // Zero-out tiny non-zero components due to round-off errors
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        if (fabs(P[j][k]) < 1e-10) P[j][k] = 0.0;
                
                // Compute nodal velocity
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

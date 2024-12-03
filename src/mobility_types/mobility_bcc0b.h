/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_BCC0B_H
#define EXADIS_MOBILITY_BCC0B_H

#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityBCC0b
 *
 *-------------------------------------------------------------------------*/
struct MobilityBCC0b
{
    bool non_linear = false;
    double Beclimbj;
    double Bscrew2, Beclimb2;
    double Bline, BlmBsc, BlmBecl;
    double invBscrew2, invBedge2;
    double Bedge, Bscrew, Beclimb;
    double vmax, vscale;
    double Fedge, Fscrew;
    
    struct Params {
        double Medge, Mscrew, Mclimb;
        double Fedge = 0.0, Fscrew = 0.0;
        double vmax = -1.0;
        Params() { Medge = Mscrew = Mclimb = vmax = -1.0; }
        Params(double _Medge, double _Mscrew, double _Mclimb) {
            Medge = _Medge;
            Mscrew = _Mscrew;
            Mclimb = _Mclimb;
            vmax = -1.0;
        }
        Params(double _Medge, double _Mscrew, double _Mclimb, double _vmax) {
            Medge = _Medge;
            Mscrew = _Mscrew;
            Mclimb = _Mclimb;
            vmax = _vmax;
        }
        Params(double _Medge, double _Mscrew, double _Mclimb,
               double _Fedge, double _Fscrew, double _vmax) {
            Medge = _Medge;
            Mscrew = _Mscrew;
            Mclimb = _Mclimb;
            Fedge = _Fedge;
            Fscrew = _Fscrew;
            vmax = _vmax;
        }
    };
    
    MobilityBCC0b(System* system, Params& params)
    {
        if (system->crystal.type != BCC_CRYSTAL)
            ExaDiS_fatal("Error: MobilityBCC0b() must be used with BCC crystal type\n");
            
        if (params.Medge < 0 || params.Mscrew < 0.0 || params.Mclimb < 0.0 ||
            params.Fedge < 0 || params.Fscrew < 0.0)
            ExaDiS_fatal("Error: invalid MobilityBCC0b() parameter values\n");
        
        Bedge   = 1.0 / params.Medge;
        Bscrew  = 1.0 / params.Mscrew;
        Beclimb = 1.0 / params.Mclimb;
        vmax    = params.vmax;
        vscale  = system->params.burgmag; //vscale (convert factor from m/s)
        Fedge   = params.Fedge;
        Fscrew  = params.Fscrew;
        
        if (Fedge > 1e-5 || Fscrew > 1e-5)
            non_linear = true;
        
        // Initialization
        Beclimbj   = Beclimb;

        Bscrew2    = Bscrew * Bscrew;
        Beclimb2   = Beclimb * Beclimb;

        Bline      = 1.0e-2 * MIN(Bscrew, Bedge);
        BlmBsc     = Bline - Bscrew;
        BlmBecl    = Bline - Beclimbj;

        invBscrew2 = 1.0 / (Bscrew*Bscrew);
        invBedge2  = 1.0 / (Bedge*Bedge);
    }
    
    template<class N>
    KOKKOS_INLINE_FUNCTION
    Vec3 node_velocity(System *system, N *net, const int &i, const Vec3 &fi)
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto conn = net->get_conn();
        auto cell = net->cell;
        
        Vec3 vi(0.0);
        
        if (conn[i].num >= 2 && nodes[i].constraint != PINNED_NODE) {
            
            int linejunc = 0;
            Vec3 tjunc(0.0);
            
            if (system->params.split3node) {
                
                int binaryjunc = 0;
                int planarjunc = 0;
                binaryjunc = BCC_binary_junction_node(system, net, i, tjunc, &planarjunc);                
                binaryjunc = (binaryjunc > -1);

                if (binaryjunc) {
                    if (planarjunc) {
                        linejunc = 1;
                    } else {
                        int unzipping = (dot(fi, tjunc) > 0.0);
                        // If the node is unzipping the junction and we are
                        // not treating unzipping as a purely topological
                        // operation, then orthogonalize the climb direction
                        if (unzipping) linejunc = 1;
                    }
                }
            }

            double eps = 1e-12;
            Mat33 Btotal = Mat33().zero();
            double FricForce = 0.0;

            // Build drag matrix
            Vec3 r1 = nodes[i].pos;
            int numNonZeroLenSegs = 0;
            for (int j = 0; j < conn[i].num; j++) {

                int k = conn[i].node[j];
                int s = conn[i].seg[j];
                int order = conn[i].order[j];

                Vec3 burg = order*segs[s].burg;
                double bMag = burg.norm();
                double bMag2 = bMag*bMag;
                double invbMag2 = 1.0 / bMag2;

                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
                
                Vec3 dr = r2-r1;
                double mag = dr.norm();
                if (mag < eps) continue;
                numNonZeroLenSegs++;

                double halfMag = 0.5 * mag;
                double invMag  = 1.0 / mag;
                dr = invMag * dr;

                double costheta = dot(dr, burg);
                double costheta2 = (costheta*costheta) * invbMag2;
                
                double dangle = 1.0 / bMag * fabs(costheta);
                double fricStress = Fedge+(Fscrew-Fedge)*dangle;
                FricForce += fricStress * bMag * mag;
                
                if (bMag > 1.0+eps) {
                    // [0 0 1] arms don't move as readily as other arms, so must be
                    // handled specially.
                    
                    // The junction node move along the junction line freely
                    // No drag on the junction line
                    if (linejunc == 1) {
                        Btotal[0][0] += halfMag * Bline;
                        Btotal[1][1] += halfMag * Bline;
                        Btotal[2][2] += halfMag * Bline;
                        continue;
                    }
                    
                    if (conn[i].num == 2) {
                        Btotal[0][0] += halfMag * Beclimbj;
                        Btotal[1][1] += halfMag * Beclimbj;
                        Btotal[2][2] += halfMag * Beclimbj;
                    } else {
                        Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBecl + Beclimbj);
                        Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBecl);
                        Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBecl);
                        Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBecl + Beclimbj);
                        Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBecl);
                        Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBecl + Beclimbj);
                    }
                } else  {
                    // Arm is not [0 0 1], so build the drag matrix assuming the
                    // dislocation is screw type
                    Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBsc + Bscrew);
                    Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBsc);
                    Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBsc);
                    Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBsc + Bscrew);
                    Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBsc);
                    Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBsc + Bscrew);

                    // Now correct the drag matrix for dislocations that are
                    // not screw
                    if ((1.0 - costheta2) > eps) {

                        double invsqrt1mcostheta2 = 1.0 / sqrt((1.0 - costheta2) * bMag2);
                        Vec3 nr = cross(burg, dr);
                        nr = invsqrt1mcostheta2 * nr;

                        Vec3 mr = cross(nr, dr);
                        
                        // Orthogonalize climb direction wrt junction line direction
                        // to avoid numerical issues with binary junction nodes
                        if (linejunc == 1) {
                            nr -= dot(nr, tjunc) * tjunc;
                            nr = nr.normalized();
                        }
                        
                        double Bglide = sqrt(invBedge2+(invBscrew2-invBedge2)*costheta2);
                        Bglide = 1.0 / Bglide;
                        double Bclimb = sqrt(Beclimb2 + (Bscrew2 - Beclimb2) * costheta2);
                        double BclmBsc = Bclimb - Bscrew;
                        double BglmBsc = Bglide - Bscrew;

                        Btotal[0][0] += halfMag * (nr.x*nr.x * BclmBsc +
                                                   mr.x*mr.x * BglmBsc);
                        Btotal[0][1] += halfMag * (nr.x*nr.y * BclmBsc +
                                                   mr.x*mr.y * BglmBsc);
                        Btotal[0][2] += halfMag * (nr.x*nr.z * BclmBsc +
                                                   mr.x*mr.z * BglmBsc);
                        Btotal[1][1] += halfMag * (nr.y*nr.y * BclmBsc +
                                                   mr.y*mr.y * BglmBsc);
                        Btotal[1][2] += halfMag * (nr.y*nr.z * BclmBsc +
                                                   mr.y*mr.z * BglmBsc);
                        Btotal[2][2] += halfMag * (nr.z*nr.z * BclmBsc +
                                                   mr.z*mr.z * BglmBsc);
                    }
                }  // End non-[0 0 1] arm
            }  // End loop over arms

            Btotal[1][0] = Btotal[0][1];
            Btotal[2][0] = Btotal[0][2];
            Btotal[2][1] = Btotal[1][2];
            FricForce /= 2.0;

            if (numNonZeroLenSegs > 0) {
                Mat33 invDragMatrix = Btotal.inverse();
                
                Vec3 f = fi;
                if (FricForce > eps) {
                    double fmag = f.norm();
                    if (fmag > FricForce) {
                        f -= FricForce/fmag * f;
                    } else {
                        f = Vec3(0.0);
                    }
                }
                
                vi = invDragMatrix * f;
                if (vmax > 0.0) 
                    apply_velocity_cap(vmax, vscale, vi);
            }
        }
        
        return vi;
    }
    
    static constexpr const char* name = "MobilityBCC0b";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityBCC0b> BCC_0B;
}

} // namespace ExaDiS

#endif

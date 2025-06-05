/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_FCC0B_H
#define EXADIS_MOBILITY_FCC0B_H

#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:       MobilityFCC0b
 *                  This is alternate version of a generic mobility function
 *                  for FCC materials. It is adapted from the ParaDiS source
 *                  file ParaDiS/src/MobilityLaw_FCC_0b.cc.
 *                  
 *                  It is very similar in function and structure to the 
 *                  BCC_0B mobility although glide planes rather than 
 *                  Burgers vectors are used to identify junctions.
 *
 *-------------------------------------------------------------------------*/
struct MobilityFCC0b
{
    const bool non_linear = false;
    double Bedge, Bscrew;
    double Beclimb0, Beclimbj;
    double Bscrew2;
    double Bline, BlmBsc;
    double invBscrew2, invBedge2;
    double minseg, shortSegCutoff;
    double vmax, vscale;
    
    struct Params {
        double Medge, Mscrew, Mclimb, Mclimbjunc;
        double vmax;
        Params() { Medge = Mscrew = Mclimb = Mclimbjunc = vmax = -1.0; }
        Params(double _Medge, double _Mscrew, double _Mclimb, double _Mclimbjunc=-1.0, double _vmax=-1.0) {
            Medge = _Medge;
            Mscrew = _Mscrew;
            Mclimb = _Mclimb;
            Mclimbjunc = _Mclimbjunc;
            vmax = _vmax;
        }
    };
    
    MobilityFCC0b(System* system, Params& params)
    {
        if (system->crystal.type != FCC_CRYSTAL)
            ExaDiS_fatal("Error: MobilityFCC0b must be used with FCC crystal type\n");
        
        if (!system->crystal.use_glide_planes)
            ExaDiS_fatal("Error: MobilityFCC0b requires the use of glide planes\n");
        
        if (params.Medge < 0 || params.Mscrew < 0.0 || params.Mclimb < 0.0)
            ExaDiS_fatal("Error: invalid MobilityFCC0b() parameter values\n");
        
        Bedge    = 1.0 / params.Medge;
        Bscrew   = 1.0 / params.Mscrew;
        Beclimb0 = 1.0 / params.Mclimb;
        Beclimbj = (params.Mclimbjunc > 0.0) ? 1.0 / params.Mclimbjunc : Beclimb0;
        vmax     = params.vmax;
        vscale   = system->params.burgmag; //vscale (convert factor from m/s)

        Bscrew2    = Bscrew * Bscrew;
        Bline      = 1.0 * MIN(Bscrew, Bedge);
        BlmBsc     = Bline - Bscrew;
        invBscrew2 = 1.0 / (Bscrew*Bscrew);
        invBedge2  = 1.0 / (Bedge*Bedge);
        
        minseg = system->params.minseg;
        shortSegCutoff = 0.5 * minseg;
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
    Vec3 node_velocity(System *system, N *net, const int &i, const Vec3 &fi)
    {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto conn = net->get_conn();
        auto cell = net->cell;
        
        double eps = 1e-12;
        double tor = 1e-5;

        Vec3 vi(0.0);
        
        int nconn = conn[i].num;
        if (nconn >= 2 && nodes[i].constraint != PINNED_NODE) {
            
            double Beclimb  = (nconn > 2) ? Beclimbj : Beclimb0;
            double Beclimb2 = Beclimb * Beclimb;
            double BlmBecl  = Bline - Beclimb;
            
            Vec3 norm[MAX_CONN];
            Vec3 glideConstraints[3];
            Vec3 normCrystal;
            int numNorms = 0;
            
            Vec3 r1 = nodes[i].pos;

            // Build drag matrix
            Mat33 Btotal = Mat33().zero();
            
            int numNonZeroLenSegs = 0;
            for (int j = 0; j < nconn; j++) {

                int k = conn[i].node[j];
                int s = conn[i].seg[j];
                int order = conn[i].order[j];

                Vec3 burg = order*segs[s].burg;
                double bMag2 = burg.norm2();
                double invbMag2 = 1.0 / bMag2;

                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
                Vec3 dr = r2-r1;
                
                norm[j] = segs[s].plane.normalized();
                Vec3 nCryst = system->crystal.Rinv * norm[j];
                
                double mag = dr.norm();
                if (mag < eps) continue;
                numNonZeroLenSegs++;

                double halfMag = 0.5 * mag;
                double invMag  = 1.0 / mag;
                dr = invMag * dr;
                
                // Calculate how close to screw the arm is
                double costheta = dot(dr, burg);
                double costheta2 = (costheta*costheta) * invbMag2;
                
                if (j == 0) {
                    glideConstraints[0] = norm[j];
                    // If this test passes then the dislocation segment is a junction
                    // segment which will be constrained to grow and shrink but not glide.
                    // Here glide planes are assumed to be of 111 type and any zero in the
                    // plane data is assumed to be associated with a non-glide plane
                    double temp = nCryst.x * nCryst.y * nCryst.z;
                    if (fabs(temp) < eps) {
                        numNorms = 1;
                        glideConstraints[2] = dr;
                        glideConstraints[1] = cross(glideConstraints[2], glideConstraints[0]);
                    }
                } else {
                    Vec3 n = norm[j];
                    if (numNorms == 0) {
                        double temp = fabs(dot(glideConstraints[0], n));
                        if (fabs(1.0e0-temp) > tor) {
                            numNorms = 1;
                            n -= temp * glideConstraints[0];
                            glideConstraints[1] = n.normalized();
                            glideConstraints[2] = cross(glideConstraints[0], glideConstraints[1]);
                        }
                    } else if (numNorms == 1) {
                        double temp = dot(glideConstraints[2], n);
                        if (fabs(temp) > tor) numNorms = 2;
                    }
                    
                    // Check to see if the normal is a non-glide plane and then add to
                    // the checks so that it constrains the junction dislocation to
                    // only move along its line
                    double temp = nCryst.x * nCryst.y * nCryst.z;
                    if (fabs(temp) < eps) {
                        Vec3 n2 = cross(n, dr);
                        if (numNorms == 0) {
                            temp = fabs(dot(glideConstraints[0], n2));
                            if (fabs(1.0e0-temp) > tor) {
                                numNorms = 1;
                                n2 -= temp * glideConstraints[0];
                                glideConstraints[1] = n2.normalized();
                                glideConstraints[2] = cross(glideConstraints[0], glideConstraints[1]);
                            }
                        } else if (numNorms == 1) {
                            temp = dot(glideConstraints[2], n2);
                            if (fabs(temp) > tor) numNorms = 2;
                        }
                    }
                }
                
                // Arms not on [1 1 1] planes don't move as readily as
                // other arms, so must be handled specially.
                if (fabs(nCryst.x * nCryst.y * nCryst.z) < eps) {
                    if (nconn == 2) {
                        Btotal[0][0] += halfMag * Beclimb;
                        Btotal[1][1] += halfMag * Beclimb;
                        Btotal[2][2] += halfMag * Beclimb;
                    } else {
                        Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBecl + Beclimb);
                        Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBecl);
                        Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBecl);
                        Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBecl + Beclimb);
                        Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBecl);
                        Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBecl + Beclimb);
                    }
                } else  {
                    // Arm is a regular glide arm, so build the drag matrix
                    // assuming the dislocation is screw type
                    Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBsc + Bscrew);
                    Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBsc);
                    Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBsc);
                    Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBsc + Bscrew);
                    Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBsc);
                    Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBsc + Bscrew);

                    // Now correct the drag matrix for dislocations that are
                    // not screw
                    if ((1.0 - costheta2) > eps) {
                        
                        Vec3 n = norm[j];
                        Vec3 m = cross(n, dr);
                        
                        double Bglide = sqrt(invBedge2+(invBscrew2-invBedge2)*costheta2);
                        Bglide = 1.0 / Bglide;
                        double Bclimb = sqrt(Beclimb2 + (Bscrew2 - Beclimb2) * costheta2);
                        double BclmBsc = Bclimb - Bscrew;
                        double BglmBsc = Bglide - Bscrew;

                        Btotal[0][0] += halfMag * (n.x*n.x * BclmBsc +
                                                   m.x*m.x * BglmBsc);
                        Btotal[0][1] += halfMag * (n.x*n.y * BclmBsc +
                                                   m.x*m.y * BglmBsc);
                        Btotal[0][2] += halfMag * (n.x*n.z * BclmBsc +
                                                   m.x*m.z * BglmBsc);
                        Btotal[1][1] += halfMag * (n.y*n.y * BclmBsc +
                                                   m.y*m.y * BglmBsc);
                        Btotal[1][2] += halfMag * (n.y*n.z * BclmBsc +
                                                   m.y*m.z * BglmBsc);
                        Btotal[2][2] += halfMag * (n.z*n.z * BclmBsc +
                                                   m.z*m.z * BglmBsc);
                    }
                }  // End of regular arm
            }  // End loop over arms

            Btotal[1][0] = Btotal[0][1];
            Btotal[2][0] = Btotal[0][2];
            Btotal[2][1] = Btotal[1][2];

            if (numNonZeroLenSegs > 0 && 
                (numNorms < 2 || !system->crystal.enforce_glide_planes))
            {
                Mat33 invDragMatrix = Btotal.inverse();
                vi = invDragMatrix * fi;
                
                // Get glide constraints projection matrix
                if (system->crystal.enforce_glide_planes) {
                    Mat33 P = glide_constraints(nconn, norm);
                    vi = P * vi;
                }
                
                if (vmax > 0.0) 
                    apply_velocity_cap(vmax, vscale, vi);
            }
        }
        
        return vi;
    }
    
    static constexpr const char* name = "MobilityFCC0b";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityFCC0b> FCC_0B;
}

} // namespace ExaDiS

#endif

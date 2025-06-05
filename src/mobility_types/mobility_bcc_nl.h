/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_BCC_NL_H
#define EXADIS_MOBILITY_BCC_NL_H

#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:       MobilityBCC_nl
 *                  This mobility function is an example of a non-linear
 *                  mobility for BCC crystals where the nodal velocity is
 *                  solved for iteratively using a Newton-Raphson scheme 
 *                  by balancing the drag force with the driving force 
 *                  F_i^{driv}+F_i^{drag}=0.
 *
 *                  It is a translation in ExaDiS of ParaDiS source file
 *                  ParaDiS/src/Mobility_BCC_Fe_nl.cc with some corrections.
 *                  It includes temperature dependent linear edge behavior 
 *                  and non-linear screw behavior.
 *                  Default parameters are fitted to atomistic simulations
 *                  of dislocations in BCC Fe.
 *
 *-------------------------------------------------------------------------*/
struct MobilityBCC_nl
{
    const bool non_linear = true;
    struct Params {
        double PEIERLS_SCREW = 1.2e9; // Pa
        double B_SCREW       = 4.6e-4; // Pa.s
        double B0_EDGE       = 0.0; // Pa.s
        double B1_EDGE       = 7.7e-7; // Pa.s/K
        double tempK;
        double vmax;
        Params() { tempK = 300.0; vmax = -1.0; }
        Params(double _tempK, double _vmax) : tempK(_tempK), vmax(_vmax) {}
        Params(double _tempK, double _vmax, double _PEIERLS_SCREW, double _B_SCREW,
               double _B0_EDGE, double _B1_EDGE) : tempK(_tempK), vmax(_vmax) {
            PEIERLS_SCREW = _PEIERLS_SCREW;
            B_SCREW       = _B_SCREW;
            B0_EDGE       = _B0_EDGE;
            B1_EDGE       = _B1_EDGE;
        }
    };
    Params params;
    
    MobilityBCC_nl(System* system, Params& _params)
    {
        if (system->crystal.type != BCC_CRYSTAL)
            ExaDiS_fatal("Error: MobilityBCC_nl() must be used with BCC crystal type\n");
        
        params = _params;
        if (params.PEIERLS_SCREW < 0.0 || params.B_SCREW < 0.0 || params.B0_EDGE < 0.0)
            ExaDiS_fatal("Error: invalid MobilityBCC_nl() parameter values\n");
        
        if (params.tempK < 100 || params.tempK > 1200)
            ExaDiS_fatal("Error: MobilityBCC_nl() only valid in range T = 100-1200 K\n");
    }
    
    KOKKOS_INLINE_FUNCTION
    void FscaleEdge(System* system, double feinit, double vin, const Vec3& burg,
                    double *fout, double *dfdv, double *dfdvlin, double *d2fdv2)
    {
        double bt = params.B0_EDGE + params.tempK * params.B1_EDGE;

        double vmag = fabs(vin);
        double vsign = SIGN(vin);

        double fdrag = bt * vmag;
        double dfdragdv = bt;

        *fout = fdrag * vsign;
        *dfdv = dfdragdv;
        *dfdvlin = *dfdv; 
        *d2fdv2 = 0.0;
    }
    
    KOKKOS_INLINE_FUNCTION
    void EdgeDrag(System* system, Vec3& vel, const Vec3& burg, Vec3& climbdir,
                  Vec3& fedrag, Mat33& dfedragdv)
    {
        Vec3 glidedir = burg;
        glidedir = glidedir.normalized();
        Vec3 linedir = cross(glidedir, climbdir);
    
        double vmag = dot(vel, glidedir);
        double ftest = dot(fedrag, glidedir);
        
        double fout, dfdv, dfdvlin, d2fdv2;
        FscaleEdge(system, ftest, vmag, burg, &fout, &dfdv, &dfdvlin, &d2fdv2);

        double Beclimb = dfdvlin * 1.0e+03;
        double Beline = dfdvlin * 1.0e-03;
        
        dfedragdv = Beclimb * outer(climbdir, climbdir) 
                  + Beline * outer(linedir, linedir);
        fedrag = dfedragdv * vel + fout * glidedir;
        dfedragdv += dfdv * outer(glidedir, glidedir);
    }
    
    KOKKOS_INLINE_FUNCTION
    void FscaleScrew(System* system, double fsinit, double vin, const Vec3& burg,
                     double *fout, double *dfdv, double *dfdvlin, double *d2fdv2)
    {
        double burgmag = system->params.burgmag;
        double factor110 = sqrt(3.0)/2.0; 
        double bnorm = burg.norm();
        double tempK = params.tempK;
        double taus = params.PEIERLS_SCREW*(1.0-1.e-3*tempK)*(1.0-1.e-3*tempK)*bnorm;
        taus = taus + 1.0e3; // avoid singularity around 1000K
        double v0 = 1.33318333e-10*tempK*tempK*tempK - 1.4985e-8*tempK*tempK -
                    2.3379833333e-6*tempK+2.5045e-4;
        double c0 = 3710.0/sqrt(tempK);
        double c0p = c0/burgmag;
        double alpha = 3.3;
        double beta = 2e-4*tempK;
        double n = 10;
        double ninv = 1.0 / n;
        double bzero = params.B_SCREW;

        double vmag = fabs(vin);
        vmag = vmag * factor110;
        double vsign = SIGN(vin);
        double ratio = vmag / c0p;
        
        if (vmag < 1.0) {
            // below a very small velocity value, same as taus*dfthermdv but
            // set the vmag to be zero
            // note: dfdvlin = taus * dfdragdv
            *fout = 0.0e0;
            *dfdv = taus * alpha * beta / c0p * pow(v0, beta-1.0);
            *dfdvlin = bzero;
            *d2fdv2 = 0.0;
            return;
        }
            
        double ftherm = alpha * (pow(ratio+v0, beta) - pow(v0, beta));
        double dfthermdv = alpha * beta / c0p * pow(ratio+v0, beta-1.0);
        double fdrag = bzero * vmag / taus;
        double dfdragdv = bzero / taus;
        
        double fmag = taus * pow(pow(fdrag, n)+pow(ftherm, n), ninv);
        double dfmagdfdrag = taus * pow(fdrag, n-1.0) *
                             pow(pow(fdrag, n)+pow(ftherm, n), ninv-1.0);
        double dfmagdftherm = taus * pow(ftherm, n-1.0) *
                              pow(pow(fdrag, n)+pow(ftherm, n), ninv-1.0);
        double dfmagdv = dfmagdfdrag * dfdragdv + dfmagdftherm * dfthermdv;
        
        *fout = fmag * vsign;
        *dfdv = dfmagdv;
        *dfdvlin = bzero;
        *d2fdv2 = 0.0;
    }
    
    KOKKOS_INLINE_FUNCTION
    void ScrewDrag(System* system, Vec3& vel, const Vec3& burg,
                   Vec3& finit, Mat33& dfsdragdv)
    {
        double eps = 1.0e-12;
        double bnorm = burg.norm();
        Vec3 linedir = 1.0 / bnorm * burg;
        Mat33 linedirTM = outer(linedir, linedir);
        Mat33 glidedirmat = Mat33().eye() - linedirTM;
        
        Vec3 vproj = glidedirmat * vel;
        double vMag = vproj.norm();
        double vMagInv = 1.0 / (vMag + eps);
        Vec3 vdir = vMagInv * vproj;
        Mat33 dvdirdv = vMagInv * (glidedirmat - outer(vdir, vdir));
        
        Vec3 fproj = glidedirmat * finit;
        double ftest = fproj.norm();

        double fout = 0.0;
        double dfdv = 0.0;
        double d2fdv2 = 0.0;
        double dfdvlin;
        FscaleScrew(system, ftest, vMag, burg, &fout, &dfdv, &dfdvlin, &d2fdv2);
        
        double Bsline = dfdvlin * 1.0e-03;
        dfsdragdv = Bsline * linedirTM;
        finit = (dfsdragdv * vel) + fout * vdir;
        double vNorm = vel.norm();
        
        if ((vNorm > eps) && ((vMag/vNorm) > eps)) {
            dfsdragdv += dfdv * outer(vdir, vdir) + fout * dvdirdv;
        } else {
            dfsdragdv += dfdv * glidedirmat;
        }
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
            
            Vec3 r1 = nodes[i].pos;
            
            Vec3 norm[3];
            int ngc = 0;
            Mat33 P = Mat33().eye();
            
            int numNonZeroLenSegs = 0;
            int numscrews = 0;
            int numedges = 0;
            int numjunct = 0;
            Vec3 junctb[MAX_CONN], junctdir[MAX_CONN];
            Vec3 screwb[MAX_CONN], edgeb[MAX_CONN], edgenorm[MAX_CONN];
            double dfndfjunct[MAX_CONN], dfndfscrew[MAX_CONN], dfndfedge[MAX_CONN];
            
            // Loop over all arms attached to the node
            for (int j = 0; j < conn[i].num; j++) {

                int k = conn[i].node[j];
                int s = conn[i].seg[j];
                int order = conn[i].order[j];
                
                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
                Vec3 dr = r2-r1;
                double mag2 = dr.norm2();
                if (mag2 < eps) continue;
                numNonZeroLenSegs++;
                
                Vec3 burg = order*segs[s].burg;
                double bnorm = burg.norm();
                
                double LTotal = sqrt(mag2);
                double LScrew = fabs(dot(dr, burg)) / bnorm;
                LScrew = fmin(LTotal, LScrew);
                double L2 = fmax(0.0, (LTotal*LTotal-LScrew*LScrew));
                double LEdge = (L2 <= 0.0) ? 0.0 : sqrt(L2);
                
                Vec3 linedir = 1.0/LTotal * dr;
                bool edgeExists  = ((LEdge / LTotal) > 1.0e-06);
                bool screwExists = ((LScrew / LTotal) > 1.0e-06);
                
                Vec3 ndir(0.0);
                if (edgeExists)
                    ndir = cross(burg, linedir).normalized();
                
                //double burgProd = burg[0] * burg[1] * burg[2];
                //if (fabs(burgProd) <= eps) {
                if (bnorm > 1.0+eps) {
                    
                    int junctid = -1;
                    for (int l = 0; l < numjunct; l++) {
                        double mag1 = dot(burg, junctb[l]) / bnorm / junctb[l].norm();
                        double mag2 = dot(linedir, junctdir[l]) / junctdir[l].norm();
                        if ((fabs(fabs(mag1)-1.0) < eps) &&
                            (fabs(fabs(mag2)-1.0) < eps)) {
                            junctid = l;
                        }
                    }
                    
                    if (junctid < 0) {
                        junctb[numjunct] = burg;
                        if (linejunc) {
                            junctdir[numjunct] = tjunc;
                        } else {
                            junctdir[numjunct] = linedir;
                        }
                        dfndfjunct[numjunct] = 0.5 * LTotal;
                        numjunct++;
                    } else {
                        dfndfjunct[junctid] += 0.5 * LTotal;
                    }
                    
                } else {
                    
                    if (screwExists) {
                        int screwid = -1;
                        for (int l = 0; l < numscrews; l++) {
                            double mag = dot(burg, screwb[l]) / bnorm / screwb[l].norm();
                            if (fabs(fabs(mag) - 1.0) < eps) screwid = l;
                        }
                        if (screwid < 0) {
                            screwb[numscrews] = burg;
                            dfndfscrew[numscrews] = 0.5*LScrew;
                            numscrews++;
                        } else {
                            dfndfscrew[screwid] += 0.5*LScrew;
                        }
                    }
                    
                    if (edgeExists) {
                        int edgeid = -1;
                        for (int l = 0; l < numedges; l++) {
                            double mag1 = dot(ndir, edgenorm[l]);
                            double mag2 = dot(burg, edgeb[l]) / bnorm / edgeb[l].norm();
                            if ((fabs(fabs(mag1) - 1.0) < eps) &&
                                (fabs(fabs(mag2) - 1.0) < eps)) {
                                edgeid = l;
                            }
                        }
                        if (edgeid < 0) {
                            edgeb[numedges] = burg;
                            edgenorm[numedges] = ndir;
                            dfndfedge[numedges] = 0.5 * LEdge;
                            numedges++;
                        } else {
                            dfndfedge[edgeid] += 0.5 * LEdge;
                        }
                    }
                }
                
                if (system->crystal.enforce_glide_planes) {
                    // Find independent glide constraints
                    Vec3 plane = segs[s].plane.normalized();
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
                }
            
            }  // End loop over arms
            
            if (numNonZeroLenSegs == 0) return vi;
            
            Vec3 fscrew[MAX_CONN], fedge[MAX_CONN];
            Mat33 dferrordvold = Mat33().zero();
            
            double mu = system->params.MU;
            double lmax = system->params.maxseg;
            double ferrortol = fmax((dot(fi,fi)*1.0e-16),(mu*mu*lmax*lmax*1.0e-24));
            
            double BJline = 1.e-3;
            double C0 = 2000.0 / system->params.burgmag;
            double EpsC0 = 1.0e-06 * C0;
            double rtol = system->params.rtol;
            double dt = system->realdt;
            double velerrortol = fmax((1.0e-14*rtol*rtol/(dt*dt)),(EpsC0*EpsC0*1.0e-20));
            
            double minvelerror = velerrortol;
            double correctionAdjustment = 1.0;
            bool notConverged = 1;
            
            int iterCnt = 0;
            int maxIter = 100;
            int maxIter2 = 200;
            
            Vec3 ferrorold(0.0);
            Vec3 vtestold(0.0);
            Vec3 vtmp(0.0);
            
            Vec3 vtest = nodes[i].v;
            if (numjunct > 1) {
                vtest.zero();
                notConverged = 0;
            } else {
                if (numjunct == 1) {
                    vtest = dot(vtest, junctdir[0]) * junctdir[0];
                }
            }
            
            int count = 0;
            while (notConverged) {
                count++;
                
                Vec3 correction(0.0);
                Vec3 ferror = fi;
                Mat33 dferrordv = Mat33().zero();
                Mat33 dfdv;

                for (int j = 0; j < numscrews; j++) {
                    ScrewDrag(system, vtest, screwb[j], fscrew[j], dfdv);
                    ferror -= dfndfscrew[j] * fscrew[j];
                    dferrordv -= dfndfscrew[j] * dfdv;
                }
                
                for (int j = 0; j < numedges; j++) {
                    EdgeDrag(system, vtest, edgeb[j], edgenorm[j], fedge[j], dfdv);
                    ferror -= dfndfedge[j] * fedge[j];
                    dferrordv -= dfndfedge[j] * dfdv;
                }

                for (int j = 0; j < numjunct; j++) {
                    ferror -= BJline * dfndfjunct[j] * vtest;
                    dferrordv -= (BJline*dfndfjunct[j]) * outer(junctdir[j], junctdir[j]);
                }
                
                            
                // Look for oscillatory behavior
                double forceerror, fscaleError;
                if (numjunct == 1) {
                    fscaleError = dot(ferror, junctdir[0]);
                    forceerror = fscaleError * fscaleError;
                } else {
                    forceerror = dot(ferror, ferror);
                }
                
                double velerror;
                if (forceerror > ferrortol) {
                    Vec3 tmp3 = ferror - ferrorold;
                    double linmin;
                    if (dot(tmp3, ferrorold) != 0.0) {
                        linmin = -dot(tmp3, ferrorold) / tmp3.norm2();
                    } else {
                        linmin = 0.0;
                    }
                    Vec3 vtestnew;
                    if (fabs(linmin - 0.5) < 1.0e-5) {
                        vtestnew = linmin * vtest + (1.0 - linmin) * vtestold;
                        tmp3 = vtestnew - vtest;
                        velerror = dot(tmp3, tmp3);
                    } else {
                        if (numjunct == 1) {
                            tmp3 = dferrordv * junctdir[0];
                            double dfdvscale = dot(junctdir[0], tmp3);
                            correction = (fscaleError / dfdvscale) * junctdir[0];
                            velerror = dot(correction, correction);
                        } else {
                            forceerror = dot(ferror, ferror);
                            Mat33 invdferrordv = dferrordv.inverse();
                            correction = invdferrordv * ferror;
                            velerror = dot(correction, correction);
                        }
                        correction = correctionAdjustment * correction;
                        vtestnew = vtest - correction;
                    }
                    
                    ferrorold = ferror;
                    vtestold = vtest;
                    vtest = vtestnew;
                    dferrordvold = dferrordv;
                } else {
                    notConverged = 0;
                    break;
                }
                
                // If the initial velocity guess was too far off, it's
                // better to just zero the vtest value.  So, after the
                // first iteration, if the force error is too large (an
                // indication the initial velocity guess was way off),
                // just zero the vtest value.
                if ((count == 1) && (forceerror > 1.0e+02 * ferrortol)) {
                    vtest.zero();
                }
                
                // Preserve the vtest value for the iteration that has the
                // lowest absolute error.  If the function fails to converge
                // we may be able to use this preserved velocity.
                if (velerror < minvelerror) {
                    minvelerror = velerror;
                    vtmp = vtest;
                }
                
                if (++iterCnt > maxIter) {
    
                    // We didn't converge on an answer, but if there was an iteration
                    // of the above loop that resulted in an absolute error within
                    // the absolute error tolerance, go ahead and use the velocity
                    // calculated during that iteration.
                    if (minvelerror < velerrortol) {
                        vtest = vtmp;
                        break;
                    }
                    
                    // If the function failed to converge on a velocity in the
                    // first set of iterations, try a second set of iterations
                    // adjusting the correction values by a factor of 0.5. This
                    // helps in some cases and although it increases the cost of
                    // calculating the mobility for a single node (on occassion),
                    // it can keep the timestep from being cut, which more than
                    // compensates.
                    if (iterCnt < maxIter2) {
                        maxIter = maxIter2;
                        count = 0;
                        for (int j = 0; j < MAX_CONN; j++) {
                            fscrew[j].zero();
                            fedge[j].zero();
                        }
                        dferrordvold.zero();
                        velerror = 10.0 * velerrortol;
                        minvelerror = velerror;
                        correctionAdjustment = 0.50;
                        vtest = nodes[i].v;
                        continue;
                    }

                    notConverged = 0;
                }
            
            } /* while (notConverged) */
            
            if (system->crystal.enforce_glide_planes) {
                // Zero-out tiny non-zero components due to round-off errors
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        if (fabs(P[j][k]) < 1e-10) P[j][k] = 0.0;
                
                // Apply glide constraints    
                vtest = P * vtest;
            }
            
            vi = vtest;
            if (params.vmax > 0.0) {
                double vscale = system->params.burgmag; //vscale (convert factor from m/s)
                apply_velocity_cap(params.vmax, vscale, vi);
            }
        }
        
        return vi;
    }
    
    static constexpr const char* name = "MobilityBCC_nl";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityBCC_nl> BCC_NL;
}

} // namespace ExaDiS

#endif

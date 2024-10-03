/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_MOBILITY_FCC0_FRIC_H
#define EXADIS_MOBILITY_FCC0_FRIC_H

#include "mobility.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityField
 *
 *-------------------------------------------------------------------------*/
struct MobilityField {
    bool active;
    Kokkos::View<double***> fieldval;
    int Ng[3];
    
    MobilityField() : active(false) {}
    MobilityField(System* system, std::string mobility_file) : active(true)
    {
        ExaDiS_log("Reading mobility field file %s\n", mobility_file.c_str());
        FILE* fp = fopen(mobility_file.c_str(), "r");
        if (fp == NULL)
            ExaDiS_fatal("Error: MobilityField file %s not found!\n", mobility_file.c_str());

        fscanf(fp, "%d", &Ng[0]);
        fscanf(fp, "%d", &Ng[1]);
        fscanf(fp, "%d", &Ng[2]);
        printf(" MobilityField: %d x %d x %d points\n", Ng[0], Ng[1], Ng[2]);

        for (int i = 0; i < 3; i++)
            if (Ng[i] == 0) Ng[i] = 1;
        
        Kokkos::resize(fieldval, Ng[0], Ng[1], Ng[2]);
        auto h_fieldval = Kokkos::create_mirror_view(fieldval);
        
        for (int i = 0; i < Ng[0]; i++)
            for (int j = 0; j < Ng[1]; j++)
                for (int k = 0; k < Ng[2]; k++)
                    fscanf(fp, "%le\n", &h_fieldval(i, j, k));
        fclose(fp);
        
        Kokkos::deep_copy(fieldval, h_fieldval);
    }
    
    KOKKOS_INLINE_FUNCTION
    double interpolate(const Cell& cell, const Vec3& p)
    {
        Vec3 s = cell.scaled_position(p);
        
        double q[3];
        q[0] = s.x * Ng[0] - 0.5;
        q[1] = s.y * Ng[1] - 0.5;
        q[2] = s.z * Ng[2] - 0.5;
        
        int g[3];
        g[0] = (int)floor(q[0]);
        g[1] = (int)floor(q[1]);
        g[2] = (int)floor(q[2]);

        double xi[3];
        xi[0] = 2.0*(q[0]-g[0]) - 1.0;
        xi[1] = 2.0*(q[1]-g[1]) - 1.0;
        xi[2] = 2.0*(q[2]-g[2]) - 1.0;

        // Determine elements for interpolation and apply PBC
        int ind1d[3][2];
        for (int i = 0; i < 2; i++) {
            ind1d[0][i] = (g[0]+i)%Ng[0];
            if (ind1d[0][i] < 0) ind1d[0][i] += Ng[0];
            ind1d[1][i] = (g[1]+i)%Ng[1];
            if (ind1d[1][i] < 0) ind1d[1][i] += Ng[1];
            ind1d[2][i] = (g[2]+i)%Ng[2];
            if (ind1d[2][i] < 0) ind1d[2][i] += Ng[2];
        }

        // 1d shape functions
        double phi1d[3][2];
        for (int i = 0; i < 3; i++) {
            phi1d[i][0] = 0.5*(1.0-xi[i]);
            phi1d[i][1] = 0.5*(1.0+xi[i]);
        }

        // 3d shape functions and indices
        double val = 0.0;
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    double phi = phi1d[0][i]*phi1d[1][j]*phi1d[2][k];
                    val += phi * fieldval(ind1d[0][i], ind1d[1][j], ind1d[2][k]);
                }
            }
        }
        
        return val;
    }
};

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityFCC0_fric
 *
 *-------------------------------------------------------------------------*/
struct MobilityFCC0_fric : MobilityFCC0
{
    double Fedge, Fscrew;
    MobilityField mobility_field;
    MobilityField friction_field;
    
    struct Params {
        MobilityFCC0::Params params;
        double Fedge = 0.0;
        double Fscrew = 0.0;
        std::string mobility_field_file = "";
        std::string friction_field_file = "";
        
        Params() { params = MobilityFCC0::Params(); }
        Params(double _Medge, double _Mscrew, double _Fedge, double _Fscrew, double _vmax,
               std::string _mobility_field_file="", std::string _friction_field_file="") {
            params = MobilityFCC0::Params(_Medge, _Mscrew, _vmax);
            Fedge = _Fedge;
            Fscrew = _Fscrew;
            mobility_field_file = _mobility_field_file;
            friction_field_file = _friction_field_file;
        }
    };
    
    MobilityFCC0_fric(System* system, Params& params) : MobilityFCC0(system, params.params)
    {
        if (params.Fedge < 0.0 || params.Fscrew < 0.0)
            ExaDiS_fatal("Error: invalid MobilityFCC0_fric parameter values\n");
        
        Fedge  = params.Fedge;
        Fscrew = params.Fscrew;
        
        if (!params.mobility_field_file.empty())
            mobility_field = MobilityField(system, params.mobility_field_file);
        if (!params.friction_field_file.empty())
            friction_field = MobilityField(system, params.friction_field_file);
        
        non_linear = true;
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
            double FricForce = 0.0;
            for (int j = 0; j < nconn; j++) {

                int k = conn[i].node[j];
                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);
                Vec3 dr = r2-r1;
                double L = dr.norm();
                if (L < eps) continue;
                numNonZeroLenSegs++;
                dr = 1.0/L * dr;
                Vec3 rmid = 0.5*(r1+r2);
                
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
                if (mobility_field.active)
                    Mob *= mobility_field.interpolate(cell, rmid);
                LtimesB += (L / Mob);
                
                double fricStress = Fedge+(Fscrew-Fedge)*dangle;
                if (friction_field.active)
                    fricStress *= friction_field.interpolate(cell, rmid);
                FricForce += fricStress * bMag * L;
            }
            LtimesB /= 2.0;
            FricForce /= 2.0;
            
            if (numNonZeroLenSegs > 0) {
                // Get glide constraints projection matrix
                Mat33 P = glide_constraints(nconn, norm, line);
                
                Vec3 f = fi;
                if (FricForce > 0.0) {
                    double fmag = f.norm();
                    if (fmag > FricForce) {
                        f -= FricForce/fmag * f;
                    } else {
                        f = Vec3(0.0);
                    }
                }
                
                vi = P * (1.0/LtimesB * f);
                if (vmax > 0.0) 
                    apply_velocity_cap(vmax, vscale, vi);
            }
        }
        
        return vi;
    }
    
    static constexpr const char* name = "MobilityFCC0_fric";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityFCC0_fric> FCC_0_FRIC;
}

} // namespace ExaDiS

#endif

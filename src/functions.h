/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FUNCTIONS_H
#define EXADIS_FUNCTIONS_H

#include "types.h"
#include "crystal.h"

namespace ExaDiS {

// utils.cpp
std::string replace_string(std::string& str, const std::string& from, const std::string& to);
int create_directory(std::string dirname);
void remove_directory(std::string dirname);

// generate.cpp
void insert_frs(SerialDisNet *network, Vec3 burg, Vec3 plane, Vec3 ldir,
                double L, Vec3 center, Mat33 R, int numnodes=10);
void insert_frs(SerialDisNet *network, Vec3 burg, Vec3 plane, 
                double thetadeg, double L, Vec3 center,
                Mat33 R, int numnodes=10);
double insert_infinite_line(SerialDisNet* network, Vec3 burg, Vec3 plane, Vec3 ldir, 
                            Vec3 origin, Mat33 R, double maxseg=-1);
double insert_infinite_line(SerialDisNet* network, Vec3 burg, Vec3 plane, double thetadeg, 
                            Vec3 origin, Mat33 R, double maxseg=-1);
void insert_prismatic_loop(Crystal& crystal, SerialDisNet *network, Vec3 burg, 
                           double radius, Vec3 center, double maxseg=-1);
SerialDisNet* generate_frs_config(Crystal crystal, Cell cell, int numsources,
                                  double Lsource, double maxseg=-1, int seed=1234);
SerialDisNet* generate_frs_config(Crystal crystal, double Lbox, int numsources,
                                  double Lsource, double maxseg=-1, int seed=1234);
SerialDisNet* generate_prismatic_config(Crystal crystal, Cell cell, int numsources, double radius,
                                        double maxseg=-1, int seed=1234, bool uniform=0);
SerialDisNet* generate_prismatic_config(Crystal crystal, double Lbox, int numsources, double radius,
                                        double maxseg=-1, int seed=1234, bool uniform=0);
SerialDisNet* read_paradis(const char* file, bool verbose=true);


/*------------------------------------------------------------------------
 *
 *    Function:     get_min_dist2
 *                  Calculate the minimum distance between two segments
 *
 *-----------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
double get_min_dist2(const Vec3 &r1, const Vec3 &r2, const Vec3 &r3, const Vec3 &r4)
{
    double  eps = 1.0e-12;

    Vec3 r1mr3 = r1 - r3;
    Vec3 r2mr1 = r2 - r1;
    Vec3 r4mr3 = r4 - r3;

    Vec3 seg1L = r2mr1;
    Vec3 seg2L = r4mr3;

    double M[2][2];
    M[0][0] =  dot(r2mr1, r2mr1);
    M[1][0] = -dot(r4mr3, r2mr1);
    M[1][1] =  dot(r4mr3, r4mr3);
    M[0][1] =  M[1][0];

    double rhs[2];
    rhs[0] = -dot(r2mr1, r1mr3);
    rhs[1] =  dot(r4mr3, r1mr3);

    double detM = 1.0 - M[1][0] * M[1][0] / M[0][0] / M[1][1];

    double A = M[0][0];
    double B = -2.0 * rhs[0];
    //double C = -2.0 * M[1][0];
    double D = -2.0 * rhs[1];
    double E = M[1][1];

    int didDist2 = 0;
    double L1, L2, dist2;

    if (A < eps) {
        // If segment 1 is just a point...
        L1 = 0.0;
        if (E < eps) L2 = 0.0;
        else L2 = -0.5 * D / E;
        
    } else if (E < eps) {
        // If segment 2 is just a point...
        L2 = 0.0;
        if (A < eps) L1 = 0.0;
        else L1 = -0.5 * B / A;

    } else if (detM < 1e-6) {
        // If segments are parallel
        Vec3 r4mr1 = r4 - r1;
        Vec3 r3mr2 = r3 - r2;
        Vec3 r4mr2 = r4 - r2;

        double dist[4];
        dist[0] = dot(r1mr3, r1mr3);
        dist[1] = dot(r4mr1, r4mr1);
        dist[2] = dot(r3mr2, r3mr2);
        dist[3] = dot(r4mr2, r4mr2);

        dist2 = dist[0];
        int pos = 1;
        for (int i = 1; i < 4; i++) {
            if (dist[i] < dist2) {
                dist2 = dist[i];
                pos = i+1;
            }
        }

        L1 = floor((double)pos/2.1);
        L2 = (double)(1 - (pos % 2));
        didDist2 = 1;
    
    } else {
        // Solve the general case
        detM *= M[0][0]*M[1][1];

        double sol[2];
        sol[0] = ( M[1][1]*rhs[0] - M[0][1]*rhs[1]) / detM;
        sol[1] = (-M[1][0]*rhs[0] + M[0][0]*rhs[1]) / detM;

        if ((sol[0]>=0) && (sol[0]<=1) && (sol[1]>=0) && (sol[1]<=1)) {
            // we are done here
            L1 = sol[0];
            L2 = sol[1];

        } else {
            // enumerate four cases
            int icase;
            double trial[4][2];

            // alpha = 0
            icase = 0;
            trial[icase][0] = 0;
            trial[icase][1] = (rhs[1] - M[1][0]*trial[icase][0]) / M[1][1];

            // alpha = 1
            icase = 1;
            trial[icase][0] = 1;
            trial[icase][1] = (rhs[1] - M[1][0]*trial[icase][0]) / M[1][1];

            // beta = 0
            icase = 2;
            trial[icase][1] = 0;
            trial[icase][0] = (rhs[0] - M[0][1]*trial[icase][1]) / M[0][0];

            // beta = 1
            icase = 3;
            trial[icase][1] = 1;
            trial[icase][0] = (rhs[0] - M[0][1]*trial[icase][1]) / M[0][0];

            // find the minimum out of four trials
            double d2min = 1e100;
            for (icase = 0; icase < 4; icase++) {
                trial[icase][0] = fmin(fmax(trial[icase][0], 0.0), 1.0);
                trial[icase][1] = fmin(fmax(trial[icase][1], 0.0), 1.0);  
                Vec3 dist = r1 + trial[icase][0]*seg1L - r3 - trial[icase][1]*seg2L;
                double d2 = dist.norm2();
                if (d2 < d2min) {
                    L1 = trial[icase][0];
                    L2 = trial[icase][1];
                    d2min = d2;
                }
            }
            dist2 = d2min;
            didDist2 = 1;
        }
    }

    // Make sure L1 and L2 are between 0 and 1
    L1 = fmin(fmax(L1, 0.0), 1.0);
    L2 = fmin(fmax(L2, 0.0), 1.0);

    if (!didDist2) {
        Vec3 dist = r1 + L1*seg1L - r3 - L2*seg2L;
        dist2 = dist.norm2();
    }

    return dist2;
}

/*------------------------------------------------------------------------
 *
 *    Function:     get_min_dist2_segseg
 *                  Calculate the minimum distance between two segments.
 *                  When hinge=1, the distance for connected segments is
 *                  calculated as the distance between the free node of
 *                  the shorter segment and the other segment, otherwise
 *                  the distance for connected segments is 0.
 *
 *-----------------------------------------------------------------------*/
template<class N>
KOKKOS_INLINE_FUNCTION
double get_min_dist2_segseg(N *net, const int s1, const int s2, bool hinge=0)
{
    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto cell = net->cell;
    
    int n1 = segs[s1].n1;
    int n2 = segs[s1].n2;
    Vec3 r1 = nodes[n1].pos;
    Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
    double l1 = (r2-r1).norm2();
        
    int n3 = segs[s2].n1;
    int n4 = segs[s2].n2;
    Vec3 r3 = cell.pbc_position(r1, nodes[n3].pos);
    Vec3 r4 = cell.pbc_position(r3, nodes[n4].pos);
    double l2 = (r4-r3).norm2();

    double dist2 = -1.0;
    if (l1 >= 1.e-20 && l2 >= 1.e-20) {
        
        int nhinge = 0;
        if (n1 == n3)      nhinge = 1;
        else if (n2 == n3) nhinge = 2;
        else if (n2 == n4) nhinge = 3;
        else if (n1 == n4) nhinge = 4;
        
        if (!nhinge) {
            dist2 = get_min_dist2(r1, r2, r3, r4);
        } else if (!hinge) {
            dist2 = 0.0;
        } else {

            Vec3 m, h1, h2;
            if (nhinge == 1)      { m = r1; h1 = r2; h2 = r4; }
            else if (nhinge == 2) { m = r2; h1 = r1; h2 = r4; }
            else if (nhinge == 3) { m = r2; h1 = r1; h2 = r3; }
            else if (nhinge == 4) { m = r1; h1 = r2; h2 = r3; }

            if (l1 > l2) {
                dist2 = get_min_dist2(m , h1, h2, h2);
            } else {
                dist2 = get_min_dist2(m, h2, h1, h1);
            }
        }
    }
    return dist2;
}

} // namespace ExaDiS

#endif

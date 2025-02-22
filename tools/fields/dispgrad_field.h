/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  This file implements functions to compute displacement gradient fields
 *  due to dislocations.
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_DISPGRAD_FIELD_H
#define EXADIS_DISPGRAD_FIELD_H

#include "system.h"
#include "fields.h"

namespace ExaDiS { namespace tools {

/*---------------------------------------------------------------------------
 *
 *      Function:     DispGradientDueToSeg
 *      Description:  Compute the displacement gradient due to a single
 *                    dislocation segment at a given field point.
 * 					  Coordinate-independant expression.
 * 					  See Bertin and Cai, CMS, 2018.
 *
 *-------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
Mat33 DispGradientDueToSeg(const Vec3& p, const Vec3& p1, const Vec3& p2, 
                           const Vec3& b, double a, double NU)
{
    int    i, j, k;
    double  t[3], d[3];
    double  L2, Linv;
    double  Rx, Ry, Rz;
    double  Rdt, Rdtx, Rdty, Rdtz;
    double  p0x, p0y, p0z;
    double  s1, s2, a2, d2, da2, da2inv;
    double  Ra1, Ra2, Ra1inv, Ra1inv3, Ra2inv, Ra2inv3;
    double  J03, J05, J13, J15, J25, J35;
    double  A, Ab[3][3][3];
    double  U1[3][3], U2[3][3], U3[3][3];
    double  m8pi, m8pinu;

    Mat33 dudx = Mat33().zero();

    t[0] = p2.x - p1.x;
    t[1] = p2.y - p1.y;
    t[2] = p2.z - p1.z;

    L2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
    if (L2 < 1.0e-20) return dudx;
    Linv = 1.0/sqrt(L2);

    t[0] *= Linv;
    t[1] *= Linv;
    t[2] *= Linv;

    Rx = p.x - p1.x;
    Ry = p.y - p1.y;
    Rz = p.z - p1.z;

    Rdt = Rx*t[0] + Ry*t[1] + Rz*t[2];
    Rdtx = Rdt*t[0];
    Rdty = Rdt*t[1];
    Rdtz = Rdt*t[2];

    p0x = p1.x + Rdtx;
    p0y = p1.y + Rdty;
    p0z = p1.z + Rdtz;

    d[0] = Rx - Rdtx;
    d[1] = Ry - Rdty;
    d[2] = Rz - Rdtz;

    s1 = (p1.x - p0x)*t[0] + (p1.y - p0y)*t[1] + (p1.z - p0z)*t[2];
    s2 = (p2.x - p0x)*t[0] + (p2.y - p0y)*t[1] + (p2.z - p0z)*t[2];

    a2 = a*a;
    d2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    da2 = d2 + a2;
    da2inv = 1.0/da2;
    Ra1 = sqrt(s1*s1 + da2);
    Ra2 = sqrt(s2*s2 + da2);
    Ra1inv = 1.0/Ra1;
    Ra1inv3 = Ra1inv*Ra1inv*Ra1inv;
    Ra2inv = 1.0/Ra2;
    Ra2inv3 = Ra2inv*Ra2inv*Ra2inv;

    J03 = da2inv*(s2*Ra2inv - s1*Ra1inv);
    J13 = -Ra2inv + Ra1inv;
    J15 = -1.0/3.0*(Ra2inv3 - Ra1inv3);
    J25 = 1.0/3.0*da2inv*(s2*s2*s2*Ra2inv3 - s1*s1*s1*Ra1inv3);
    J05 = da2inv*(2.0*J25 + s2*Ra2inv3 - s1*Ra1inv3);
    J35 = 2.0*da2*J15 - s2*s2*Ra2inv3 + s1*s1*Ra1inv3;

    for (i = 0; i < 3; i++) {
        A = 3.0*a2*(d[i]*J05 - t[i]*J15) + 2.0*(d[i]*J03 - t[i]*J13);
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                Ab[i][j][k] = -A*t[j]*b[k];
            }
        }
    }

    for (i = 0; i < 3; i++) {
        U1[i][0] = Ab[2][1][i] - Ab[1][2][i];
        U1[i][1] = Ab[0][2][i] - Ab[2][0][i];
        U1[i][2] = Ab[1][0][i] - Ab[0][1][i];
        U2[0][i] = Ab[i][2][1] - Ab[i][1][2];
        U2[1][i] = Ab[i][0][2] - Ab[i][2][0];
        U2[2][i] = Ab[i][1][0] - Ab[i][0][1];
    }

    double B111,B222,B333,B112,B113,B221,B223,B331,B332,B123;

    B111 = -3.0*d[0]*J03+3.0*t[0]*J13+3.0*d[0]*d[0]*d[0]*J05-9.0*(d[0]*d[0]*t[0])*J15
           +9.0*(d[0]*t[0]*t[0])*J25-3.0*t[0]*t[0]*t[0]*J35;

    B222 = -3.0*d[1]*J03+3.0*t[1]*J13+3.0*d[1]*d[1]*d[1]*J05-9.0*(d[1]*d[1]*t[1])*J15
           +9.0*(d[1]*t[1]*t[1])*J25-3.0*t[1]*t[1]*t[1]*J35;

    B333 = -3.0*d[2]*J03+3.0*t[2]*J13+3.0*d[2]*d[2]*d[2]*J05-9.0*(d[2]*d[2]*t[2])*J15
           +9.0*(d[2]*t[2]*t[2])*J25-3.0*t[2]*t[2]*t[2]*J35;

    B112 = -d[1]*J03+t[1]*J13+3.0*d[0]*d[0]*d[1]*J05
           -3.0*(d[0]*d[0]*t[1]+d[0]*t[0]*d[1]+t[0]*d[0]*d[1])*J15
           +3.0*(t[0]*t[0]*d[1]+t[0]*d[0]*t[1]+d[0]*t[0]*t[1])*J25
           -3.0*t[0]*t[0]*t[1]*J35;

    B113 = -d[2]*J03+t[2]*J13+3.0*d[0]*d[0]*d[2]*J05
           -3.0*(d[0]*d[0]*t[2]+d[0]*t[0]*d[2]+t[0]*d[0]*d[2])*J15
           +3.0*(t[0]*t[0]*d[2]+t[0]*d[0]*t[2]+d[0]*t[0]*t[2])*J25
           -3.0*t[0]*t[0]*t[2]*J35;

    B221 = -d[0]*J03+t[0]*J13+3.0*d[1]*d[1]*d[0]*J05
           -3.0*(d[1]*d[1]*t[0]+d[1]*t[1]*d[0]+t[1]*d[1]*d[0])*J15
           +3.0*(t[1]*t[1]*d[0]+t[1]*d[1]*t[0]+d[1]*t[1]*t[0])*J25
           -3.0*t[1]*t[1]*t[0]*J35;

    B223 = -d[2]*J03+t[2]*J13+3.0*d[1]*d[1]*d[2]*J05
           -3.0*(d[1]*d[1]*t[2]+d[1]*t[1]*d[2]+t[1]*d[1]*d[2])*J15
           +3.0*(t[1]*t[1]*d[2]+t[1]*d[1]*t[2]+d[1]*t[1]*t[2])*J25
           -3.0*t[1]*t[1]*t[2]*J35;

    B331 = -d[0]*J03+t[0]*J13+3.0*d[2]*d[2]*d[0]*J05
           -3.0*(d[2]*d[2]*t[0]+d[2]*t[2]*d[0]+t[2]*d[2]*d[0])*J15
           +3.0*(t[2]*t[2]*d[0]+t[2]*d[2]*t[0]+d[2]*t[2]*t[0])*J25
           -3.0*t[2]*t[2]*t[0]*J35;

    B332 = -d[1]*J03+t[1]*J13+3.0*d[2]*d[2]*d[1]*J05
           -3.0*(d[2]*d[2]*t[1]+d[2]*t[2]*d[1]+t[2]*d[2]*d[1])*J15
           +3.0*(t[2]*t[2]*d[1]+t[2]*d[2]*t[1]+d[2]*t[2]*t[1])*J25
           -3.0*t[2]*t[2]*t[1]*J35;

    B123 =  3.0*d[0]*d[1]*d[2]*J05
           -3.0*(d[0]*d[1]*t[2]+d[0]*t[1]*d[2]+t[0]*d[1]*d[2])*J15
           +3.0*(t[0]*t[1]*d[2]+t[0]*d[1]*t[2]+d[0]*t[1]*t[2])*J25
           -3.0*t[0]*t[1]*t[2]*J35;

    U3[0][0] = (B112*t[2]-B113*t[1])*b[0] + (B113*t[0]-B111*t[2])*b[1] + (B111*t[1]-B112*t[0])*b[2];
    U3[1][1] = (B222*t[2]-B223*t[1])*b[0] + (B223*t[0]-B221*t[2])*b[1] + (B221*t[1]-B222*t[0])*b[2];
    U3[2][2] = (B332*t[2]-B333*t[1])*b[0] + (B333*t[0]-B331*t[2])*b[1] + (B331*t[1]-B332*t[0])*b[2];
    U3[0][1] = (B221*t[2]-B123*t[1])*b[0] + (B123*t[0]-B112*t[2])*b[1] + (B112*t[1]-B221*t[0])*b[2];
    U3[0][2] = (B123*t[2]-B331*t[1])*b[0] + (B331*t[0]-B113*t[2])*b[1] + (B113*t[1]-B123*t[0])*b[2];
    U3[1][2] = (B223*t[2]-B332*t[1])*b[0] + (B332*t[0]-B123*t[2])*b[1] + (B123*t[1]-B223*t[0])*b[2];
    U3[1][0] = U3[0][1];
    U3[2][0] = U3[0][2];
    U3[2][1] = U3[1][2];

    m8pi = -0.125/M_PI;
    m8pinu = m8pi/(1.0-NU);
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            dudx[i][j] = m8pi*(U1[i][j] + U2[i][j]) + m8pinu*U3[i][j];
            
    return dudx;
}

/*---------------------------------------------------------------------------
 *
 *    Struct:     DispGradIso
 *
 *-------------------------------------------------------------------------*/
struct DispGradIso {
    typedef Mat33 T_val;
    
    struct Params {
        double NU, a;
        Params() { NU = a = -1.0; }
        Params(double _NU, double _a) : NU(_NU), a(_a) {}
    };
    Params params;
    
    DispGradIso() {}
    DispGradIso(Params _params) {
        params = _params;
        if (params.NU < 0.0 || params.a < 0.0)
            ExaDiS_fatal("Error: invalid DispGradIso() parameter values\n");
    }
    
    KOKKOS_INLINE_FUNCTION
    T_val field_seg_value(const Vec3& r1, const Vec3& r2, const Vec3& b, const Vec3& r) const {
        return DispGradientDueToSeg(r, r1, r2, b, params.a, params.NU);
    }
};

template<class N>
using DispGradFieldGrid = FieldGrid<DispGradIso, N>;

} } // namespace ExaDiS::tools

#endif

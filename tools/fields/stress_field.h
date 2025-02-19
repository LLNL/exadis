/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  This file implements functions to compute stress fields of dislocations.
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_STRESS_FIELD_H
#define EXADIS_STRESS_FIELD_H

#include "system.h"
#include "fields.h"

namespace ExaDiS { namespace tools {

/*---------------------------------------------------------------------------
 *
 *      Function:    StressDueToSeg
 *      Description: Calculate the stress at point p from the segment
 *                   starting at point p1 and ending at point p2.
 *
 *      Arguments:
 *         px, py, pz     coordinates of field point at which stress is to
 *                        be evaluated
 *         p1x, p1y, p1z  starting position of the dislocation segment
 *         p2x, p2y, p2z  ending position of the dislocation segment
 *         bx, by, bz     burgers vector associated with segment going
 *                        from p1 to p2
 *         a              core value
 *         MU             shear modulus
 *         NU             poisson ratio
 *         stress         array of stresses form the indicated segment
 *                        at the field point requested
 *                            [0] = stressxx
 *                            [1] = stressyy
 *                            [2] = stresszz
 *                            [3] = stressxy
 *                            [4] = stressxz
 *                            [5] = stressyz
 *
 *-------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
void StressDueToSeg(const Vec3& p, const Vec3& p1, const Vec3& p2, const Vec3& b,
                    double a, double MU, double NU, double stress[6])
{
    double oneoverLp, common;
    double vec1x, vec1y, vec1z;
    double tpx, tpy, tpz;
    double Rx, Ry, Rz, Rdt;
    double ndx, ndy, ndz;
    double d2, s1, s2, a2, a2_d2, a2d2inv;
    double Ra, Rainv, Ra3inv, sRa3inv;
    double s_03a, s_13a, s_05a, s_15a, s_25a;
    double s_03b, s_13b, s_05b, s_15b, s_25b;
    double s_03, s_13, s_05, s_15, s_25;
    double m4p, m8p, m4pn, mn4pn, a2m8p;
    double txbx, txby, txbz;
    double dxbx, dxby, dxbz;
    double dxbdt, dmdxx, dmdyy, dmdzz, dmdxy, dmdyz, dmdxz;
    double tmtxx, tmtyy, tmtzz, tmtxy, tmtyz, tmtxz;
    double tmdxx, tmdyy, tmdzz, tmdxy, tmdyz, tmdxz;
    double tmtxbxx, tmtxbyy, tmtxbzz, tmtxbxy, tmtxbyz, tmtxbxz;
    double dmtxbxx, dmtxbyy, dmtxbzz, dmtxbxy, dmtxbyz, dmtxbxz;
    double tmdxbxx, tmdxbyy, tmdxbzz, tmdxbxy, tmdxbyz, tmdxbxz;
    double I_03xx, I_03yy, I_03zz, I_03xy, I_03yz, I_03xz;
    double I_13xx, I_13yy, I_13zz, I_13xy, I_13yz, I_13xz;
    double I_05xx, I_05yy, I_05zz, I_05xy, I_05yz, I_05xz;
    double I_15xx, I_15yy, I_15zz, I_15xy, I_15yz, I_15xz;
    double I_25xx, I_25yy, I_25zz, I_25xy, I_25yz, I_25xz;

    vec1x = p2.x - p1.x;
    vec1y = p2.y - p1.y;
    vec1z = p2.z - p1.z;

    oneoverLp = 1 / sqrt(vec1x*vec1x + vec1y*vec1y + vec1z*vec1z);

    tpx = vec1x * oneoverLp;
    tpy = vec1y * oneoverLp;
    tpz = vec1z * oneoverLp;

    Rx = p.x - p1.x;
    Ry = p.y - p1.y;
    Rz = p.z - p1.z;

    Rdt = Rx*tpx + Ry*tpy + Rz*tpz;

    ndx = Rx - Rdt*tpx;
    ndy = Ry - Rdt*tpy;
    ndz = Rz - Rdt*tpz;

    d2 = ndx*ndx + ndy*ndy + ndz*ndz;

    s1 = -Rdt;
    s2 = -((p.x-p2.x)*tpx + (p.y-p2.y)*tpy + (p.z-p2.z)*tpz);
    a2 = a * a;
    a2_d2 = a2 + d2;
    a2d2inv = 1 / a2_d2;

    Ra = sqrt(a2_d2 + s1*s1);
    Rainv = 1 / Ra;
    Ra3inv = Rainv * Rainv * Rainv;
    sRa3inv = s1 * Ra3inv;

    s_03a = s1 * Rainv * a2d2inv;
    s_13a = -Rainv;
    s_05a = (2*s_03a + sRa3inv) * a2d2inv;
    s_15a = -Ra3inv;
    s_25a = s_03a - sRa3inv;

    Ra = sqrt(a2_d2 + s2*s2);
    Rainv = 1 / Ra;
    Ra3inv = Rainv * Rainv * Rainv;
    sRa3inv = s2 * Ra3inv;

    s_03b = s2 * Rainv * a2d2inv;
    s_13b = -Rainv;
    s_05b = (2*s_03b + sRa3inv) * a2d2inv;
    s_15b = -Ra3inv;
    s_25b = s_03b - sRa3inv;

    s_03 = s_03b - s_03a;
    s_13 = s_13b - s_13a;
    s_05 = s_05b - s_05a;
    s_15 = s_15b - s_15a;
    s_25 = s_25b - s_25a;

    m4p = 0.25 * MU / M_PI;
    m8p = 0.5 * m4p;
    m4pn = m4p / (1 - NU);
    mn4pn = m4pn * NU;
    a2m8p = a2 * m8p;

    txbx = tpy*b.z - tpz*b.y;
    txby = tpz*b.x - tpx*b.z;
    txbz = tpx*b.y - tpy*b.x;

    dxbx = ndy*b.z - ndz*b.y;
    dxby = ndz*b.x - ndx*b.z;
    dxbz = ndx*b.y - ndy*b.x;

    dxbdt = dxbx*tpx + dxby*tpy + dxbz*tpz;

    dmdxx = ndx * ndx;
    dmdyy = ndy * ndy;
    dmdzz = ndz * ndz;
    dmdxy = ndx * ndy;
    dmdyz = ndy * ndz;
    dmdxz = ndx * ndz;

    tmtxx = tpx * tpx;
    tmtyy = tpy * tpy;
    tmtzz = tpz * tpz;
    tmtxy = tpx * tpy;
    tmtyz = tpy * tpz;
    tmtxz = tpx * tpz;

    tmdxx = 2 * tpx * ndx;
    tmdyy = 2 * tpy * ndy;
    tmdzz = 2 * tpz * ndz;
    tmdxy = tpx*ndy + tpy*ndx;
    tmdyz = tpy*ndz + tpz*ndy;
    tmdxz = tpx*ndz + tpz*ndx;

    tmtxbxx = 2 * tpx * txbx;
    tmtxbyy = 2 * tpy * txby;
    tmtxbzz = 2 * tpz * txbz;
    tmtxbxy = tpx*txby + tpy*txbx;
    tmtxbyz = tpy*txbz + tpz*txby;
    tmtxbxz = tpx*txbz + tpz*txbx;

    dmtxbxx = 2 * ndx * txbx;
    dmtxbyy = 2 * ndy * txby;
    dmtxbzz = 2 * ndz * txbz;
    dmtxbxy = ndx*txby + ndy*txbx;
    dmtxbyz = ndy*txbz + ndz*txby;
    dmtxbxz = ndx*txbz + ndz*txbx;

    tmdxbxx = 2 * tpx * dxbx;
    tmdxbyy = 2 * tpy * dxby;
    tmdxbzz = 2 * tpz * dxbz;
    tmdxbxy = tpx*dxby + tpy*dxbx;
    tmdxbyz = tpy*dxbz + tpz*dxby;
    tmdxbxz = tpx*dxbz + tpz*dxbx;

    common = m4pn * dxbdt;

    I_03xx = common + m4pn*dmtxbxx - m4p*tmdxbxx;
    I_03yy = common + m4pn*dmtxbyy - m4p*tmdxbyy;
    I_03zz = common + m4pn*dmtxbzz - m4p*tmdxbzz;
    I_03xy = m4pn*dmtxbxy - m4p*tmdxbxy;
    I_03yz = m4pn*dmtxbyz - m4p*tmdxbyz;
    I_03xz = m4pn*dmtxbxz - m4p*tmdxbxz;

    I_13xx = -mn4pn * tmtxbxx;
    I_13yy = -mn4pn * tmtxbyy;
    I_13zz = -mn4pn * tmtxbzz;
    I_13xy = -mn4pn * tmtxbxy;
    I_13yz = -mn4pn * tmtxbyz;
    I_13xz = -mn4pn * tmtxbxz;

    I_05xx = common*(a2+dmdxx) - a2m8p*tmdxbxx;
    I_05yy = common*(a2+dmdyy) - a2m8p*tmdxbyy;
    I_05zz = common*(a2+dmdzz) - a2m8p*tmdxbzz;
    I_05xy = common*dmdxy - a2m8p*tmdxbxy;
    I_05yz = common*dmdyz - a2m8p*tmdxbyz;
    I_05xz = common*dmdxz - a2m8p*tmdxbxz;

    I_15xx = a2m8p*tmtxbxx - common*tmdxx;
    I_15yy = a2m8p*tmtxbyy - common*tmdyy;
    I_15zz = a2m8p*tmtxbzz - common*tmdzz;
    I_15xy = a2m8p*tmtxbxy - common*tmdxy;
    I_15yz = a2m8p*tmtxbyz - common*tmdyz;
    I_15xz = a2m8p*tmtxbxz - common*tmdxz;

    I_25xx = common * tmtxx;
    I_25yy = common * tmtyy;
    I_25zz = common * tmtzz;
    I_25xy = common * tmtxy;
    I_25yz = common * tmtyz;
    I_25xz = common * tmtxz;

    stress[0] = I_03xx*s_03 + I_13xx*s_13 + I_05xx*s_05 +
                I_15xx*s_15 + I_25xx*s_25;

    stress[1] = I_03yy*s_03 + I_13yy*s_13 + I_05yy*s_05 +
                I_15yy*s_15 + I_25yy*s_25;

    stress[2] = I_03zz*s_03 + I_13zz*s_13 + I_05zz*s_05 +
                I_15zz*s_15 + I_25zz*s_25;

    stress[3] = I_03xy*s_03 + I_13xy*s_13 + I_05xy*s_05 +
                I_15xy*s_15 + I_25xy*s_25;

    stress[5] = I_03yz*s_03 + I_13yz*s_13 + I_05yz*s_05 +
                I_15yz*s_15 + I_25yz*s_25;

    stress[4] = I_03xz*s_03 + I_13xz*s_13 + I_05xz*s_05 +
                I_15xz*s_15 + I_25xz*s_25;
}

/*---------------------------------------------------------------------------
 *
 *    Struct:     StressIso
 *
 *-------------------------------------------------------------------------*/
struct StressIso {
    typedef Mat33 T_val;
    
    struct Params {
        double MU, NU, a;
        Params() { MU = NU = a = -1.0; }
        Params(double _MU, double _NU, double _a) : MU(_MU), NU(_NU), a(_a) {}
    };
    Params params;
    
    StressIso(Params _params) {
        params = _params;
        if (params.MU < 0.0 || params.NU < 0.0 || params.a < 0.0)
            ExaDiS_fatal("Error: invalid StressIso() parameter values\n");
    }
    
    KOKKOS_INLINE_FUNCTION
    T_val field_seg_value(const Vec3& r1, const Vec3& r2, const Vec3& b, const Vec3& r) {
        double stress[6];
        StressDueToSeg(r, r1, r2, b, params.a, params.MU, params.NU, stress);
        return Mat33().symmetric(stress);
    }
};

template<class N>
using StressFieldGrid = FieldGrid<StressIso, N>;

} } // namespace ExaDiS::tools

#endif

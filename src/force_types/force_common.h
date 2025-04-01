/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_FORCE_COMMON_H
#define EXADIS_FORCE_COMMON_H

#include "vec.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Function:     self_force()
 *                  Self-force coming from the non-singular self-stress
 *
 *-------------------------------------------------------------------------*/
KOKKOS_FORCEINLINE_FUNCTION
Vec3 self_force(const Vec3 &b, const Vec3 &t, const double L, 
                double MU, double NU, double a)
{
    double bs = dot(b, t);
    Vec3 be = b - bs*t;

    double La = sqrt(L*L+a*a);
    double S = (-(2*NU*La+(1-NU)*a*a/La-(1+NU)*a)/L +
               (NU*log((La+L)/a)-(1-NU)*0.5*L/La))*MU/4/M_PI/(1-NU)*bs;
    
    Vec3 f = S*be;
    return -1.0*f; // return f1, f2=-f1
}

/*---------------------------------------------------------------------------
 *
 *    Function:     pk_force()
 *                  Peach-Koehler force
 *
 *-------------------------------------------------------------------------*/
KOKKOS_FORCEINLINE_FUNCTION
Vec3 pk_force(const Vec3 &b, const Vec3 &r1, const Vec3 &r2, const Mat33 &stress)
{
    Vec3 sigb = stress * b;
    return 0.5*cross(sigb, r2-r1);
}

/*------------------------------------------------------------------------
 *
 *    Function:     SegSegForceIsotropicCorr
 *
 *-----------------------------------------------------------------------*/
KOKKOS_FORCEINLINE_FUNCTION
void SegSegForceIsotropicCorr(const Vec3 &r1, const Vec3 &r2, const Vec3 &r3, const Vec3 &r4, 
                              const Vec3 &b1, const Vec3 &b2, double a, double MU, double NU,
                              Vec3 &fn1, Vec3 &fn2, Vec3 &fn3, Vec3 &fn4)
{
        double x1[3], x2[3], x3[3], x4[3], b[3], bp[3];
        double f3[3], f4[3];
        double vec1[3], vec2[3], t[3], tp[3], tctp[3];
        double R[2][3], tempa[2], tempb[2], y[2], z[2];
        int i, j , alt1[3]={1,2,0}, alt2[3]={2,0,1};
        double d, c, c2, onemc2, onemc2inv, oneoverL, oneoverLp;
        double a2, m4p, m4pd, m8p, m8pd, m4pn, m4pnd, m4pnd2, m4pnd3;
        double a2m4pnd, a2m8pd, a2m4pn, a2m8p, a2_d2, a2_d2inv, denom;
        double temp1, temp2, temp3, temp4[8], tmp[10];
        double yv[4], zv[4], y2[4], z2[4], Ra[4], Rainv[4];
        double Ra_Rdot_tp[8], Ra_Rdot_t[8], log_Ra_Rdot_tp[4], log_Ra_Rdot_t[4];
        double Ra2_R_tpinv[4], Ra2_R_tinv[4], ylog_Ra_Rdot_tp[4], zlog_Ra_Rdot_t[4];
        double yRa2_R_tpinv[4], zRa2_R_tinv[4], y2Ra2_R_tpinv[4], z2Ra2_R_tinv[4];
        double adf_003[4], commonf223[4], commonf225[4], commonf025[4], commonf205[4];
        double commonf305[4], commonf035[4], ycommonf025[4], zcommonf205[4], zcommonf305[4];
        double tf_113[4];
        double f_003v[4], f_103v[4], f_013v[4], f_113v[4];
        double f_203v[4], f_023v[4], f_005v[4], f_105v[4];
        double f_003,  f_103,  f_013,  f_113,  f_203,  /*f_023,*/  f_005,  f_105;
        double f_015v[4], f_115v[4], f_205v[4], f_025v[4];
        double f_215v[4], f_125v[4], f_225v[4], f_305v[4];
        double f_015,  f_115,  f_205,  f_025,  f_215,  f_125,  f_225,  f_305;
        double f_035v[4], f_315v[4], f_135v[4];
        double f_035,  f_315,  f_135;
        double Fint_003, Fint_005, Fint_013, Fint_015, Fint_025, Fint_103;
        double Fint_105, Fint_115, Fint_125, Fint_205, Fint_215;
        double I_003[3], I_005[3], I_013[3], I_015[3], I_025[3], I_103[3];
        double I_105[3], I_115[3], I_125[3], I_205[3], I_215[3];
        double I00a[3], I01a[3], I10a[3], I00b[3], /*I01b[3],*/ I10b[3];
        double /*bctctp[3],*/ bct[3], bpctpct[3], bpctp[3], tcbpct[3];
        double bctdbp, bpctpdb, tcbpdb, tcbpdtp /*, tpcbdbp*/;
        double tctpct[3], tpct[3];
        double tctpcbpdb, tctpcbpdtp, tctpdb, tdb, tdbp;
        //double tpcbctp[3], tpctctp[3];
        double /*tpcbdt, tpctcbdbp, tpctcbdt,*/ tpctdbp, tpdb, tpdbp;
        double pivalue=3.141592653589793;

        fn1.x = 0.0;
        fn1.y = 0.0;
        fn1.z = 0.0;

        fn2.x = 0.0;
        fn2.y = 0.0;
        fn2.z = 0.0;

        fn3.x = 0.0;
        fn3.y = 0.0;
        fn3.z = 0.0;

        fn4.x = 0.0;
        fn4.y = 0.0;
        fn4.z = 0.0;

        x1[0]=r1.x;
        x1[1]=r1.y;
        x1[2]=r1.z;
        x2[0]=r2.x;
        x2[1]=r2.y;
        x2[2]=r2.z;
        x3[0]=r3.x;
        x3[1]=r3.y;
        x3[2]=r3.z;
        x4[0]=r4.x;
        x4[1]=r4.y;
        x4[2]=r4.z;

        b[0]=b2.x;
        b[1]=b2.y;
        b[2]=b2.z;
        bp[0]=b1.x;
        bp[1]=b1.y;
        bp[2]=b1.z;

        #pragma unroll
        for(i=0;i<3;i++) {
            vec1[i]=x4[i]-x3[i];
            vec2[i]=x2[i]-x1[i];
        }

        temp1=0.0e0;
        temp2=0.0e0;

        #pragma unroll
        for(i=0;i<3;i++) {
            temp1+=vec1[i]*vec1[i];
            temp2+=vec2[i]*vec2[i];
        }

        oneoverL =1/sqrt(temp1);
        oneoverLp=1/sqrt(temp2);

        #pragma unroll
        for(i=0;i<3;i++) {
            t[i]=vec1[i]*oneoverL;
            tp[i]=vec2[i]*oneoverLp;
        }

        c=0.0e0;

        #pragma unroll
        for(i=0;i<3;i++) {
            c+=t[i]*tp[i];
        }

        c2=c*c;
        onemc2=1-c2;

        {
			#pragma unroll
            for(i=0;i<3;i++) {
                tctp[i]=t[alt1[i]]*tp[alt2[i]]-t[alt2[i]]*tp[alt1[i]];
            }

            onemc2inv = 1/onemc2;

            #pragma unroll
            for(i=0;i<3;i++) {
                R[0][i]=x3[i]-x1[i];
                R[1][i]=x4[i]-x2[i];
            }

            d=0.0e0;

            #pragma unroll
            for (j=0;j<2;j++) {
                tempa[j]=0.0e0;
                tempb[j]=0.0e0;
            }

            #pragma unroll
            for(i=0;i<3;i++) {
                d+=0.5e0*((x4[i]+x3[i])-(x2[i]+x1[i]))*tctp[i];
                for (j=0;j<2;j++) {
                    tempa[j]+=R[j][i]*t[i];
                    tempb[j]+=R[j][i]*tp[i];
                }
            }

            d*=onemc2inv;

            #pragma unroll
            for (j=0;j<2;j++) {
                y[j]=(tempa[j]-c*tempb[j])*onemc2inv;
                z[j]=(tempb[j]-c*tempa[j])*onemc2inv;
            }

/*          now we calculate the definite integrals of the force calculation  */

            #pragma unroll
            for (j=0;j<2;j++) {
                yv[2*j]=y[j];
                yv[2*j+1]=y[j];
                zv[j]=z[j];
                zv[j+2]=z[j];
            }

            a2_d2 = a*a+d*d*onemc2;

            #pragma unroll
            for (j=0;j<4;j++) {
                y2[j] = yv[j]*yv[j];
                z2[j] = zv[j]*zv[j];

            }

            #pragma unroll
            for (j=0;j<4;j++) {
                temp4[j]=a2_d2 + y2[j] + z2[j] + 2.0e0*yv[j]*zv[j]*c;
            }

            temp1=onemc2*a2_d2;

            #pragma unroll
            for (j=0;j<4;j++) {
                Ra[j]=sqrt(temp4[j]);
            }

            temp2=sqrt(temp1);

            #pragma unroll
            for (j=0;j<4;j++) {
                Rainv[j]=1.0e0/Ra[j];
            }

            denom=1.0e0/temp2;
            a2_d2inv=1.0e0/a2_d2;

            #pragma unroll
            for (j=0;j<4;j++) {
                Ra_Rdot_tp[j] = Ra[j]+(zv[j]+yv[j]*c);
                Ra_Rdot_t[j]  = Ra[j]+(yv[j]+zv[j]*c);
				Ra_Rdot_tp[j+4] = Ra[j]-(zv[j]+yv[j]*c);
                Ra_Rdot_t[j+4]  = Ra[j]-(yv[j]+zv[j]*c);
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                log_Ra_Rdot_tp[j] =0.5e0*(log(Ra_Rdot_tp[j])-log(Ra_Rdot_tp[j+4]));
                log_Ra_Rdot_t[j]  =0.5e0*(log(Ra_Rdot_t[j])-log(Ra_Rdot_t[j+4]));
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                Ra2_R_tpinv[j] = 0.5e0*(Rainv[j]/Ra_Rdot_tp[j]- Rainv[j]/Ra_Rdot_tp[j+4]);
                Ra2_R_tinv[j] =  0.5e0*(Rainv[j]/Ra_Rdot_t[j]- Rainv[j]/Ra_Rdot_t[j+4]);
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                ylog_Ra_Rdot_tp[j] = yv[j]*log_Ra_Rdot_tp[j];
                yRa2_R_tpinv[j]    = yv[j]*   Ra2_R_tpinv[j];
                zlog_Ra_Rdot_t[j]  = zv[j]*log_Ra_Rdot_t[j];
                zRa2_R_tinv[j]     = zv[j]*   Ra2_R_tinv[j];

            }

            #pragma unroll
            for (j=0;j<4;j++) {
                y2Ra2_R_tpinv[j] = yv[j]* yRa2_R_tpinv[j];
                z2Ra2_R_tinv[j]  = zv[j]*  zRa2_R_tinv[j];
            }

            temp1=denom*(1+c);

            #pragma unroll
            for (j=0;j<4;j++) {
                temp4[j]=temp1*(Ra[j]+(yv[j]+zv[j]));
				temp4[j+4]=temp1*(Ra[j]-(yv[j]+zv[j]));
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                f_003v[j]=0.5e0*(atan(temp4[j])+atan(temp4[j+4]));
            }

            temp1=-2.0e0*denom;

            #pragma unroll
            for (j=0;j<4;j++) {
                f_003v[j]*=temp1;
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                adf_003[j]=f_003v[j]*a2_d2;
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                commonf223[j] = c*Ra[j] - adf_003[j];
                f_103v[j] = c*log_Ra_Rdot_t[j]  - log_Ra_Rdot_tp[j];
                f_013v[j] = c*log_Ra_Rdot_tp[j] - log_Ra_Rdot_t [j];
                f_113v[j] = c*adf_003[j] - Ra[j];
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                commonf223[j] *= onemc2inv;
                f_103v[j] *=      onemc2inv;
                f_013v[j] *=      onemc2inv;
                f_113v[j] *=      onemc2inv;
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                commonf225[j] = f_003v[j] - c*Rainv[j];
                commonf025[j] = c*yRa2_R_tpinv[j] - Rainv[j];
                commonf205[j] = c*zRa2_R_tinv[j]  - Rainv[j];
                commonf305[j] = log_Ra_Rdot_t[j]  -(yv[j]-c*zv[j])*Rainv[j] - c2*z2Ra2_R_tinv[j];
                commonf035[j] = log_Ra_Rdot_tp[j] -(zv[j]-c*yv[j])*Rainv[j] - c2*y2Ra2_R_tpinv[j];
                f_203v[j] =  zlog_Ra_Rdot_t[j]  + commonf223[j];
                f_023v[j] =  ylog_Ra_Rdot_tp[j] + commonf223[j];
                f_005v[j] = f_003v[j] - yRa2_R_tpinv[j] - zRa2_R_tinv[j];
                f_105v[j] = Ra2_R_tpinv[j] - c*Ra2_R_tinv[j];
                f_015v[j] = Ra2_R_tinv[j]  - c*Ra2_R_tpinv[j];
                f_115v[j] = Rainv[j] - c*(yRa2_R_tpinv[j] + zRa2_R_tinv[j] + f_003v[j]);
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                ycommonf025[j] = yv[j]*commonf025[j];
                zcommonf205[j] = zv[j]*commonf205[j];
                zcommonf305[j] = zv[j]*commonf305[j];
                tf_113[j]=2.0e0*f_113v[j];
                f_205v[j] = yRa2_R_tpinv[j] + c2*zRa2_R_tinv[j]  + commonf225[j];
                f_025v[j] = zRa2_R_tinv[j]  + c2*yRa2_R_tpinv[j] + commonf225[j];
                f_305v[j] = y2Ra2_R_tpinv[j] + c*commonf305[j] + 2.0e0*f_103v[j];
                f_035v[j] = z2Ra2_R_tinv[j]  + c*commonf035[j] + 2.0e0*f_013v[j];
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                f_215v[j] = f_013v[j] - ycommonf025[j] + c*(zcommonf205[j]-f_103v[j]);
                f_125v[j] = f_103v[j] - zcommonf205[j] + c*(ycommonf025[j] - f_013v[j]);
                f_225v[j] = f_203v[j] - zcommonf305[j] + c*(y2[j]*commonf025[j] - tf_113[j]);
                f_315v[j] = tf_113[j] - y2[j]*commonf025[j] + c*(zcommonf305[j] - f_203v[j]);
                f_135v[j] = tf_113[j] - z2[j]*commonf205[j] + c*(yv[j]*commonf035[j]-f_023v[j]);
            }


            f_003= (f_003v[0]+f_003v[3])-(f_003v[1]+f_003v[2]);
            f_013= (f_013v[0]+f_013v[3])-(f_013v[1]+f_013v[2]);
            f_103= (f_103v[0]+f_103v[3])-(f_103v[1]+f_103v[2]);
            f_113= (f_113v[0]+f_113v[3])-(f_113v[1]+f_113v[2]);
            //f_023= (f_023v[0]+f_023v[3])-(f_023v[1]+f_023v[2]);
            f_203= (f_203v[0]+f_203v[3])-(f_203v[1]+f_203v[2]);
            f_005= (f_005v[0]+f_005v[3])-(f_005v[1]+f_005v[2]);
            f_015= (f_015v[0]+f_015v[3])-(f_015v[1]+f_015v[2]);
            f_105= (f_105v[0]+f_105v[3])-(f_105v[1]+f_105v[2]);
            f_115= (f_115v[0]+f_115v[3])-(f_115v[1]+f_115v[2]);
            f_025= (f_025v[0]+f_025v[3])-(f_025v[1]+f_025v[2]);
            f_205= (f_205v[0]+f_205v[3])-(f_205v[1]+f_205v[2]);
            f_215= (f_215v[0]+f_215v[3])-(f_215v[1]+f_215v[2]);
            f_125= (f_125v[0]+f_125v[3])-(f_125v[1]+f_125v[2]);
            f_035= (f_035v[0]+f_035v[3])-(f_035v[1]+f_035v[2]);
            f_305= (f_305v[0]+f_305v[3])-(f_305v[1]+f_305v[2]);
            f_225= (f_225v[0]+f_225v[3])-(f_225v[1]+f_225v[2]);
            f_135= (f_135v[0]+f_135v[3])-(f_135v[1]+f_135v[2]);
            f_315= (f_315v[0]+f_315v[3])-(f_315v[1]+f_315v[2]);


            f_005 *= a2_d2inv;
            f_105 *= onemc2inv;
            f_015 *= onemc2inv;
            f_115 *= onemc2inv;
            f_205 *= onemc2inv;
            f_025 *= onemc2inv;
            f_305 *= onemc2inv;
            f_035 *= onemc2inv;
            f_215 *= onemc2inv;
            f_125 *= onemc2inv;
            f_225 *= onemc2inv;
            f_315 *= onemc2inv;
            f_135 *= onemc2inv;


/* now construct the vector coefficients for the definite integrals */

            a2 = a*a;
            m4p = 0.25 * MU / pivalue;
            m4pd =  m4p * d;
            m8p = 0.5 * m4p;
            m8pd = m8p * d;
            m4pn = m4p / ( 1 - NU );
            m4pnd = m4pn * d;
            m4pnd2 = m4pnd * d;
            m4pnd3 = m4pnd2 * d;
            a2m4pnd = a2 * m4pnd;
            a2m8pd = a2 * m8pd;
            a2m4pn = a2 * m4pn;
            a2m8p = a2 * m8p;

            #pragma unroll
            for (i=0;i<3;i++) {
                tpct[i]=-tctp[i];
                bct[i]=b[alt1[i]]*t[alt2[i]]-b[alt2[i]]*t[alt1[i]];
                bpctp[i]=bp[alt1[i]]*tp[alt2[i]]-bp[alt2[i]]*tp[alt1[i]];

            }

            tdb=0.0e0;
            tdbp=0.0e0;
            tpdb=0.0e0;
            tpdbp=0.0e0;
            tctpdb=0.0e0;
            tpctdbp=0.0e0;
            bpctpdb=0.0e0;
            bctdbp=0.0e0;

            #pragma unroll
            for (i=0;i<3;i++) {
                tdb    +=t[i]*b[i];
                tdbp   +=t[i]*bp[i];
                tpdb   +=tp[i]*b[i];
                tpdbp  +=tp[i]*bp[i];
                tctpdb +=tctp[i]*b[i];
                tpctdbp+=tpct[i]*bp[i];
                bpctpdb+=bpctp[i]*b[i];
                bctdbp +=bct[i]*bp[i];
            }

            #pragma unroll
            for (i=0;i<3;i++) {
                tctpct[i]    =        tp[i] -     c*t[i];
                //tpctctp[i]   =         t[i] -    c*tp[i];
                tcbpct[i]    =        bp[i] -  tdbp*t[i];
                //tpcbctp[i]   =         b[i] - tpdb*tp[i];
                bpctpct[i]   =   tdbp*tp[i] -    c*bp[i];
                //bctctp[i]    =    tpdb*t[i] -     c*b[i];
            }


            tctpcbpdtp = tdbp - tpdbp*c;
            //tpctcbdt = tpdb - tdb*c;
            tctpcbpdb =  tdbp*tpdb - tpdbp*tdb;
            //tpctcbdbp = tctpcbpdb;
            tcbpdtp = tpctdbp;
            //tpcbdt = tctpdb;
            tcbpdb = bctdbp;
            //tpcbdbp = bpctpdb;

/*
 *          Only calculate the forces for segment p3->p4 if at least one
 *          of the segment's nodes is local to the current domain.
 */
            //if (seg34Local) {

                temp1 = tdbp*tpdb + tctpcbpdb;

                #pragma unroll
                for (i=0;i<3;i++) {
                    I00a[i] = temp1 * tpct[i];
                    I00b[i] = tctpcbpdtp * bct[i];
                }

                temp1 = (m4pnd * tctpdb);
                temp2 = (m4pnd * bpctpdb);
                temp3 = (m4pnd3 * tctpcbpdtp*tctpdb);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_003[i] = m4pd*I00a[i] - m4pnd*I00b[i] + temp1*bpctpct[i] +
                            temp2*tctpct[i];
                    I_005[i] = a2m8pd*I00a[i] - a2m4pnd*I00b[i] - temp3*tctpct[i];
                    I10a[i] = tcbpct[i]*tpdb - tctp[i]*tcbpdb;
                    I10b[i] = bct[i] * tcbpdtp;

                }

                temp1 = (m4pn * tdb);
                temp2 = m4pnd2 * (tcbpdtp*tctpdb + tctpcbpdtp*tdb);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_103[i] = temp1*bpctpct[i] + m4p*I10a[i] - m4pn*I10b[i];
                    I_105[i] = a2m8p*I10a[i] - a2m4pn*I10b[i] - temp2*tctpct[i];
                    I01a[i] = tctp[i]*bpctpdb - bpctpct[i]*tpdb;
                }

                tmp[0] = (m4pn * tpdb);
                tmp[1] = (m4pn * bpctpdb);
                tmp[2] = (m4pnd2 * tctpcbpdtp * tpdb);
                tmp[3] = (m4pnd2 * tctpcbpdtp * tctpdb);
                tmp[4] = (m4pnd * tcbpdtp * tdb);
                tmp[5] = (m4pnd * tctpcbpdtp * tpdb) ;
                tmp[6] = (m4pnd * (tctpcbpdtp*tdb + tcbpdtp*tctpdb));
                tmp[7] = (m4pnd * tcbpdtp * tpdb);
                tmp[8] = (m4pn * tcbpdtp * tdb);
                tmp[9] = (m4pn * tcbpdtp * tpdb);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_013[i] = m4p*I01a[i] + tmp[0]*bpctpct[i] - tmp[1]*tctp[i];
                    I_015[i] = a2m8p*I01a[i] - tmp[2]*tctpct[i] + tmp[3]*tctp[i];
                    I_205[i] = -tmp[4] * tctpct[i];
                    I_025[i] = tmp[5] * tctp[i];
                    I_115[i] =  tmp[6]*tctp[i] - tmp[7]*tctpct[i];
                    I_215[i] = tmp[8] * tctp[i];
                    I_125[i] = tmp[9] * tctp[i];
                }

                Fint_003 = f_103 - y[0]*f_003;
                Fint_103 = f_203 - y[0]*f_103;
                Fint_013 = f_113 - y[0]*f_013;
                Fint_005 = f_105 - y[0]*f_005;
                Fint_105 = f_205 - y[0]*f_105;
                Fint_015 = f_115 - y[0]*f_015;
                Fint_115 = f_215 - y[0]*f_115;
                Fint_205 = f_305 - y[0]*f_205;
                Fint_025 = f_125 - y[0]*f_025;
                Fint_215 = f_315 - y[0]*f_215;
                Fint_125 = f_225 - y[0]*f_125;

                #pragma unroll
                for (i=0;i<3;i++) {
                    f4[i]=(I_003[i]*Fint_003 + I_103[i]*Fint_103 + I_013[i]*Fint_013 +
                           I_005[i]*Fint_005 + I_105[i]*Fint_105 + I_015[i]*Fint_015 +
                           I_115[i]*Fint_115 + I_205[i]*Fint_205 + I_025[i]*Fint_025 +
                           I_215[i]*Fint_215 + I_125[i]*Fint_125) * oneoverL;
                }

                Fint_003 = y[1]*f_003 - f_103;
                Fint_103 = y[1]*f_103 - f_203;
                Fint_013 = y[1]*f_013 - f_113;
                Fint_005 = y[1]*f_005 - f_105;
                Fint_105 = y[1]*f_105 - f_205;
                Fint_015 = y[1]*f_015 - f_115;
                Fint_115 = y[1]*f_115 - f_215;
                Fint_205 = y[1]*f_205 - f_305;
                Fint_025 = y[1]*f_025 - f_125;
                Fint_215 = y[1]*f_215 - f_315;
                Fint_125 = y[1]*f_125 - f_225;

                #pragma unroll
                for (i=0;i<3;i++) {
                    f3[i]=(I_003[i]*Fint_003 + I_103[i]*Fint_103 + I_013[i]*Fint_013 +
                           I_005[i]*Fint_005 + I_105[i]*Fint_105 + I_015[i]*Fint_015 +
                           I_115[i]*Fint_115 + I_205[i]*Fint_205 + I_025[i]*Fint_025 +
                           I_215[i]*Fint_215 + I_125[i]*Fint_125) * oneoverL;
                }

                fn3.x=f3[0];
                fn3.y=f3[1];
                fn3.z=f3[2];
                fn4.x=f4[0];
                fn4.y=f4[1];
                fn4.z=f4[2];

            //} /* if segment p3->p4 is "local" */
       }

       return;
}

/*------------------------------------------------------------------------
 *
 *    Function:     SpecialSegSegForceHalf
 *
 *-----------------------------------------------------------------------*/
KOKKOS_FORCEINLINE_FUNCTION
void SpecialSegSegForceHalf(const Vec3 &r1, const Vec3 &r2, const Vec3 &r3, const Vec3 &r4,
                            const Vec3 &b1, const Vec3 &b2, double a, double MU, double NU,
                            Vec3 &fn3, Vec3 &fn4)
{
        int i, j , alt1[3]={1,2,0}, alt2[3]={2,0,1};
        double eps, ecrit, c, a2, d2, a2_d2, a2d2inv;
        double x1[3], x2[3], x3[3], x4[3], b[3], bp[3];
        double f3[3], f4[3];
        double vec1[3], vec2[3], t[3], nd[3];
        double temp1;
        double R[3], Rdt, x1mod[3], x2mod[3];
        double oneoverL;
        double y[2], z[2], yv[4], zv[4], ypz[4], ymz[4];
        double Ra[4], Rainv[4], Log_Ra_ypz[4];
        double temp, tmp[8];
        double common1[4], common2[3], common3[3];
        double magdiff, diffMag2, x1modMag2, x2modMag2;
        double f_003v[4], f_103v[4], f_113v[4], f_213v[4];
        double f_005v[4], f_105v[4], f_115v[4], f_215v[4];
        double f_003, f_103, f_113, f_213;
        double f_005, f_105, f_115, f_215;
        double Fint_003, Fint_113, Fint_005, Fint_115;
        double I_003[3], I_113[3], I_005[3], I_115[3];
        double m4p, m8p, m4pn, a2m4pn, a2m8p;
        double tdb, tdbp, nddb, bpctdb, bpctdnd;
        double bct[3], bpct[3], ndct[3], bpctct[3];
        double cotanthetac;
        double pivalue=3.141592653589793;

		ecrit  = 1e-4;
        cotanthetac = sqrt((1 - ecrit*1.01) / (ecrit*1.01));

        eps    = 1e-12;
        a2     = a*a;
        m4p    = 0.25 * MU / pivalue;
        m8p    = 0.5 * m4p;
        m4pn   = m4p / ( 1 - NU );
        a2m4pn = a2 * m4pn;
        a2m8p  = a2 * m8p;

        fn3.x = 0.0;
        fn3.y = 0.0;
        fn3.z = 0.0;

        fn4.x = 0.0;
        fn4.y = 0.0;
        fn4.z = 0.0;

        x1[0]=r1.x;
        x1[1]=r1.y;
        x1[2]=r1.z;
        x2[0]=r2.x;
        x2[1]=r2.y;
        x2[2]=r2.z;
        x3[0]=r3.x;
        x3[1]=r3.y;
        x3[2]=r3.z;
        x4[0]=r4.x;
        x4[1]=r4.y;
        x4[2]=r4.z;

        b[0]=b2.x;
        b[1]=b2.y;
        b[2]=b2.z;
        bp[0]=b1.x;
        bp[1]=b1.y;
        bp[2]=b1.z;

        #pragma unroll
        for(i=0;i<3;i++) {
            vec1[i]=x4[i]-x3[i];
            vec2[i]=x2[i]-x1[i];
        }

        temp1=0.0e0;

        #pragma unroll
        for(i=0;i<3;i++) {
            temp1+=vec1[i]*vec1[i];
        }

        oneoverL =1/sqrt(temp1);

        for(i=0;i<3;i++) {
            t[i]=vec1[i]*oneoverL;
        }

        c=0.0e0;

        #pragma unroll
        for(i=0;i<3;i++) {
            c+=t[i]*vec2[i];
        }

        if (c < 0) {
			#pragma unroll
            for(i=0;i<3;i++) {
                temp=x2[i];
                x2[i]=x1[i];
                x1[i]=temp;
                bp[i]=-bp[i];
                vec2[i]=-vec2[i];
            }
        }
/*
 *      Find f3 and f4, but only if at least one of the segment
 *      endpoints is local to the domain.
 */
        temp=0.0e0;

        #pragma unroll
        for (i=0;i<3;i++) {
            temp+=vec2[i]*t[i];
        }

        #pragma unroll
        for (i=0;i<3;i++) {
            x2mod[i]=x1[i]+temp*t[i];
        }

        #pragma unroll
        for (i=0;i<3;i++) {
            vec2[i]=x2[i]-x2mod[i];
        }

        temp=0.0e0;

        #pragma unroll
        for (i=0;i<3;i++) {
            temp+=vec2[i]*vec2[i];
        }

        magdiff=sqrt(temp);
        temp=magdiff*0.5e0 * cotanthetac;

        #pragma unroll
        for (i=0;i<3;i++) {
            vec1[i]=temp*t[i];
        }

        #pragma unroll
        for (i=0;i<3;i++) {
            x1mod[i]=x1[i]+0.5e0*vec2[i]+vec1[i];
            x2mod[i]+=0.5e0*vec2[i]-vec1[i];
        }

        #pragma unroll
        for (i=0;i<3;i++) {
            R[i]=0.5e0*((x3[i]+x4[i])-(x1mod[i]+x2mod[i]));
        }

        Rdt=0.0e0;

        #pragma unroll
        for (i=0;i<3;i++) {
            Rdt+=R[i]*t[i];
        }

        #pragma unroll
        for (i=0;i<3;i++) {
            nd[i]=R[i]-Rdt*t[i];
        }

        d2=0.0e0;

        #pragma unroll
        for (i=0;i<3;i++) {
            d2+=nd[i]*nd[i];
        }

        #pragma unroll
        for (j=0;j<2;j++) {
            y[j]=0.0e0;
            z[j]=0.0e0;
        }

        #pragma unroll
        for (i=0;i<3;i++) {
            y[0]+=x3[i]*t[i];
            y[1]+=x4[i]*t[i];
            z[0]+=-x1mod[i]*t[i];
            z[1]+=-x2mod[i]*t[i];
        }

        #pragma unroll
        for (j=0;j<2;j++) {
            yv[2*j]=y[j];
            yv[2*j+1]=y[j];
            zv[j]=z[j];
            zv[j+2]=z[j];
        }

        a2_d2 = a2 + d2;

        #pragma unroll
        for (j=0;j<4;j++) {
            ypz[j] = yv[j] + zv[j];
            ymz[j] = yv[j] - zv[j];
        }

        #pragma unroll
        for (j=0;j<4;j++) {
            tmp[j]=a2_d2 + ypz[j]*ypz[j];
        }

        #pragma unroll
        for (j=0;j<4;j++) {
            Ra[j]=sqrt(tmp[j]);
        }

        #pragma unroll
        for (j=0;j<4;j++) {
            Rainv[j]=1.0e0/Ra[j];
        }

        a2d2inv = 1.0e0 / a2_d2;

        #pragma unroll
        for (j=0;j<4;j++) {
            tmp[j]=Ra[j] + ypz[j];
			tmp[j+4]=Ra[j]-ypz[j];
        }

        #pragma unroll
        for (j=0;j<4;j++) {
            Log_Ra_ypz[j]=0.5e0*(log(tmp[j])-log(tmp[j+4]));
        }

        #pragma unroll
        for (j=0;j<4;j++) {
            common1[j] = ymz[j] * Ra[j] * a2d2inv;
            f_115v[j] = -a2d2inv * ypz[j] * Rainv[j];
        }

        temp=2.0e0*a2d2inv;

        #pragma unroll
        for (j=0;j<4;j++) {
            f_003v[j] = Ra[j];
            f_103v[j] = Log_Ra_ypz[j] - common1[j];
            f_113v[j] = -Log_Ra_ypz[j];
            f_213v[j] = zv[j]*Log_Ra_ypz[j] - Ra[j];
            f_005v[j] = temp*Ra[j] - Rainv[j];
            f_105v[j] = common1[j] - yv[j]*Rainv[j];
            f_215v[j] =  Rainv[j] - zv[j] * f_115v[j];
        }

        f_003 = 0.0e0;
        f_103 = 0.0e0;
        f_113 = 0.0e0;
        f_213 = 0.0e0;
        f_005 = 0.0e0;
        f_105 = 0.0e0;
        f_115 = 0.0e0;
        f_215 = 0.0e0;

        #pragma unroll
        for (j=1;j<3;j++) {
            f_003v[j] = -f_003v[j];
            f_103v[j] = -f_103v[j];
            f_113v[j] = -f_113v[j];
            f_213v[j] = -f_213v[j];
            f_005v[j] = -f_005v[j];
            f_105v[j] = -f_105v[j];
            f_115v[j] = -f_115v[j];
            f_215v[j] = -f_215v[j];
        }

        #pragma unroll
        for (j=0;j<4;j++) {
            f_003 += f_003v[j];
            f_103 += f_103v[j];
            f_113 += f_113v[j];
            f_213 += f_213v[j];
            f_005 += f_005v[j];
            f_105 += f_105v[j];
            f_115 += f_115v[j];
            f_215 += f_215v[j];
        }

        f_103 *= -0.5e0;
        f_003 *=  a2d2inv;
        f_005 *=  a2d2inv;
        f_105 *=  a2d2inv;

        #pragma unroll
        for (i=0;i<3;i++) {
            bct[i]=b[alt1[i]]*t[alt2[i]] - b[alt2[i]]*t[alt1[i]];
            bpct[i]=bp[alt1[i]]*t[alt2[i]] - bp[alt2[i]]*t[alt1[i]];
            ndct[i]=nd[alt1[i]]*t[alt2[i]] - nd[alt2[i]]*t[alt1[i]];
        }

        tdb=0.0e0;
        tdbp=0.0e0;
        nddb=0.0e0;
        bpctdb=0.0e0;
        bpctdnd=0.0e0;

        #pragma unroll
        for (i=0;i<3;i++) {
            tdb += t[i]*b[i];
            tdbp+= t[i]*bp[i];
            nddb+= nd[i]*b[i];
            bpctdb += bpct[i]*b[i];
            bpctdnd += bpct[i]*nd[i];

        }

        temp = tdb*tdbp;

        #pragma unroll
        for (i=0;i<3;i++) {
            bpctct[i] = tdbp*t[i] - bp[i];
            common2[i] = temp*nd[i];
            common3[i] = bpctdnd*bct[i];
        }

        tmp[0]=(m4pn-m4p)*tdb;
        tmp[1]=m4pn*bpctdnd*nddb;
        tmp[2]=a2m8p*tdb;
        tmp[3]=m4pn*bpctdnd*tdb;

        #pragma unroll
        for (i=0;i<3;i++) {
            I_003[i] = m4pn*(nddb*bpctct[i] + bpctdb*ndct[i] - common3[i]) -
                       m4p*common2[i];
            I_113[i] =  tmp[0]*bpctct[i];
            I_005[i] = -a2m8p*common2[i] - a2m4pn*common3[i] - tmp[1]*ndct[i];
            I_115[i] = -tmp[2]*bpctct[i] - tmp[3]*ndct[i];
        }

        Fint_003 = f_103 - y[0]*f_003;
        Fint_113 = f_213 - y[0]*f_113;
        Fint_005 = f_105 - y[0]*f_005;
        Fint_115 = f_215 - y[0]*f_115;

        #pragma unroll
        for (i=0;i<3;i++) {
            f4[i] = (I_003[i]*Fint_003 + I_113[i]*Fint_113 + I_005[i]*Fint_005 +
                     I_115[i]*Fint_115) * oneoverL;
        }

        Fint_003 = y[1]*f_003 - f_103;
        Fint_113 = y[1]*f_113 - f_213;
        Fint_005 = y[1]*f_005 - f_105;
        Fint_115 = y[1]*f_115 - f_215;

        #pragma unroll
        for (i=0;i<3;i++) {
            f3[i] = (I_003[i]*Fint_003 + I_113[i]*Fint_113 + I_005[i]*Fint_005 +
                     I_115[i]*Fint_115) * oneoverL;
        }

        fn3.x = f3[0];
        fn3.y = f3[1];
        fn3.z = f3[2];
        fn4.x = f4[0];
        fn4.y = f4[1];
        fn4.z = f4[2];

        x1modMag2 = 0.0e0;
        x2modMag2 = 0.0e0;

        #pragma unroll
        for (i=0;i<3;i++) {
            x1modMag2 += x1mod[i]*x1mod[i];
            x2modMag2 += x2mod[i]*x2mod[i];
        }

        diffMag2 = magdiff*magdiff;

        if (diffMag2 > (eps * (x1modMag2+x2modMag2))) {

			Vec3 fn3cor, fn4cor, fw, fq, bx1, bx2;
			Vec3 rx1, rx2, rx3, rx4, rx1mod, rx2mod;

			rx1.x = x1[0]; rx1.y = x1[1]; rx1.z = x1[2];
			rx2.x = x2[0]; rx2.y = x2[1]; rx2.z = x2[2];
			rx3.x = x3[0]; rx3.y = x3[1]; rx3.z = x3[2];
			rx4.x = x4[0]; rx4.y = x4[1]; rx4.z = x4[2];

			rx1mod.x = x1mod[0]; rx1mod.y = x1mod[1]; rx1mod.z = x1mod[2];
			rx2mod.x = x2mod[0]; rx2mod.y = x2mod[1]; rx2mod.z = x2mod[2];

			bx1.x = bp[0]; bx1.y = bp[1]; bx1.z = bp[2];
			bx2.x = b[0]; bx2.y = b[1]; bx2.z = b[2];

            SegSegForceIsotropicCorr(rx1, rx1mod, rx3, rx4, bx1, bx2, a, MU, NU, fw, fq, fn3cor, fn4cor);

            fn3.x += fn3cor.x;
            fn3.y += fn3cor.y;
            fn3.z += fn3cor.z;
            fn4.x += fn4cor.x;
            fn4.y += fn4cor.y;
            fn4.z += fn4cor.z;

            SegSegForceIsotropicCorr(rx2mod, rx2, rx3, rx4, bx1, bx2, a, MU, NU, fw, fq, fn3cor, fn4cor);

            fn3.x += fn3cor.x;
            fn3.y += fn3cor.y;
            fn3.z += fn3cor.z;
            fn4.x += fn4cor.x;
            fn4.y += fn4cor.y;
            fn4.z += fn4cor.z;
        }

        return;
}

/*------------------------------------------------------------------------
 *
 *    Function:     SegSegForceIsotropic
 *
 *-----------------------------------------------------------------------*/
KOKKOS_FORCEINLINE_FUNCTION
void SegSegForceIsotropic(const Vec3 &r1, const Vec3 &r2, const Vec3 &r3, const Vec3 &r4, 
                          const Vec3 &b1, const Vec3 &b2, double a, double MU, double NU,
                          Vec3 &fn1, Vec3 &fn2, Vec3 &fn3, Vec3 &fn4,
                          int seg12Local=1, int seg34Local=1)
{
        double x1[3], x2[3], x3[3], x4[3], b[3], bp[3];
        double f1[3], f2[3], f3[3], f4[3];
        double vec1[3], vec2[3], t[3], tp[3], tctp[3];
        double R[2][3], tempa[2], tempb[2], y[2], z[2];
        int i, j , alt1[3]={1,2,0}, alt2[3]={2,0,1};
        double eps, d, c, c2, onemc2, onemc2inv, oneoverL, oneoverLp;
        double a2, m4p, m4pd, m8p, m8pd, m4pn, m4pnd, m4pnd2, m4pnd3;
        double a2m4pnd, a2m8pd, a2m4pn, a2m8p, a2_d2, a2_d2inv, denom;
        double temp1, temp2, temp3, temp4[8], tmp[10];
        double yv[4], zv[4], y2[4], z2[4], Ra[4], Rainv[4];
        double Ra_Rdot_tp[8], Ra_Rdot_t[8], log_Ra_Rdot_tp[4], log_Ra_Rdot_t[4];
        double Ra2_R_tpinv[4], Ra2_R_tinv[4], ylog_Ra_Rdot_tp[4], zlog_Ra_Rdot_t[4];
        double yRa2_R_tpinv[4], zRa2_R_tinv[4], y2Ra2_R_tpinv[4], z2Ra2_R_tinv[4];
        double adf_003[4], commonf223[4], commonf225[4], commonf025[4], commonf205[4];
        double commonf305[4], commonf035[4], ycommonf025[4], zcommonf205[4], zcommonf305[4];
        double tf_113[4];
        double f_003v[4], f_103v[4], f_013v[4], f_113v[4];
        double f_203v[4], f_023v[4], f_005v[4], f_105v[4];
        double f_003,  f_103,  f_013,  f_113,  f_203,  f_023,  f_005,  f_105;
        double f_015v[4], f_115v[4], f_205v[4], f_025v[4];
        double f_215v[4], f_125v[4], f_225v[4], f_305v[4];
        double f_015,  f_115,  f_205,  f_025,  f_215,  f_125,  f_225,  f_305;
        double f_035v[4], f_315v[4], f_135v[4];
        double f_035,  f_315,  f_135;
        double Fint_003, Fint_005, Fint_013, Fint_015, Fint_025, Fint_103;
        double Fint_105, Fint_115, Fint_125, Fint_205, Fint_215;
        double I_003[3], I_005[3], I_013[3], I_015[3], I_025[3], I_103[3];
        double I_105[3], I_115[3], I_125[3], I_205[3], I_215[3];
        double I00a[3], I01a[3], I10a[3], I00b[3], I01b[3], I10b[3];
        double bctctp[3], bct[3], bpctpct[3], bpctp[3], tcbpct[3];
        double bctdbp, bpctpdb, tcbpdb, tcbpdtp, tpcbdbp;
        double tctpct[3], tpct[3];
        double tctpcbpdb, tctpcbpdtp, tctpdb, tdb, tdbp;
        double tpcbctp[3], tpctctp[3];
        double tpcbdt, tpctcbdbp, tpctcbdt, tpctdbp, tpdb, tpdbp;
        double pivalue=3.141592653589793;

        eps = 1e-4;

        fn1.x = 0.0;
        fn1.y = 0.0;
        fn1.z = 0.0;

        fn2.x = 0.0;
        fn2.y = 0.0;
        fn2.z = 0.0;

        fn3.x = 0.0;
        fn3.y = 0.0;
        fn3.z = 0.0;

        fn4.x = 0.0;
        fn4.y = 0.0;
        fn4.z = 0.0;

        x1[0]=r1.x;
        x1[1]=r1.y;
        x1[2]=r1.z;
        x2[0]=r2.x;
        x2[1]=r2.y;
        x2[2]=r2.z;
        x3[0]=r3.x;
        x3[1]=r3.y;
        x3[2]=r3.z;
        x4[0]=r4.x;
        x4[1]=r4.y;
        x4[2]=r4.z;

        b[0]=b2.x;
        b[1]=b2.y;
        b[2]=b2.z;
        bp[0]=b1.x;
        bp[1]=b1.y;
        bp[2]=b1.z;

        #pragma unroll
        for(i=0;i<3;i++) {
            vec1[i]=x4[i]-x3[i];
            vec2[i]=x2[i]-x1[i];
        }

        temp1=0.0e0;
        temp2=0.0e0;

        #pragma unroll
        for(i=0;i<3;i++) {
            temp1+=vec1[i]*vec1[i];
            temp2+=vec2[i]*vec2[i];
        }

        oneoverL =1/sqrt(temp1);
        oneoverLp=1/sqrt(temp2);

        #pragma unroll
        for(i=0;i<3;i++) {
            t[i]=vec1[i]*oneoverL;
            tp[i]=vec2[i]*oneoverLp;
        }

        c=0.0e0;

        #pragma unroll
        for(i=0;i<3;i++) {
            c+=t[i]*tp[i];
        }

        c2=c*c;
        onemc2=1-c2;

        if (onemc2 > eps) {

			#pragma unroll
            for(i=0;i<3;i++) {
                tctp[i]=t[alt1[i]]*tp[alt2[i]]-t[alt2[i]]*tp[alt1[i]];
            }

            onemc2inv = 1/onemc2;

            #pragma unroll
            for(i=0;i<3;i++) {
                R[0][i]=x3[i]-x1[i];
                R[1][i]=x4[i]-x2[i];
            }

            d=0.0e0;

            #pragma unroll
            for (j=0;j<2;j++) {
                tempa[j]=0.0e0;
                tempb[j]=0.0e0;
            }

            #pragma unroll
            for(i=0;i<3;i++) {
                d+=0.5e0*((x4[i]+x3[i])-(x2[i]+x1[i]))*tctp[i];
                for (j=0;j<2;j++) {
                    tempa[j]+=R[j][i]*t[i];
                    tempb[j]+=R[j][i]*tp[i];
                }
            }

            d*=onemc2inv;

            #pragma unroll
            for (j=0;j<2;j++) {
                y[j]=(tempa[j]-c*tempb[j])*onemc2inv;
                z[j]=(tempb[j]-c*tempa[j])*onemc2inv;
            }

/*          now we calculate the definite integrals of the force calculation  */

            #pragma unroll
            for (j=0;j<2;j++) {
                yv[2*j]=y[j];
                yv[2*j+1]=y[j];
                zv[j]=z[j];
                zv[j+2]=z[j];
            }

            a2_d2 = a*a+d*d*onemc2;

            #pragma unroll
            for (j=0;j<4;j++) {
                y2[j] = yv[j]*yv[j];
                z2[j] = zv[j]*zv[j];

            }

            #pragma unroll
            for (j=0;j<4;j++) {
                temp4[j]=a2_d2 + y2[j] + z2[j] + 2.0e0*yv[j]*zv[j]*c;
            }

            temp1=onemc2*a2_d2;

            #pragma unroll
            for (j=0;j<4;j++) {
                Ra[j]=sqrt(temp4[j]);
            }

            temp2=sqrt(temp1);

            #pragma unroll
            for (j=0;j<4;j++) {
                Rainv[j]=1.0e0/Ra[j];
            }

            denom=1.0e0/temp2;
            a2_d2inv=1.0e0/a2_d2;

            #pragma unroll
            for (j=0;j<4;j++) {
                Ra_Rdot_tp[j] = Ra[j]+(zv[j]+yv[j]*c);
                Ra_Rdot_t[j]  = Ra[j]+(yv[j]+zv[j]*c);
				Ra_Rdot_tp[j+4] = Ra[j]-(zv[j]+yv[j]*c);
                Ra_Rdot_t[j+4]  = Ra[j]-(yv[j]+zv[j]*c);
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                log_Ra_Rdot_tp[j] =0.5e0*(log(Ra_Rdot_tp[j])-log(Ra_Rdot_tp[j+4]));
                log_Ra_Rdot_t[j]  =0.5e0*(log(Ra_Rdot_t[j])-log(Ra_Rdot_t[j+4]));
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                Ra2_R_tpinv[j] = 0.5e0*(Rainv[j]/Ra_Rdot_tp[j]- Rainv[j]/Ra_Rdot_tp[j+4]);
                Ra2_R_tinv[j] =  0.5e0*(Rainv[j]/Ra_Rdot_t[j]- Rainv[j]/Ra_Rdot_t[j+4]);
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                ylog_Ra_Rdot_tp[j] = yv[j]*log_Ra_Rdot_tp[j];
                yRa2_R_tpinv[j]    = yv[j]*   Ra2_R_tpinv[j];
                zlog_Ra_Rdot_t[j]  = zv[j]*log_Ra_Rdot_t[j];
                zRa2_R_tinv[j]     = zv[j]*   Ra2_R_tinv[j];

            }

            #pragma unroll
            for (j=0;j<4;j++) {
                y2Ra2_R_tpinv[j] = yv[j]* yRa2_R_tpinv[j];
                z2Ra2_R_tinv[j]  = zv[j]*  zRa2_R_tinv[j];
            }

            temp1=denom*(1+c);

            #pragma unroll
            for (j=0;j<4;j++) {
                temp4[j]=temp1*(Ra[j]+(yv[j]+zv[j]));
				temp4[j+4]=temp1*(Ra[j]-(yv[j]+zv[j]));
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                f_003v[j]=0.5e0*(atan(temp4[j])+atan(temp4[j+4]));
            }

            temp1=-2.0e0*denom;

            #pragma unroll
            for (j=0;j<4;j++) {
                f_003v[j]*=temp1;
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                adf_003[j]=f_003v[j]*a2_d2;
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                commonf223[j] = c*Ra[j] - adf_003[j];
                f_103v[j] = c*log_Ra_Rdot_t[j]  - log_Ra_Rdot_tp[j];
                f_013v[j] = c*log_Ra_Rdot_tp[j] - log_Ra_Rdot_t [j];
                f_113v[j] = c*adf_003[j] - Ra[j];
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                commonf223[j] *= onemc2inv;
                f_103v[j] *=      onemc2inv;
                f_013v[j] *=      onemc2inv;
                f_113v[j] *=      onemc2inv;
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                commonf225[j] = f_003v[j] - c*Rainv[j];
                commonf025[j] = c*yRa2_R_tpinv[j] - Rainv[j];
                commonf205[j] = c*zRa2_R_tinv[j]  - Rainv[j];
                commonf305[j] = log_Ra_Rdot_t[j]  -(yv[j]-c*zv[j])*Rainv[j] - c2*z2Ra2_R_tinv[j];
                commonf035[j] = log_Ra_Rdot_tp[j] -(zv[j]-c*yv[j])*Rainv[j] - c2*y2Ra2_R_tpinv[j];
                f_203v[j] =  zlog_Ra_Rdot_t[j]  + commonf223[j];
                f_023v[j] =  ylog_Ra_Rdot_tp[j] + commonf223[j];
                f_005v[j] = f_003v[j] - yRa2_R_tpinv[j] - zRa2_R_tinv[j];
                f_105v[j] = Ra2_R_tpinv[j] - c*Ra2_R_tinv[j];
                f_015v[j] = Ra2_R_tinv[j]  - c*Ra2_R_tpinv[j];
                f_115v[j] = Rainv[j] - c*(yRa2_R_tpinv[j] + zRa2_R_tinv[j] + f_003v[j]);
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                ycommonf025[j] = yv[j]*commonf025[j];
                zcommonf205[j] = zv[j]*commonf205[j];
                zcommonf305[j] = zv[j]*commonf305[j];
                tf_113[j]=2.0e0*f_113v[j];
                f_205v[j] = yRa2_R_tpinv[j] + c2*zRa2_R_tinv[j]  + commonf225[j];
                f_025v[j] = zRa2_R_tinv[j]  + c2*yRa2_R_tpinv[j] + commonf225[j];
                f_305v[j] = y2Ra2_R_tpinv[j] + c*commonf305[j] + 2.0e0*f_103v[j];
                f_035v[j] = z2Ra2_R_tinv[j]  + c*commonf035[j] + 2.0e0*f_013v[j];
            }

            #pragma unroll
            for (j=0;j<4;j++) {
                f_215v[j] = f_013v[j] - ycommonf025[j] + c*(zcommonf205[j]-f_103v[j]);
                f_125v[j] = f_103v[j] - zcommonf205[j] + c*(ycommonf025[j] - f_013v[j]);
                f_225v[j] = f_203v[j] - zcommonf305[j] + c*(y2[j]*commonf025[j] - tf_113[j]);
                f_315v[j] = tf_113[j] - y2[j]*commonf025[j] + c*(zcommonf305[j] - f_203v[j]);
                f_135v[j] = tf_113[j] - z2[j]*commonf205[j] + c*(yv[j]*commonf035[j]-f_023v[j]);
            }


            f_003= (f_003v[0]+f_003v[3])-(f_003v[1]+f_003v[2]);
            f_013= (f_013v[0]+f_013v[3])-(f_013v[1]+f_013v[2]);
            f_103= (f_103v[0]+f_103v[3])-(f_103v[1]+f_103v[2]);
            f_113= (f_113v[0]+f_113v[3])-(f_113v[1]+f_113v[2]);
            f_023= (f_023v[0]+f_023v[3])-(f_023v[1]+f_023v[2]);
            f_203= (f_203v[0]+f_203v[3])-(f_203v[1]+f_203v[2]);
            f_005= (f_005v[0]+f_005v[3])-(f_005v[1]+f_005v[2]);
            f_015= (f_015v[0]+f_015v[3])-(f_015v[1]+f_015v[2]);
            f_105= (f_105v[0]+f_105v[3])-(f_105v[1]+f_105v[2]);
            f_115= (f_115v[0]+f_115v[3])-(f_115v[1]+f_115v[2]);
            f_025= (f_025v[0]+f_025v[3])-(f_025v[1]+f_025v[2]);
            f_205= (f_205v[0]+f_205v[3])-(f_205v[1]+f_205v[2]);
            f_215= (f_215v[0]+f_215v[3])-(f_215v[1]+f_215v[2]);
            f_125= (f_125v[0]+f_125v[3])-(f_125v[1]+f_125v[2]);
            f_035= (f_035v[0]+f_035v[3])-(f_035v[1]+f_035v[2]);
            f_305= (f_305v[0]+f_305v[3])-(f_305v[1]+f_305v[2]);
            f_225= (f_225v[0]+f_225v[3])-(f_225v[1]+f_225v[2]);
            f_135= (f_135v[0]+f_135v[3])-(f_135v[1]+f_135v[2]);
            f_315= (f_315v[0]+f_315v[3])-(f_315v[1]+f_315v[2]);


            f_005 *= a2_d2inv;
            f_105 *= onemc2inv;
            f_015 *= onemc2inv;
            f_115 *= onemc2inv;
            f_205 *= onemc2inv;
            f_025 *= onemc2inv;
            f_305 *= onemc2inv;
            f_035 *= onemc2inv;
            f_215 *= onemc2inv;
            f_125 *= onemc2inv;
            f_225 *= onemc2inv;
            f_315 *= onemc2inv;
            f_135 *= onemc2inv;


/* now construct the vector coefficients for the definite integrals */

            a2 = a*a;
            m4p = 0.25 * MU / pivalue;
            m4pd =  m4p * d;
            m8p = 0.5 * m4p;
            m8pd = m8p * d;
            m4pn = m4p / ( 1 - NU );
            m4pnd = m4pn * d;
            m4pnd2 = m4pnd * d;
            m4pnd3 = m4pnd2 * d;
            a2m4pnd = a2 * m4pnd;
            a2m8pd = a2 * m8pd;
            a2m4pn = a2 * m4pn;
            a2m8p = a2 * m8p;

            #pragma unroll
            for (i=0;i<3;i++) {
                tpct[i]=-tctp[i];
                bct[i]=b[alt1[i]]*t[alt2[i]]-b[alt2[i]]*t[alt1[i]];
                bpctp[i]=bp[alt1[i]]*tp[alt2[i]]-bp[alt2[i]]*tp[alt1[i]];

            }

            tdb=0.0e0;
            tdbp=0.0e0;
            tpdb=0.0e0;
            tpdbp=0.0e0;
            tctpdb=0.0e0;
            tpctdbp=0.0e0;
            bpctpdb=0.0e0;
            bctdbp=0.0e0;

            #pragma unroll
            for (i=0;i<3;i++) {
                tdb    +=t[i]*b[i];
                tdbp   +=t[i]*bp[i];
                tpdb   +=tp[i]*b[i];
                tpdbp  +=tp[i]*bp[i];
                tctpdb +=tctp[i]*b[i];
                tpctdbp+=tpct[i]*bp[i];
                bpctpdb+=bpctp[i]*b[i];
                bctdbp +=bct[i]*bp[i];
            }

            #pragma unroll
            for (i=0;i<3;i++) {
                tctpct[i]    =        tp[i] -     c*t[i];
                tpctctp[i]   =         t[i] -    c*tp[i];
                tcbpct[i]    =        bp[i] -  tdbp*t[i];
                tpcbctp[i]   =         b[i] - tpdb*tp[i];
                bpctpct[i]   =   tdbp*tp[i] -    c*bp[i];
                bctctp[i]    =    tpdb*t[i] -     c*b[i];
            }


            tctpcbpdtp = tdbp - tpdbp*c;
            tpctcbdt = tpdb - tdb*c;
            tctpcbpdb =  tdbp*tpdb - tpdbp*tdb;
            tpctcbdbp = tctpcbpdb;
            tcbpdtp = tpctdbp;
            tpcbdt = tctpdb;
            tcbpdb = bctdbp;
            tpcbdbp = bpctpdb;

/*
 *          Only calculate the forces for segment p3->p4 if at least one
 *          of the segment's nodes is local to the current domain.
 */
            if (seg34Local) {

                temp1 = tdbp*tpdb + tctpcbpdb;

                #pragma unroll
                for (i=0;i<3;i++) {
                    I00a[i] = temp1 * tpct[i];
                    I00b[i] = tctpcbpdtp * bct[i];
                }

                temp1 = (m4pnd * tctpdb);
                temp2 = (m4pnd * bpctpdb);
                temp3 = (m4pnd3 * tctpcbpdtp*tctpdb);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_003[i] = m4pd*I00a[i] - m4pnd*I00b[i] + temp1*bpctpct[i] +
                            temp2*tctpct[i];
                    I_005[i] = a2m8pd*I00a[i] - a2m4pnd*I00b[i] - temp3*tctpct[i];
                    I10a[i] = tcbpct[i]*tpdb - tctp[i]*tcbpdb;
                    I10b[i] = bct[i] * tcbpdtp;

                }

                temp1 = (m4pn * tdb);
                temp2 = m4pnd2 * (tcbpdtp*tctpdb + tctpcbpdtp*tdb);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_103[i] = temp1*bpctpct[i] + m4p*I10a[i] - m4pn*I10b[i];
                    I_105[i] = a2m8p*I10a[i] - a2m4pn*I10b[i] - temp2*tctpct[i];
                    I01a[i] = tctp[i]*bpctpdb - bpctpct[i]*tpdb;
                }

                tmp[0] = (m4pn * tpdb);
                tmp[1] = (m4pn * bpctpdb);
                tmp[2] = (m4pnd2 * tctpcbpdtp * tpdb);
                tmp[3] = (m4pnd2 * tctpcbpdtp * tctpdb);
                tmp[4] = (m4pnd * tcbpdtp * tdb);
                tmp[5] = (m4pnd * tctpcbpdtp * tpdb) ;
                tmp[6] = (m4pnd * (tctpcbpdtp*tdb + tcbpdtp*tctpdb));
                tmp[7] = (m4pnd * tcbpdtp * tpdb);
                tmp[8] = (m4pn * tcbpdtp * tdb);
                tmp[9] = (m4pn * tcbpdtp * tpdb);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_013[i] = m4p*I01a[i] + tmp[0]*bpctpct[i] - tmp[1]*tctp[i];
                    I_015[i] = a2m8p*I01a[i] - tmp[2]*tctpct[i] + tmp[3]*tctp[i];
                    I_205[i] = -tmp[4] * tctpct[i];
                    I_025[i] = tmp[5] * tctp[i];
                    I_115[i] =  tmp[6]*tctp[i] - tmp[7]*tctpct[i];
                    I_215[i] = tmp[8] * tctp[i];
                    I_125[i] = tmp[9] * tctp[i];
                }

                Fint_003 = f_103 - y[0]*f_003;
                Fint_103 = f_203 - y[0]*f_103;
                Fint_013 = f_113 - y[0]*f_013;
                Fint_005 = f_105 - y[0]*f_005;
                Fint_105 = f_205 - y[0]*f_105;
                Fint_015 = f_115 - y[0]*f_015;
                Fint_115 = f_215 - y[0]*f_115;
                Fint_205 = f_305 - y[0]*f_205;
                Fint_025 = f_125 - y[0]*f_025;
                Fint_215 = f_315 - y[0]*f_215;
                Fint_125 = f_225 - y[0]*f_125;

                #pragma unroll
                for (i=0;i<3;i++) {
                    f4[i]=(I_003[i]*Fint_003 + I_103[i]*Fint_103 + I_013[i]*Fint_013 +
                           I_005[i]*Fint_005 + I_105[i]*Fint_105 + I_015[i]*Fint_015 +
                           I_115[i]*Fint_115 + I_205[i]*Fint_205 + I_025[i]*Fint_025 +
                           I_215[i]*Fint_215 + I_125[i]*Fint_125) * oneoverL;
                }

                Fint_003 = y[1]*f_003 - f_103;
                Fint_103 = y[1]*f_103 - f_203;
                Fint_013 = y[1]*f_013 - f_113;
                Fint_005 = y[1]*f_005 - f_105;
                Fint_105 = y[1]*f_105 - f_205;
                Fint_015 = y[1]*f_015 - f_115;
                Fint_115 = y[1]*f_115 - f_215;
                Fint_205 = y[1]*f_205 - f_305;
                Fint_025 = y[1]*f_025 - f_125;
                Fint_215 = y[1]*f_215 - f_315;
                Fint_125 = y[1]*f_125 - f_225;

                #pragma unroll
                for (i=0;i<3;i++) {
                    f3[i]=(I_003[i]*Fint_003 + I_103[i]*Fint_103 + I_013[i]*Fint_013 +
                           I_005[i]*Fint_005 + I_105[i]*Fint_105 + I_015[i]*Fint_015 +
                           I_115[i]*Fint_115 + I_205[i]*Fint_205 + I_025[i]*Fint_025 +
                           I_215[i]*Fint_215 + I_125[i]*Fint_125) * oneoverL;
                }

                fn3.x=f3[0];
                fn3.y=f3[1];
                fn3.z=f3[2];
                fn4.x=f4[0];
                fn4.y=f4[1];
                fn4.z=f4[2];

            } /* if segment p3->p4 is "local" */

/*
 *          Only calculate the forces for segment p1->p2 if at least one
 *          of the segment's nodes is local to the current domain.
 */
            if (seg12Local) {

                temp1 = tpdb*tdbp + tpctcbdbp;

                #pragma unroll
                for (i=0;i<3;i++) {
                    I00a[i] = temp1 * tctp[i];
                    I00b[i] = bpctp[i] * tpctcbdt;
                }

                temp1 = m4pnd * tpctdbp;
                temp2 = m4pnd * bctdbp;
                temp3 = m4pnd3 * tpctcbdt * tpctdbp;

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_003[i] = m4pd*I00a[i] - m4pnd*I00b[i] + temp1*bctctp[i] +
                               temp2*tpctctp[i];
                    I_005[i] = a2m8pd*I00a[i] - a2m4pnd*I00b[i] - temp3*tpctctp[i];
                    I01a[i] = tpct[i]*tpcbdbp - tpcbctp[i]*tdbp;
                    I01b[i] = -bpctp[i] * tpcbdt;
                }

                temp1 = m4pn * tpdbp;
                temp2 = m4pnd2 * (tpcbdt*tpctdbp + tpctcbdt*tpdbp);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_013[i] = -temp1 * bctctp[i] + m4p*I01a[i] - m4pn*I01b[i];
                    I_015[i] = a2m8p*I01a[i] - a2m4pn*I01b[i] + temp2*tpctctp[i];
                    I10a[i] = bctctp[i]*tdbp - tpct[i]*bctdbp;
                }

                tmp[0] = m4pn * tdbp;
                tmp[1] = m4pn * bctdbp;
                tmp[2] = m4pnd2 * tpctcbdt * tdbp;
                tmp[3] = m4pnd2 * tpctcbdt * tpctdbp;
                tmp[4] = (m4pnd * tpcbdt * tpdbp);
                tmp[5] = (m4pnd * tpctcbdt * tdbp);
                tmp[6] = m4pnd * (tpctcbdt*tpdbp + tpcbdt*tpctdbp);
                tmp[7] = m4pnd * tpcbdt * tdbp;
                tmp[8] = (m4pn * tpcbdt * tpdbp);
                tmp[9] = (m4pn * tpcbdt * tdbp);

                #pragma unroll
                for (i=0;i<3;i++) {
                    I_103[i] = m4p*I10a[i] - tmp[0]*bctctp[i] + tmp[1]*tpct[i];
                    I_105[i] = a2m8p*I10a[i] + tmp[2]*tpctctp[i] - tmp[3]*tpct[i];
                    I_025[i] = -tmp[4] * tpctctp[i];
                    I_205[i] = tmp[5] * tpct[i];
                    I_115[i] = tmp[6]*tpct[i] - tmp[7]*tpctctp[i];
                    I_125[i] = -tmp[8] * tpct[i];
                    I_215[i] = -tmp[9] * tpct[i];
                }

                Fint_003 = f_013 - z[1]*f_003;
                Fint_103 = f_113 - z[1]*f_103;
                Fint_013 = f_023 - z[1]*f_013;
                Fint_005 = f_015 - z[1]*f_005;
                Fint_105 = f_115 - z[1]*f_105;
                Fint_015 = f_025 - z[1]*f_015;
                Fint_115 = f_125 - z[1]*f_115;
                Fint_205 = f_215 - z[1]*f_205;
                Fint_025 = f_035 - z[1]*f_025;
                Fint_215 = f_225 - z[1]*f_215;
                Fint_125 = f_135 - z[1]*f_125;

                #pragma unroll
                for (i=0;i<3;i++) {
                    f1[i]=(I_003[i]*Fint_003 + I_103[i]*Fint_103 + I_013[i]*Fint_013 +
                           I_005[i]*Fint_005 + I_105[i]*Fint_105 + I_015[i]*Fint_015 +
                           I_115[i]*Fint_115 + I_205[i]*Fint_205 + I_025[i]*Fint_025 +
                           I_215[i]*Fint_215 + I_125[i]*Fint_125) * oneoverLp;
                }

                Fint_003 = z[0]*f_003 - f_013;
                Fint_103 = z[0]*f_103 - f_113;
                Fint_013 = z[0]*f_013 - f_023;
                Fint_005 = z[0]*f_005 - f_015;
                Fint_105 = z[0]*f_105 - f_115;
                Fint_015 = z[0]*f_015 - f_025;
                Fint_115 = z[0]*f_115 - f_125;
                Fint_205 = z[0]*f_205 - f_215;
                Fint_025 = z[0]*f_025 - f_035;
                Fint_215 = z[0]*f_215 - f_225;
                Fint_125 = z[0]*f_125 - f_135;

                #pragma unroll
                for (i=0;i<3;i++) {
                    f2[i]=(I_003[i]*Fint_003 + I_103[i]*Fint_103 + I_013[i]*Fint_013 +
                           I_005[i]*Fint_005 + I_105[i]*Fint_105 + I_015[i]*Fint_015 +
                           I_115[i]*Fint_115 + I_205[i]*Fint_205 + I_025[i]*Fint_025 +
                           I_215[i]*Fint_215 + I_125[i]*Fint_125) * oneoverLp;
                }

                fn1.x=f1[0];
                fn1.y=f1[1];
                fn1.z=f1[2];
                fn2.x=f2[0];
                fn2.y=f2[1];
                fn2.z=f2[2];


            } /* if segment p1->p2 is "local" */

        } else {
/*
 *          The two lines are parallel, so we have to use a special
 *          lower dimensional function
 */

			SpecialSegSegForceHalf(r1, r2, r3, r4, b1, b2, a, MU, NU, fn3, fn4);

            SpecialSegSegForceHalf(r3, r4, r1, r2, b2, b1, a, MU, NU, fn1, fn2);
       }

       return;
}

} // namespace ExaDiS

#endif

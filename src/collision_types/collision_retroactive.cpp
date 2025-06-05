/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "collision_retroactive.h"

namespace ExaDiS {

typedef double real8;

#define V3_DOT(a,b)             ( ((a)[0]*(b)[0]) + ((a)[1]*(b)[1]) + ((a)[2]*(b)[2]) )
#define V3_SUB(a,b,c)           { (a)[0]=((b)[0]-(c)[0]); \
                                  (a)[1]=((b)[1]-(c)[1]); \
                                  (a)[2]=((b)[2]-(c)[2]); }
#define M33_DET(m)                \
(                                 \
     ((m)[0] * (m)[4] * (m)[8])   \
   + ((m)[1] * (m)[5] * (m)[6])   \
   + ((m)[2] * (m)[3] * (m)[7])   \
   - ((m)[2] * (m)[4] * (m)[6])   \
   - ((m)[1] * (m)[3] * (m)[8])   \
   - ((m)[0] * (m)[5] * (m)[7])   \
)
#define M33_INV(a,b)                                 \
{                                                    \
   real8 det = M33_DET(b);                           \
         det = ( (fabs(det)>0.0) ? 1.0/det : 0.0 );  \
                                                     \
   (a)[0] = det * ((b)[4]*(b)[8] - (b)[5]*(b)[7]);   \
   (a)[1] = det * ((b)[2]*(b)[7] - (b)[1]*(b)[8]);   \
   (a)[2] = det * ((b)[1]*(b)[5] - (b)[2]*(b)[4]);   \
   (a)[3] = det * ((b)[5]*(b)[6] - (b)[3]*(b)[8]);   \
   (a)[4] = det * ((b)[0]*(b)[8] - (b)[2]*(b)[6]);   \
   (a)[5] = det * ((b)[2]*(b)[3] - (b)[0]*(b)[5]);   \
   (a)[6] = det * ((b)[3]*(b)[7] - (b)[4]*(b)[6]);   \
   (a)[7] = det * ((b)[1]*(b)[6] - (b)[0]*(b)[7]);   \
   (a)[8] = det * ((b)[0]*(b)[4] - (b)[1]*(b)[3]);   \
}
#define V3_M33_V3_MUL(q,m,p)                      \
{                                                 \
   real8 _x = (p)[0];                             \
   real8 _y = (p)[1];                             \
   real8 _z = (p)[2];                             \
                                                  \
   (q)[0] = ((m)[0]*_x)+((m)[1]*_y)+((m)[2]*_z);  \
   (q)[1] = ((m)[3]*_x)+((m)[4]*_y)+((m)[5]*_z);  \
   (q)[2] = ((m)[6]*_x)+((m)[7]*_y)+((m)[8]*_z);  \
}

/***************************************************************************
 *
 *     Function :     InSphere
 *
 *     Description :  Given a sphere located at X0 with radius=XR and a point,
 *                    will return
 *                    1 = point is inside  the sphere
 *                    0 = point is outside the sphere
 *      Arguments:
 *                    x0 : sphere center
 *                    xr : sphere radius
 *                    xp : point
 ***************************************************************************/
KOKKOS_INLINE_FUNCTION
int InSphere(const real8 *x0, const real8 xr, const real8 *xp)
{
    real8 dv[3] = { xp[0]-x0[0], xp[1]-x0[1], xp[2]-x0[2] };
    real8 dr = V3_DOT(dv,dv);
    return ( (dr<(xr*xr)) ? 1 : 0 );
}

/*---------------------------------------------------------------------------
 *
 *      Function :   GrowSphere
 *
 *     Description : Given a sphere located at x0 with radius xr and a point outside
 *                   the sphere, will adjust the sphere location and radius to include
 *                   the new point.
 *
 *-----------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
void GrowSphere (
         real8 *x0,     ///< center of the sphere <xyz>    (returned)
         real8 *xr,     ///< radius of the sphere <scalar> (returned)
   const real8 *xp )    ///< position outside of the sphere <xyz>
{
    if ( !InSphere(x0,(*xr),xp) )
    {
        real8 dx[3] = { xp[0]-x0[0], xp[1]-x0[1], xp[2]-x0[2] };
        real8 ds = sqrt( dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] );
        (*xr) = 0.5*((*xr)+ds);
        if (ds > 1e-20) {
            x0[0] = xp[0]-dx[0]/ds*(*xr);
            x0[1] = xp[1]-dx[1]/ds*(*xr);
            x0[2] = xp[2]-dx[2]/ds*(*xr);
        }
    }
}

/**************************************************************************
 *
 *       Function:       FindSphere
 *
 *      Description :    Returns a sphere located at x0 with radius=xr that
 *                       contains all 4 points provided.
 *
 **************************************************************************/
KOKKOS_INLINE_FUNCTION
void FindSphere (
         real8 *x0,     ///< center of the sphere <xyz>    (returned)
         real8 *xr,     ///< radius of the sphere <scalar> (returned)
   const real8 *x1,     ///< position 1 <xyz>
   const real8 *x2,     ///< position 2 <xyz>
   const real8 *x3,     ///< position 3 <xyz>
   const real8 *x4 )    ///< position 4 <xyz>
{
    x0[0] = (x1[0]+x2[0])*0.5;
    x0[1] = (x1[1]+x2[1])*0.5;
    x0[2] = (x1[2]+x2[2])*0.5;

    real8 dx[3] = { x1[0]-x2[0], x1[1]-x2[1], x1[2]-x2[2] };
    *xr = 0.5*sqrt( dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] );

    GrowSphere(x0,xr,x3);
    GrowSphere(x0,xr,x4);
}

/**************************************************************************
 *
 *       Function:       MinPointSegDist()
 *
 * Calculates the minimum distance between a point and a line segment
 *
 ***************************************************************************/
KOKKOS_INLINE_FUNCTION
void MinPointSegDist (
         real8 *MinDist2 ,
         real8 *L1       ,
   const real8 *x0       ,
   const real8 *y0       ,
   const real8 *y1       )
{
    // initialize
    real8 diff[3] = { x0[0]-y0[0], x0[1]-y0[1], x0[2]-y0[2] };
    (*MinDist2) = V3_DOT(diff,diff);
    (*L1)       = 0.0;

    // check other end
    real8 diff2[3] = { x0[0]-y1[0], x0[1]-y1[1], x0[2]-y1[2] };
    real8 Dist2    = V3_DOT(diff2,diff2);

    if (Dist2<(*MinDist2))
    {
        (*MinDist2) = Dist2;
        (*L1)       = 1.0;
    }

    // check middle
    diff2[0] = y0[0]-y1[0];
    diff2[1] = y0[1]-y1[1];
    diff2[2] = y0[2]-y1[2];
    real8 B = V3_DOT(diff2,diff2);

    real8 eps = 1.0e-12;
    if (B>eps)
    {
        real8 L1temp =  (-diff[0]*diff2[0]/B)
        +(-diff[1]*diff2[1]/B)
        +(-diff[2]*diff2[2]/B);

        if ( (0.0<L1temp) && (L1temp<1.0) )
        {
            real8 tempv[3] =  { diff[0]+diff2[0]*L1temp,
            diff[1]+diff2[1]*L1temp,
            diff[2]+diff2[2]*L1temp };
            Dist2 = V3_DOT(tempv,tempv);

            if ( Dist2<(*MinDist2) )
            {
                (*MinDist2) = Dist2;
                (*L1)       = L1temp;
            }
        }
    }
}

/***************************************************************************
 *
 *       Function:       MinSegSegDist()
 *
 *       Calculates the minimum distance between two line segments.
 *       segment 1 goes from x0 to x1 and segment 2 goes from y0 to y1
 **************************************************************************/
KOKKOS_INLINE_FUNCTION
void MinSegSegDist (
         real8 *MinDist2 ,
         real8 *L1       ,
         real8 *L2       ,
   const real8 *x0       ,
   const real8 *x1       ,
   const real8 *y0       ,
   const real8 *y1       )
{
    // intialize
    real8 diff[3];
    V3_SUB (diff,x0,y0);
    (*MinDist2) = V3_DOT(diff,diff); (*L1)=0.0; (*L2)=0.0;

    // Check the ends of the solution space
    // first check for (*L1) = 0
    real8 Dist2  = 0.0;
    real8 L1temp = 0.0;
    real8 L2temp = 0.0;

    MinPointSegDist(&Dist2,&L2temp,x0,y0,y1);
    if ( Dist2<(*MinDist2) ) { (*MinDist2) = Dist2; (*L1) = 0.0; (*L2) = L2temp; }

    // second check for (*L1) = 1
    MinPointSegDist(&Dist2,&L2temp,x1,y0,y1);
    if ( Dist2<(*MinDist2) ) { (*MinDist2) = Dist2; (*L1) = 1.0; (*L2) = L2temp; }

    // third check for (*L2) = 0
    MinPointSegDist(&Dist2,&L1temp,y0,x0,x1);
    if ( Dist2<(*MinDist2) ) { (*MinDist2) = Dist2; (*L2) = 0.0; (*L1) = L1temp; }

    // fourth check for L2 = 1
    MinPointSegDist(&Dist2,&L1temp,y1,x0,x1);
    if ( Dist2<(*MinDist2) ) { (*MinDist2) = Dist2; (*L2) = 1.0; (*L1) = L1temp; }

    // check for an internal solution
    real8 seg1[3]; V3_SUB(seg1,x1,x0);
    real8 seg2[3]; V3_SUB(seg2,y1,y0);

    real8 A =  V3_DOT(seg1,seg1);
    real8 B =  V3_DOT(seg2,seg2);
    real8 C =  V3_DOT(seg1,seg2);

    real8 E =  V3_DOT(seg1,diff);
    real8 D =  V3_DOT(seg2,diff);
    real8 G =  A*B-C*C;

    real8 eps = 1.0e-12; // this is a hard coded small distance number

    if ( fabs(G)>eps )
    {
        L1temp = (D*C-E*B)/G;
        L2temp = (D+C*L1temp)/B;

        if ( (0.0<L1temp) && (L1temp<1.0) && (0.0<L2temp) && (L2temp<1.0) )
        {
            real8 diff2[3] = { diff[0]+seg1[0]*L1temp-seg2[0]*L2temp,
            diff[1]+seg1[1]*L1temp-seg2[1]*L2temp,
            diff[2]+seg1[2]*L1temp-seg2[2]*L2temp };
            
            Dist2 = V3_DOT(diff2,diff2);
            if ( Dist2<(*MinDist2) ) { (*MinDist2) = Dist2; (*L2) = L2temp; (*L1) = L1temp; }
        }
    }
}

/***************************************************************************
 *
 *       Function:       MindistPtPtInTime()
 *
 * find the minimum distance between two points during a time increment
 * assuming that both points move with a constant velocity during the time
 * increment from t to tau
 ***************************************************************************/
KOKKOS_INLINE_FUNCTION
void MindistPtPtInTime (
         real8 *MinDist2 ,
         real8 *tratio   ,
   const real8 *x1t      ,
   const real8 *x1tau    ,
   const real8 *x2t      ,
   const real8 *x2tau    )
{
    real8 diff[3], diff2[3];

    // initialize with points at time t
    V3_SUB (diff,x1t,x2t);
    (*MinDist2) = V3_DOT(diff,diff);
    (*tratio)   = 0.0;

    //  check points at time tau
    V3_SUB (diff2,x1tau,x2tau);
    real8 Dist2 = V3_DOT(diff2,diff2);

    if (Dist2<(*MinDist2))
    {
        (*MinDist2) = Dist2;
        (*tratio)   = 1.0;
    }

    // check for an internal solution
    V3_SUB(diff2,diff2,diff);
    real8 B   = V3_DOT(diff2,diff2);
    real8 eps = 1.0e-12;

    if (B>eps)
    {
        real8 ttemp = -V3_DOT(diff,diff2)/B;

        if ( (0.0<ttemp) && (ttemp<1.0) )
        {
            real8 diff3[3] = { diff[0]+diff2[0]*ttemp,
            diff[1]+diff2[1]*ttemp,
            diff[2]+diff2[2]*ttemp };

            Dist2 = V3_DOT(diff3,diff3);
            if (Dist2<(*MinDist2))
            {
                (*MinDist2) = Dist2;
                (*tratio)   = ttemp;
            }
        }
    }
}

/***************************************************************************
 *
 *      Function:        MinDistPtSegInTime()
 *
 * Find the closest distance between a moving point and a moving line segment
 * during a time increment.
 *
 ****************************************************************************/
KOKKOS_INLINE_FUNCTION
void MinDistPtSegInTime (
         real8 *MinDist2 ,
         real8 *L2       ,
         real8 *tratio   ,
   const real8 *x1t      ,
   const real8 *x1tau    ,
   const real8 *x3t      ,
   const real8 *x3tau    ,
   const real8 *x4t      ,
   const real8 *x4tau    )
{
    //  intialize
    real8 diff[3];
    V3_SUB (diff,x1t,x3t);
    (*MinDist2) = V3_DOT(diff,diff);
    (*L2)       = 0.0;
    (*tratio)   = 0.0;

    // assume that the the minimum distance between the point and the segment
    // at tratio = 0 and tratio = 1 have already been included in the colision
    // calculation so don't repeat them here.  If that assumption is not true
    // then extra check are needed to check the boundaries of the box.
    // find the minimum distance between the point and the first endpoint of the
    // segment during time increment
    
    real8 Dist2=0.0, ttemp=0.0;
    MindistPtPtInTime(&Dist2,&ttemp,x1t,x1tau,x3t,x3tau);

    if ( Dist2<(*MinDist2) )
    {
        (*MinDist2) = Dist2;
        (*L2) = 0.0;
        (*tratio) = ttemp;
    }

    //  find the mininum distance between the point and the second enpoint of the
    //  segment during the time increment

    MindistPtPtInTime(&Dist2,&ttemp,x1t,x1tau,x4t,x4tau);

    if ( Dist2<(*MinDist2) )
    {
        (*MinDist2) = Dist2;
        (*L2) = 1.0;
        (*tratio) = ttemp;
    }

    // check for an internal solution
    real8 L2temp  = (*L2);
    ttemp  = (*tratio);
    real8  L34[3];  V3_SUB(L34,x3t,x4t);

    real8 dL34[3] = { (x3tau[0]-x3t[0])+(x4t[0]-x4tau[0]),
    (x3tau[1]-x3t[1])+(x4t[1]-x4tau[1]),
    (x3tau[2]-x3t[2])+(x4t[2]-x4tau[2]) };

    real8 dL13[3] = { (x1tau[0]-x1t[0])+(x3t[0]-x3tau[0]),
    (x1tau[1]-x1t[1])+(x3t[1]-x3tau[1]),
    (x1tau[2]-x1t[2])+(x3t[2]-x3tau[2]) };

    real8 A = V3_DOT(diff,diff);
    real8 B = V3_DOT(diff, L34);
    real8 C = V3_DOT(diff,dL13);
    real8 D = V3_DOT(diff,dL34);
    real8 E = V3_DOT( L34, L34);
    real8 F = V3_DOT( L34,dL13);
    real8 G = V3_DOT( L34,dL34);
    real8 H = V3_DOT(dL13,dL13);
    real8 I = V3_DOT(dL13,dL34);
    real8 J = V3_DOT(dL34,dL34);

    // optimize some loop invariants...
    D = (D+F);

    real8 twoG = 2.0*G;
    real8 twoI = 2.0*I;

    // newton iterate...
    for (int i=0, imax=20; (i<imax); i++)
    {
        real8  L2L2 = L2temp*L2temp;

        real8  Err[2];
        {
            Err[0] = C + D*L2temp + G*L2L2 + ttemp*(H + twoI*L2temp + J*L2L2);
            Err[1] = B + E*L2temp + ttemp*(D + twoG*L2temp)  + ttemp*ttemp*(I + J*L2temp);
        }

        real8  Mat[4];
        {
            Mat[0] = H + twoI*L2temp +   J*L2L2;
            Mat[1] = D + twoG*L2temp + 2.0*ttemp*(I + J*L2temp);
            Mat[2] = D + twoG*L2temp + 2.0*ttemp*(I + J*L2temp);
            Mat[3] = E + twoG* ttemp +     ttemp*ttemp*(J);
        }

        real8 detM   = Mat[0]*Mat[3] - Mat[2]*Mat[1];
        real8 Errmag = Err[0]*Err[0] + Err[1]*Err[1];

        // convergence tolerance set to 1 percent in both Lratio and tratio
        real8 eps = 1e-12;

        if ( (fabs(detM)<eps) || (Errmag<1e-6) )
            i = imax;
        else
        {
            ttemp  = ttemp  + (Mat[3]*Err[0] - Mat[1]*Err[1])/detM;
            L2temp = L2temp + (Mat[0]*Err[1] - Mat[2]*Err[0])/detM;
        }
    }

    if ( (0.0<L2temp) && (L2temp<1.0) && (0.0<ttemp) && (ttemp<1.0) )
    {
        real8  L2L2 = L2temp*L2temp;

        Dist2 = (A + 2.0*B*L2temp + E*L2L2)
        + 2.0*ttemp*(C +   D*L2temp + G*L2L2)
        + ttemp*ttemp*(H + twoI*L2temp + J*L2L2);

        if ( Dist2<(*MinDist2) )
        {
            (*MinDist2) = Dist2;
            (*L2)       = L2temp;
            (*tratio)   = ttemp;
        }
    }
}

/***************************************************************************
 *
 *      Function:        MinDistSegSegInTime()
 *
 *     Description:      Finds the minimum distance between to moving line segments.
 *                       The line segments are one segment going from x1 to x2 and
 *                       the other from x3 to x4. It assumed that the ends of the
 *                       line segments go from their t position to their
 *                       tau position linear in time.
 *
 ***************************************************************************/
KOKKOS_INLINE_FUNCTION
void MinDistSegSegInTime (
         real8 *MinDist2 ,
         real8 *L1ratio  ,
         real8 *L2ratio  ,
         real8 *dtratio  ,
   const real8 *x1t      ,
   const real8 *x1tau    ,
   const real8 *x2t      ,
   const real8 *x2tau    ,
   const real8 *x3t      ,
   const real8 *x3tau    ,
   const real8 *x4t      ,
   const real8 *x4tau    )
{
    real8 eps     = 1.0e-12;
    (*dtratio)    = 0.0;
    real8 Dist2   = 0.0;
    real8 L1temp  = 0.0;
    real8 L2temp  = 0.0;
    real8 dttemp  = 0.0;

    (*MinDist2) = 0.0;

    // Find the minimum distance between the segments at time t
    MinSegSegDist(MinDist2,L1ratio,L2ratio,x1t,x2t,x3t,x4t);

    // Find the minimum distance between the segments at time tau
    MinSegSegDist(&Dist2,&L1temp,&L2temp,x1tau,x2tau,x3tau,x4tau);

    if ( Dist2 < (*MinDist2) )
    {
        (*MinDist2) = Dist2;
        (*L1ratio)  = L1temp;
        (*L2ratio)  = L2temp;
        (*dtratio)  = 1.0;
    }

    // Check node 1 against line segment 34 during time increment
    MinDistPtSegInTime(&Dist2,&L2temp,&dttemp,x1t,x1tau,x3t,x3tau,x4t,x4tau);

    if ( Dist2<(*MinDist2) )
    {
        (*MinDist2) = Dist2;
        (*L1ratio)  = 0.0;
        (*L2ratio)  = L2temp;
        (*dtratio)  = dttemp;
    }

    //check node 2 against line segment 34 during time increment
    MinDistPtSegInTime(&Dist2,&L2temp,&dttemp,x2t,x2tau,x3t,x3tau,x4t,x4tau);

    if ( Dist2<(*MinDist2) )
    {
        (*MinDist2) = Dist2;
        (*L1ratio)  = 1.0;
        (*L2ratio)  = L2temp;
        (*dtratio)  = dttemp;
    }

    // check node 3 against line segment 12 during time increment
    MinDistPtSegInTime(&Dist2,&L1temp,&dttemp,x3t,x3tau,x1t,x1tau,x2t,x2tau);

    if ( Dist2<(*MinDist2) )
    {
        (*MinDist2) = Dist2;
        (*L1ratio)  = L1temp;
        (*L2ratio)  = 0.0;
        (*dtratio)  = dttemp;
    }

    // check node 4 against line segment 12 during time increment
    MinDistPtSegInTime(&Dist2,&L1temp,&dttemp,x4t,x4tau,x1t,x1tau,x2t,x2tau);

    if ( Dist2<(*MinDist2) )
    {
        (*MinDist2) = Dist2;
        (*L1ratio)  = L1temp;
        (*L2ratio)  = 1.0;
        (*dtratio)  = dttemp;
    }

    // All surface solutions have been investigated now check for an internal solution
    // Use the best surface solution as the initial guess for the iterative solve of the
    // non-linear internal problem.

    L1temp = (*L1ratio);
    L2temp = (*L2ratio);
    dttemp = (*dtratio);

    real8 L13 [3]= { x1t[0]-x3t[0], x1t[1]-x3t[1], x1t[2]-x3t[2] };
    real8 L21 [3]= { x2t[0]-x1t[0], x2t[1]-x1t[1], x2t[2]-x1t[2] };
    real8 L34 [3]= { x3t[0]-x4t[0], x3t[1]-x4t[1], x3t[2]-x4t[2] };

    real8 dL13[3]= { (x1tau[0]-x1t[0])+(x3t[0]-x3tau[0]),
    (x1tau[1]-x1t[1])+(x3t[1]-x3tau[1]),
    (x1tau[2]-x1t[2])+(x3t[2]-x3tau[2]) };

    real8 dL21[3]= { (x2tau[0]-x2t[0])+(x1t[0]-x1tau[0]),
    (x2tau[1]-x2t[1])+(x1t[1]-x1tau[1]),
    (x2tau[2]-x2t[2])+(x1t[2]-x1tau[2]) };

    real8 dL34[3]= { (x3tau[0]-x3t[0])+(x4t[0]-x4tau[0]),
    (x3tau[1]-x3t[1])+(x4t[1]-x4tau[1]),
    (x3tau[2]-x3t[2])+(x4t[2]-x4tau[2]) };

    real8 A = V3_DOT( L13, L13);
    real8 B = V3_DOT( L13, L21);
    real8 C = V3_DOT( L13, L34);
    real8 D = V3_DOT( L13,dL13);
    real8 E = V3_DOT( L13,dL21);
    real8 F = V3_DOT( L13,dL34);
    real8 G = V3_DOT( L21, L21);
    real8 H = V3_DOT( L21, L34);
    real8 I = V3_DOT( L21,dL13);
    real8 J = V3_DOT( L21,dL21);
    real8 K = V3_DOT( L21,dL34);
    real8 L = V3_DOT( L34, L34);
    real8 M = V3_DOT( L34,dL13);
    real8 N = V3_DOT( L34,dL21);
    real8 O = V3_DOT( L34,dL34);
    real8 P = V3_DOT(dL13,dL13);
    real8 Q = V3_DOT(dL13,dL21);
    real8 R = V3_DOT(dL13,dL34);
    real8 S = V3_DOT(dL21,dL21);
    real8 T = V3_DOT(dL21,dL34);
    real8 U = V3_DOT(dL34,dL34);

    // declare and optimize some loop invariants...
    E = (E+I);
    F = (F+M);
    K = (K+N);

    real8 twoQ = 2.0*Q;
    real8 twoR = 2.0*R;
    real8 twoT = 2.0*T;
    real8 twoJ = 2.0*J;
    real8 twoO = 2.0*O;

    // newton iterate...
    for (int i=0, imax=20; (i<imax); i++)
    {
        real8 dttemp2 = dttemp*dttemp;
        real8 L1L1    = L1temp*L1temp;
        real8 L2L2    = L2temp*L2temp;
        real8 L1L2    = L1temp*L2temp;

        real8 Err[3];
        {
            Err[0] = D+E*L1temp+F*L2temp+K*L1L2+J*L1L1+O*L2L2
            +dttemp*(P+twoQ*L1temp+twoR*L2temp+twoT*L1L2+S*L1L1+U*L2L2);
            Err[1] = B+H*L2temp+G*L1temp+dttemp*(E+K*L2temp+twoJ*L1temp)+dttemp2*(Q+T*L2temp+S*L1temp);
            Err[2] = C+H*L1temp+L*L2temp+dttemp*(F+K*L1temp+twoO*L2temp)+dttemp2*(R+T*L1temp+U*L2temp);
        }

        real8 Mat[9];
        {
            Mat[0] = P+twoQ*L1temp+twoR*L2temp+twoT*L1L2+S*L1L1+U*L2L2;
            Mat[1] = E+K*L2temp+twoJ*L1temp +2.0*dttemp*(Q+T*L2temp+S*L1temp);
            Mat[2] = F+K*L1temp+twoO*L2temp +2.0*dttemp*(R+T*L1temp+U*L2temp);
            Mat[3] = Mat[1];
            Mat[4] = G+twoJ*dttemp+S*dttemp2;
            Mat[5] = H+K*dttemp+T*dttemp2;
            Mat[6] = Mat[2];
            Mat[7] = Mat[5];
            Mat[8] = L+twoO*dttemp+U*dttemp2;
        }

        real8 Errmag = V3_DOT(Err,Err);

        if ( (fabs(M33_DET(Mat))<eps) || (Errmag<1.0e-6) )
            i=imax;
        else
        {
            real8 MatInv[9];
            real8 correction[3];

            M33_INV(MatInv,Mat);
            V3_M33_V3_MUL(correction,MatInv,Err);

            dttemp=dttemp-correction[0];
            L1temp=L1temp-correction[1];
            L2temp=L2temp-correction[2];
        }
    }

    if ( (0.0<dttemp) && (dttemp<1.0) && (0.0<L1temp) && (L1temp<1.0) && (0.0<L2temp) && (L2temp<1.0) )
    {
        real8 dttemp2 = dttemp*dttemp;
        real8 L1L1    = L1temp*L1temp;
        real8 L2L2    = L2temp*L2temp;
        real8 L1L2    = L1temp*L2temp;

        Dist2 = A+2.0*B*L1temp+2.0*C*L2temp+2.0*H*L1L2+G*L1L1+L*L2L2
        +2.0*dttemp *(D+E*L1temp+F*L2temp+K*L1L2+J*L1L1+O*L2L2)
        +dttemp2*(P+twoQ*L1temp+twoR*L2temp+twoT*L1L2+S*L1L1+U*L2L2);

        if (Dist2<(*MinDist2))
        {
            (*MinDist2) = Dist2 ;
            (*dtratio)  = dttemp;
            (*L1ratio)  = L1temp;
            (*L2ratio)  = L2temp;
        }
    }
}

/***************************************************************************
 *
 *      Function:        CollisionCriterion
 *
 *      Description:     This function determines whether two segments come
 *                       within a distance mindist during a time period from
 *                       t to tau.
 *                       If no, then CollisionCriterionIsMet=0, if yes then
 *                       CollisionCriterionIsMet=1, and the L1ratio,
 *                       L2ratio are given for the two segments.
 *                       The interval of time [t, tau] can be chosen to be
 *                       (1) the previous time step interval. In this case, mindist
 *                       is rann
 *                       or
 *                       (2) the next time step interval. In this case, mindist
 *                       is eps.
 *
 *
 *      Arguments:
 *                       L1ratio     : Distance from first segment to collision point
 *                       L2ratio     : Distance from second segment to collision point
 *                       x1t[3]      : Position of node 1 at time t
 *                       x1tau[3]    : Position of node 1 at time tau
 *                       x2t[3]      : Position of node 2 at time t
 *                       x2tau[3]    : Position of node 2 at time tau
 *                       x3t[3]      : Position of node 3 at time t
 *                       x3tau[3]    : Position of node 2 at time tau
 *                       x4t[3]      : Position of node 4 at time t
 *                       x4tau[3]    : Position of node 2 at time tau
 *                       mindist     : small distance within which the two segments
 *                                     meet.
 *
 ***************************************************************************/
KOKKOS_INLINE_FUNCTION
int CollisionCriterion(real8 *dist2, real8 *L1ratio, real8 *L2ratio, const real8  mindist,
                              const real8 *x1t, const real8 *x1tau,
                              const real8 *x2t, const real8 *x2tau,
                              const real8 *x3t, const real8 *x3tau,
                              const real8 *x4t, const real8 *x4tau)
{
    int CollisionCriterionIsMet = 0;

    *L1ratio  = 0.0;
    *L2ratio  = 0.0;

    real8 center1[3]={0,0,0},radius1=0.0;
    FindSphere(center1,&radius1,x1t,x2t,x1tau,x2tau);
    real8 center2[3]={0,0,0},radius2=0.0;
    FindSphere(center2,&radius2,x3t,x4t,x3tau,x4tau);

    real8 diff1[3];
    V3_SUB(diff1,center1,center2);
    real8 Dist2 = V3_DOT(diff1,diff1);

    real8 filterdist=radius1+radius2+mindist;
    filterdist=filterdist*filterdist;

    if ( Dist2<filterdist )
    {
        real8 L1=0.0,L2=0.0,dt=0.0;
        MinDistSegSegInTime(&Dist2,&L1,&L2,&dt,x1t,x1tau,x2t,x2tau,x3t,x3tau,x4t,x4tau);
        if (Dist2<(mindist*mindist) && L1>=0.0 && L1<=1.0 && L2>=0.0 && L2<=1.0)
        {
            CollisionCriterionIsMet=1;
            *L1ratio =L1;
            *L2ratio =L2;
        }
    }

    *dist2 = Dist2;
    return(CollisionCriterionIsMet);
}

/***************************************************************************
 *
 *      Function:        HingeCollisionCriterion
 *
 ***************************************************************************/ 
int HingeCollisionCriterion(real8 *L1ratio, const real8 *x1t,
                            const real8 *x3t, const real8 *x4t, const bool tri=0)
{
    *L1ratio = 0.0;
    int collisionConditionIsMet = 0;
    
    real8 L13 [3]= { x1t[0]-x3t[0], x1t[1]-x3t[1], x1t[2]-x3t[2] };
    real8 L14 [3]= { x1t[0]-x4t[0], x1t[1]-x4t[1], x1t[2]-x4t[2] };
    real8 A = V3_DOT( L13, L13);
    real8 B = V3_DOT( L14, L14);
    
    real8 tol = tri ? 0.9 : 0.98;
    if (V3_DOT( L13, L14) > tol * sqrt(A) * sqrt(B)) {
        collisionConditionIsMet = 1;
        //Based on the definition of hinge to have a small angle, we approximate the L1ratio as simply the ratio of length. 
        *L1ratio = sqrt( A / B );
    }
    return collisionConditionIsMet;
}

/*-------------------------------------------------------------------------
 *
 *      Function:     Matrix33Det
 *
 *------------------------------------------------------------------------*/
double Matrix33Det(double A[3][3])
{
    double C[3][3];
    C[0][0] = A[1][1]*A[2][2] - A[1][2]*A[2][1];
    C[1][1] = A[2][2]*A[0][0] - A[2][0]*A[0][2];
    C[2][2] = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    C[0][1] = A[1][2]*A[2][0] - A[1][0]*A[2][2];
    C[1][2] = A[2][0]*A[0][1] - A[2][1]*A[0][0];
    C[2][0] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
    C[0][2] = A[1][0]*A[2][1] - A[2][0]*A[1][1];
    C[1][0] = A[2][1]*A[0][2] - A[0][1]*A[2][2];
    C[2][1] = A[0][2]*A[1][0] - A[1][2]*A[0][0];

    return (A[0][0]*C[0][0] + A[0][1]*C[0][1] + A[0][2]*C[0][2]);
}

/*------------------------------------------------------------------------
 *
 *      Function:     MatrixMult
 *      Description:  A generic (sort of) matrix multiplier that can
 *                    be used to multiply partially populated
 *                    matrices.
 *
 *          NOTE: The physical memory layout of the input and output
 *                matrices may be larger than the actual portions
 *                of the matrices being multiplied.  The assumption
 *                is made that the components of matrix <a> involved
 *                in the operation reside in rows zero thru <aRows>-1 and
 *                columns zero thru <aCols>-1, and the corresponding
 *                values for matrices <b> and <c>.
 *
 *          a         Pointer to row 0 column 0 of first matrix to be
 *                    multiplied
 *          aRows     Row order of matrix <a>
 *          aCols     Column order of matrix <a>
 *          aLD       Leading dimension of matrix <a>
 *          b         Pointer to row 0 column 0 of second matrix to be
 *                    multiplied
 *          bCols     Column order of matrix <b>
 *          bLD       Leading dimension of matrix <b>
 *          c         Pointer to location in which to store the
 *                    results of the matrix multiply.  Matrix <c> is
 *                    assumed to be of at last dimensions <aRows> X
 *                    <bCols>.
 *          cLD       Leading dimension of matrix <c>
 *
 *----------------------------------------------------------------------*/
void MatrixMult(real8 *a, int aRows, int aCols, int aLD,
                real8 *b, int bCols, int bLD, real8 *c, int cLD)
{
    int  k, m, n;
    int  aCol, bCol, cCol;
    int  aRow, bRow, cRow;
    int  aIndex, bIndex, cIndex;

    for (m = 0; m < aRows; m++) {
        aRow = m;
        cRow = m;
        for (n = 0; n < bCols; n++) {
            bCol = n;
            cCol = n;
            cIndex = cRow * cLD + cCol;
            c[cIndex] = 0.0;
            for (k = 0; k < aCols; k++) {
                aCol = k;
                bRow = k;
                aIndex = aRow * aLD + aCol;
                bIndex = bRow * bLD + bCol;
                c[cIndex] += a[aIndex]*b[bIndex];
            }
        }
    }
}

/*-------------------------------------------------------------------------
 *
 *      Function:     MatrixInvert
 *      Description:  A general matrix inverter
 *
 *      Arguments:
 *          mat     Memory containing matrix to be converted.  Assumes
 *                  components to be inverted reside in rows zero thru
 *                  order-1 and columns zero thru order-1
 *          invMat  Memory into which the inverted matrix will be
 *                  returned to the caller.
 *          lda     Specifies the leading dimension of the matrices <mat>
 *                  and <invMat>
 *          order   Specifies the order of the matrix being inverted.
 *
 *------------------------------------------------------------------------*/
int MatrixInvert(real8 *mat, real8 *invMat, int order, int lda)
{
    int    i, j, k, offset1, offset2, matSize;
    real8  tmpVal, tmpMax, fmax, fval, eps = 1.0e-12;
    real8  *tmpMat;

    matSize = lda * lda * sizeof(real8);
    tmpMat = (real8 *)calloc(1, matSize);

    // Allocate a temporary array to help form the augmented matrix
    // for the system.  Initialize tmpMat to be a copy of the input
    // matrix and the inverse to the identity matrix.
    for (i = 0; i < order; i++) {
        for (j = 0; j < order; j++) {
            offset1 = i * lda + j;
            invMat[offset1] = (real8)(i == j);
            tmpMat[offset1] = mat[offset1];
        }
    }

    for (i = 0; i < order; i++) {

        fmax = fabs(tmpMat[i*lda+i]);

        // If tmpMat[i][i] is zero, find the next row with a non-zero
        // entry in column i and switch that row with row i.
        if (fmax < eps) {
            for (j = i+1; j < order; j++) {
                if ((tmpMax = fabs(tmpMat[j*lda+i])) > fmax) {
                    fmax = tmpMax;
                    for (k = 0; k < order; k++) {
                        offset1 = i * lda + k;
                        offset2 = j * lda + k;
                        tmpVal = tmpMat[offset1];
                        tmpMat[offset1] = tmpMat[offset2];
                        tmpMat[offset2] = tmpVal;
                        tmpVal = invMat[offset1];
                        invMat[offset1] = invMat[offset2];
                        invMat[offset2] = tmpVal;
                    }
                    break;
                }
            }
        }

        // If can't do the inversion, return 0
        if (fmax < eps) {
            printf("MatrixInvert(): unable to invert matrix!\n");
            return(0);
            //exit(1);
        }

        // Multiply all elements in row i by the inverse of tmpMat[i][i]
        // to obtain a 1 in tmpMat[i][i]
        fval = 1.0 / tmpMat[i*lda+i];

        for (j = 0; j < order; j++)   {
            offset1 = i * lda + j;
            tmpMat[offset1] *= fval;
            invMat[offset1] *= fval;
        }


        // Insure that the only non-zero value in column i is in row i
        for (k = 0; k < order; k++) {
            if (k != i) {
                fval = tmpMat[k*lda+i];
                for (j = 0; j < order;  j++) {
                    offset1 = k * lda + j;
                    offset2 = i * lda + j;
                    tmpMat[offset1] -= fval*tmpMat[offset2];
                    invMat[offset1] -= fval*invMat[offset2];
                }
            }
        }

    }   /* for (i = 0; i < order; ...) */

    free(tmpMat);

    return(1);
}

/*---------------------------------------------------------------------------
 *
 *      Function:       AdjustCollisionPoint
 *      Description:    This function attempts to select a collision
 *                      point on a plane common to the two nodes given a
 *						collision point. This is a modified version of the 
 *						old function FindCollisionPoint. It uses Lagrange 
 *						multipliers to find the point that minimizes the
 *						distance to the cPoint while satisfying all glide
 *						constraints.
 *
 *		Modified by R. B. Sills, 1/7/14
 *
 *      Arguments:
 *			cPoint[3]	   vector containing collision point found by the
						   collision detection algorithm
 *          node1, node2   pointers to the two node structures
 *          x, y, z        pointers to locations in which to return
 *                         the coordinates of the point at which the
 *                         two nodes should be collided.
 *
 *-------------------------------------------------------------------------*/
Vec3 AdjustCollisionPoint(System* system, SerialDisNet* network, int n1, int n2, 
                          Vec3& vn1, Vec3& vn2, Vec3 cPoint)
{
    real8 Nmat[3][3] = {{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
    real8 Matrix[6][6], invMatrix[6][6];
    real8 V[6], result[6];

    double eps = 1.0e-12;

    Vec3 p1 = network->nodes[n1].pos;
    Vec3 p2 = network->cell.pbc_position(p1, network->nodes[n2].pos);
    cPoint = network->cell.pbc_position(p1, cPoint);

    // If a node is a 'fixed' node it can't be relocated, so use
    // that node's coordinates as the collision point
    if (network->conn[n1].num == 1) {
        return p1;
    } else if (network->conn[n2].num == 1) {
        return p2;
    }

    double newplanecond = 0.875;
    double npc2         = newplanecond * newplanecond;
    double tmp          = 1.0 - newplanecond;
    double onemnpc4     = tmp * tmp * tmp * tmp;

    double vector[3];
    vector[0] = 0.0;
    vector[1] = 0.0;
    vector[2] = 0.0;


    // Loop over all arms of both nodes and determine the number of unique glide
    // planes. The Lagrange multipler method can handle at most 3. Populate the
    // Nmat matrix and vector with this information to be used later.
    int Nsize = 0;

    for (int i = 0; i < network->conn[n1].num; i++) {

        if (Nsize < 3) {

            int ni = network->conn[n1].node[i];
            Vec3 pi = network->cell.pbc_position(p1, network->nodes[ni].pos);
            Vec3 dir = p1 - pi;

            double L = dir.norm();
            dir = 1.0 / L * dir;

            int li = network->conn[n1].seg[i];
            Vec3 b = network->conn[n1].order[i]*network->segs[li].burg;
            Vec3 normal0 = network->segs[li].plane;
            Vec3 normal1 = cross(dir, b);
            Vec3 normal2 = cross(dir, vn1);

            double n0mag2 = normal0.norm2();
            double n1mag2 = normal1.norm2();
            double n2mag2 = normal2.norm2();

            int planeDefined = 0;
            Vec3 plane;

            if (system->crystal.use_glide_planes && n0mag2 > eps) {
                // Use segment's defined glide plane if glide planes enforced
                plane = 1.0/sqrt(n0mag2) * normal0;
                planeDefined = 1;
            } else if (n2mag2 > eps) {
                // Preference for plane defined by l cross v
                plane = 1.0/sqrt(n2mag2) * normal2;
                planeDefined = 1;
            } else if (n1mag2 > eps) {
                // Preference for plane defined by l cross b
                plane = 1.0/sqrt(n1mag2) * normal1;
                planeDefined = 1;
            } 

            if (planeDefined) {
                int conditionsmet;

                switch (Nsize) {
                    case 0:
                        conditionsmet = 1;
                        break;

                    case 1:
                        tmp = Nmat[0][0]*plane[0] +
                        Nmat[0][1]*plane[1] +
                        Nmat[0][2]*plane[2];
                        conditionsmet = (tmp*tmp < npc2);
                        break;

                    default:
                        Nmat[2][0] = plane[0];
                        Nmat[2][1] = plane[1];
                        Nmat[2][2] = plane[2];
                        double detN = Matrix33Det(Nmat);
                        conditionsmet = (detN*detN > onemnpc4);
                        break;
                }

                if (conditionsmet) {
                    Nmat[Nsize][0] = plane[0];
                    Nmat[Nsize][1] = plane[1];
                    Nmat[Nsize][2] = plane[2];
                    vector[Nsize] = dot(plane, p1);
                    Nsize++;
                }
            }
        }
    }

    for (int i = 0; i < network->conn[n2].num; i++) {

        if (Nsize < 3) {

            int ni = network->conn[n2].node[i];
            Vec3 pi = network->cell.pbc_position(p2, network->nodes[ni].pos);
            Vec3 dir = p2 - pi;

            double L = dir.norm();
            dir = 1.0 / L * dir;

            int li = network->conn[n2].seg[i];
            Vec3 b = network->conn[n2].order[i]*network->segs[li].burg;
            Vec3 normal0 = network->segs[li].plane;
            Vec3 normal1 = cross(dir, b);
            Vec3 normal2 = cross(dir, vn2);

            double n0mag2 = normal0.norm2();
            double n1mag2 = normal1.norm2();
            double n2mag2 = normal2.norm2();

            int planeDefined = 0;
            Vec3 plane;

            if (system->crystal.use_glide_planes && n0mag2 > eps) {
                // Use segment's defined glide plane if glide planes enforced
                plane = 1.0/sqrt(n0mag2) * normal0;
                planeDefined = 1;
            } else if (n2mag2 > eps) {
                // Preference for plane defined by l cross v
                plane = 1.0/sqrt(n2mag2) * normal2;
                planeDefined = 1;
            } else if (n1mag2 > eps) {
                // Preference for plane defined by l cross b
                plane = 1.0/sqrt(n1mag2) * normal1;
                planeDefined = 1;
            } 

            if (planeDefined) {
                int conditionsmet;

                switch (Nsize) {
                    case 1:
                        tmp = Nmat[0][0]*plane[0] +
                        Nmat[0][1]*plane[1] +
                        Nmat[0][2]*plane[2];
                        conditionsmet = (tmp*tmp < npc2);
                        break;
                    default:
                        Nmat[2][0] = plane[0];
                        Nmat[2][1] = plane[1];
                        Nmat[2][2] = plane[2];
                        double detN = Matrix33Det(Nmat);
                        conditionsmet = (detN*detN > onemnpc4);
                        break;
                }

                if (conditionsmet) {
                    Nmat[Nsize][0] = plane[0];
                    Nmat[Nsize][1] = plane[1];
                    Nmat[Nsize][2] = plane[2];
                    vector[Nsize] = dot(plane, p2);
                    Nsize++;
                }
            }
        }
    }

    // Upper left 3X3 of Matrix is identity matrix.
    // Matrix rows 3 thru 3+(Nsize-1) colums 0 thru 2 are Nmat.
    // Matrix columns 3 thru 3+(Nsize-1) rows 0 thru 2 are transpose of Nmat.
    // All remaining elements are zeroed.
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            Matrix[i][j] = 0.0;
    Matrix[0][0] = 1.0;
    Matrix[1][1] = 1.0;
    Matrix[2][2] = 1.0;

    for (int i = 0; i < Nsize; i++) {
        for (int j = 0; j < 3; j++) {
            Matrix[3+i][j] = Nmat[i][j];
            Matrix[j][3+i] = Nmat[i][j];
        }
    }

    V[0] = cPoint[0];
    V[1] = cPoint[1];
    V[2] = cPoint[2];
    V[3] = vector[0];
    V[4] = vector[1];
    V[5] = vector[2];

    Nsize += 3;

    if (MatrixInvert((real8 *)Matrix, (real8 *)invMatrix, Nsize, 6)) {
        MatrixMult((real8 *)invMatrix, Nsize, Nsize, 6,
                   (real8 *)V, 1, 1,
                   (real8 *)result, 1);
        return Vec3(result);
    } else {
        return cPoint;
    }
}

/*---------------------------------------------------------------------------
 *
 *      Function:       AdjustMergePoint
 *      Description:    This function attempts to select a collision
 *                      point on a plane common to all arms of the given 
						node. This is a modified version of the 
 *						old function FindCollisionPoint. It uses Lagrange 
 *						multipliers to find the point that minimizes the
 *						distance to the node's position while satisfying all glide
 *						constraints.
 *
 *		Modified by R. B. Sills, 1/16/14
 *
 *      Arguments:
 *          node		   pointers to the node structure
 *          x, y, z        pointers to locations in which to return
 *                         the coordinates of the point to which the
 *                         node should be moved.
 *
 *-------------------------------------------------------------------------*/
Vec3 AdjustMergePoint(System* system, SerialDisNet* network, int n, const Vec3& vn, Vec3 cPoint)
{
    real8 Nmat[3][3] = {{0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0}};
    real8 Matrix[6][6], invMatrix[6][6];
    real8 V[6], result[6];

    double eps = 1.0e-12;

    Vec3 p = network->nodes[n].pos;

    // If the node is a 'fixed' node it can't be relocated, so use
    // that node's coordinates as the collision point
    if (network->conn[n].num == 1) return p;

    double newplanecond = 0.875;
    double npc2 = newplanecond * newplanecond;
    double tmp = 1.0 - newplanecond;
    double onemnpc4 = tmp * tmp * tmp * tmp;

    double vector[3];
    vector[0] = 0.0;
    vector[1] = 0.0;
    vector[2] = 0.0;

    // Loop over all arms of the node and determine the number of unique glide
    // planes. The Lagrange multipler method can handle at most 3. Populate the
    // Nmat matrix and vector with this information to be used later.
    int Nsize = 0;

    for (int i = 0; i < network->conn[n].num; i++) {

        if (Nsize < 3) {
            
            int ni = network->conn[n].node[i];
            Vec3 pi = network->cell.pbc_position(p, network->nodes[ni].pos);
            Vec3 dir = p - pi;
            
            double L = dir.norm();
            dir = 1.0 / L * dir;
            
            int li = network->conn[n].seg[i];
            Vec3 b = network->conn[n].order[i]*network->segs[li].burg;
            Vec3 normal0 = network->segs[li].plane;
            Vec3 normal1 = cross(dir, b);
            Vec3 normal2 = cross(dir, vn);

            double n0mag2 = normal0.norm2();
            double n1mag2 = normal1.norm2();
            double n2mag2 = normal2.norm2();

            int planeDefined = 0;
            Vec3 plane;

            if (system->crystal.use_glide_planes && n0mag2 > eps) {
                // Use segment's defined glide plane if glide planes enforced
                plane = 1.0/sqrt(n0mag2) * normal0;
                planeDefined = 1;
            } else if (n2mag2 > eps) {
                // Preference for plane defined by l cross v
                plane = 1.0/sqrt(n2mag2) * normal2;
                planeDefined = 1;
            } else if (n1mag2 > eps) {
                // Preference for plane defined by l cross b
                plane = 1.0/sqrt(n1mag2) * normal1;
                planeDefined = 1;
            } 

            if (planeDefined) {
                int conditionsmet;

                switch (Nsize) {
                    case 0:
                        conditionsmet = 1;
                        break;
                    case 1:
                        tmp = Nmat[0][0]*plane[0] +
                        Nmat[0][1]*plane[1] +
                        Nmat[0][2]*plane[2];
                        conditionsmet = (tmp*tmp < npc2);
                        break;
                    default:
                        Nmat[2][0] = plane[0];
                        Nmat[2][1] = plane[1];
                        Nmat[2][2] = plane[2];
                        double detN = Matrix33Det(Nmat);
                        conditionsmet = (detN*detN > onemnpc4);
                        break;
                }

                if (conditionsmet) {
                    Nmat[Nsize][0] = plane[0];
                    Nmat[Nsize][1] = plane[1];
                    Nmat[Nsize][2] = plane[2];
                    vector[Nsize] = dot(plane, pi);
                    Nsize++;
                }
            }
        }
    }

    // Upper left 3X3 of Matrix is identity matrix.
    // Matrix rows 3 thru 3+(Nsize-1) colums 0 thru 2 are Nmat.
    // Matrix columns 3 thru 3+(Nsize-1) rows 0 thru 2 are transpose of Nmat.
    // All remaining elements are zeroed.
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            Matrix[i][j] = 0.0;
    Matrix[0][0] = 1.0;
    Matrix[1][1] = 1.0;
    Matrix[2][2] = 1.0;

    for (int i = 0; i < Nsize; i++) {
        for (int j = 0; j < 3; j++) {
            Matrix[3+i][j] = Nmat[i][j];
            Matrix[j][3+i] = Nmat[i][j];
        }
    }

    V[0] = p[0];
    V[1] = p[1];
    V[2] = p[2];
    V[3] = vector[0];
    V[4] = vector[1];
    V[5] = vector[2];

    Nsize += 3;

    if (MatrixInvert((real8 *)Matrix, (real8 *)invMatrix, Nsize, 6)) {
        MatrixMult((real8 *)invMatrix, Nsize, Nsize, 6,
                   (real8 *)V, 1, 1,
                   (real8 *)result, 1);
        return Vec3(result);
    } else {
        return cPoint;
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     test_collision_glide_planes()
 *                  Test all node arms for glide plane violation for a
 *                  potential collision event.
 *
 *-------------------------------------------------------------------------*/
bool test_collision_glide_planes(System* system, SerialDisNet* network, 
                                 int i, const Vec3& p, int coplanar)
{
    if (!system->crystal.use_glide_planes) return 1;
    
    double tol_test = 1e-2;  //tolerance for plane test
    double tol_len = 1e-10;  //tolerance for "zero-length" segment

    double violen_max = tol_test * system->params.rann;
    if (coplanar) violen_max *= 100.0;
    
    Vec3 r1 = network->nodes[i].pos;
    for (int j = 0; j < network->conn[i].num; j++) {
        int k = network->conn[i].node[j];
        int s = network->conn[i].seg[j];
        
        // First test the current position.
        Vec3 r2 = network->cell.pbc_position(r1, network->nodes[k].pos);
        Vec3 dr = r2-r1;
        double drlen = dr.norm();
        if (drlen < tol_len) continue;
        dr = 1.0/drlen * dr;
        
        Vec3 n = network->segs[s].plane;
        if (!system->crystal.is_crystallographic_plane(n)) continue;
        
        double dottest = fabs(dot(n, dr));
        double violen_old = dottest * drlen;

        // Then test the new position.
        r2 = network->cell.pbc_position(p, network->nodes[k].pos);
        dr = r2 - p;
        drlen = dr.norm();
        if (drlen < tol_len) continue;
        dr = 1.0/drlen * dr;
        dottest = fabs(dot(n, dr));

        // If the violation increases by more than violen_max,
        // it fails the test.
        if ((dottest * drlen - violen_old) > violen_max) {
            return 0;
        }
    }
    return 1;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     CollisionRetroactive::retroactive_collision()
 *
 *-------------------------------------------------------------------------*/
void CollisionRetroactive::retroactive_collision(System* system)
{
    double dt = system->realdt;
    double rann = system->params.rann;
    double mindist = rann;
    double mindist2 = mindist * mindist;

#if EXADIS_FULL_UNIFIED_MEMORY
    T_x& xold = system->xold;
#else
    T_x::HostMirror xold = Kokkos::create_mirror_view(system->xold);
    Kokkos::deep_copy(xold, system->xold);
#endif

    SerialDisNet* network = system->get_serial_network();
    
    // Determine the maximum cutoff distance for collisions
    double dr2max = 0.0;
    for (int i = 0; i < network->number_of_nodes(); i++) {
        Vec3 r = network->nodes[i].pos;
        Vec3 rold = network->cell.pbc_position(r, xold(i));
        double dr2 = (rold-r).norm2();
        dr2max = fmax(dr2, dr2max);
    }
    double l2max = 0.0;
    for (int i = 0; i < network->number_of_segs(); i++) {
        int n1 = network->segs[i].n1;
        int n2 = network->segs[i].n2;
        Vec3 p1 = network->nodes[n1].pos;
        Vec3 p2 = network->cell.pbc_position(p1, network->nodes[n2].pos);
        double l2 = (p2-p1).norm2();
        l2max = fmax(l2, l2max);
    }
    double cutoff = sqrt(4.0*dr2max+l2max/2.0) + rann;
    
    double maxseg = system->params.maxseg;
    NeighborBin *neighbor = generate_neighbor_segs(network, cutoff, maxseg);

    int nmerge = 0;
    int nnodes = network->number_of_nodes();
    int nsegs = network->number_of_segs();
    std::vector<int> skipseg(nsegs, 0);
    
    // Look for collisions between segments
    for (int i = 0; i < nsegs; i++) {
        if (skipseg[i]) continue;
        
        DisSeg *seg = &network->segs[i];
        
        int n1 = seg->n1;
        int n2 = seg->n2;
        if (n1 >= nnodes || n2 >= nnodes) continue;
        Vec3 p1 = network->nodes[n1].pos;
        Vec3 p2 = network->cell.pbc_position(p1, network->nodes[n2].pos);
        
        Vec3 l12 = p1-p2;
        if (l12.norm2() < 1e-20) continue;
        
        Vec3 pold1 = network->cell.pbc_position(p1, xold(n1));
        Vec3 pold2 = network->cell.pbc_position(p2, xold(n2));
        
        Vec3 pseg = 0.5*(p1+p2);
        auto neilist = neighbor->query(pseg);
    
        for (int j = 0; j < neilist.size(); j++) {
            int k = neilist[j];
            if (i <= k) continue; // collision with segments k>i
            if (skipseg[k]) continue;
            
            int n3 = network->segs[k].n1;
            int n4 = network->segs[k].n2;
            if (n3 >= nnodes || n4 >= nnodes) continue;
            
            // Hinges
            if (n3 == n1 || n3 == n2 || n4 == n1 || n4 == n2) continue;
            
            Vec3 p3 = network->cell.pbc_position(p1, network->nodes[n3].pos);
            Vec3 p4 = network->cell.pbc_position(p3, network->nodes[n4].pos);
            
            Vec3 l34 = p3-p4;
            if (l34.norm2() < 1e-20) continue;
            
            Vec3 pold3 = network->cell.pbc_position(p3, xold(n3));
            Vec3 pold4 = network->cell.pbc_position(p4, xold(n4));
        
            /* First interval considered : [t-delta t, t] */
            double dist2 = 0.0;
            double L1, L2;
            int collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,mindist,
                                                             &pold1[0],&p1[0],&pold2[0],&p2[0],
                                                             &pold3[0],&p3[0],&pold4[0],&p4[0]);
            //printf("try collision %d %d\n",i,k);
            if (!collisionConditionIsMet) {
                // No collision in the past, look for a possible collision in the future....
                // Determine where the points will be at the next time step (estimate only).
                // Approximation : v_{n+1} equal to v_n
                /* Second trial interval considered is : [t, t + deltat] */
                Vec3 pnext1 = p1 + dt * network->nodes[n1].v;
                Vec3 pnext2 = p2 + dt * network->nodes[n2].v;
                Vec3 pnext3 = p3 + dt * network->nodes[n3].v;
                Vec3 pnext4 = p4 + dt * network->nodes[n4].v;
                
                collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,1.e-6,
                                                             &p1[0],&pnext1[0],&p2[0],&pnext2[0],
                                                             &p3[0],&pnext3[0],&p4[0],&pnext4[0]);
            }
            if (!collisionConditionIsMet) continue;
            
            // Coplanar segments check
            bool coplanar = 0;
            if (system->crystal.use_glide_planes)
                coplanar = is_collinear(network->segs[i].plane, network->segs[k].plane);
            
            // Identify the first node to be merged
            int close2node1 = ( ((L1 * L1)         * l12.norm2()) <mindist2);
            int close2node2 = ( ((1.0-L1)*(1.0-L1) * l12.norm2()) <mindist2);
            int mergenode1;
            Vec3 vmn1;
            if (close2node1) {
                mergenode1 = n1;
                vmn1 = network->nodes[n1].v;
            } else if (close2node2) {
                mergenode1 = n2;
                vmn1 = network->nodes[n2].v;
            } else {
                Vec3 pnew = (1.0-L1) * p1 + L1 * p2;
                pnew = network->cell.pbc_fold(pnew);
                mergenode1 = network->split_seg(i, pnew);
                vmn1 = (1.0-L1) * network->nodes[n1].v + L1 * network->nodes[n2].v;
                skipseg[i] = 1;
            }
            
            // Identify the second node to be merged
            int close2node3 = ( ((L2 * L2)         * l34.norm2()) <mindist2);
            int close2node4 = ( ((1.0-L2)*(1.0-L2) * l34.norm2()) <mindist2);
            int mergenode2;
            Vec3 vmn2;
            if (close2node3) {
                mergenode2 = n3;
                vmn2 = network->nodes[n3].v;
            } else if (close2node4) {
                mergenode2 = n4;
                vmn2 = network->nodes[n4].v;
            } else {
                Vec3 pnew = (1.0-L2) * p3 + L2 * p4;
                pnew = network->cell.pbc_fold(pnew);
                mergenode2 = network->split_seg(k, pnew);
                vmn2 = (1.0-L2) * network->nodes[n3].v + L2 * network->nodes[n4].v;
                skipseg[k] = 1;
            }
            
            /*
            if (network->use_glide_planes) {
                // Compute mathematical intersection between all planes
                std::vector<SlipPlane> plist;
                for (int m : {mergenode1, mergenode2}) {
                    for (int l = 0; l < conn[m].size(); l++)
                        plist.push_back(network->links[conn[m][l].link].plane);
                }
                GeomObject inter = glide_planes_intersection(plist);
                // Skip if the intersection does not exist
                if (inter.null()) continue;
            }
            */
            
            // Find newPos: the position of the collision point.
            Vec3 c1 = network->nodes[mergenode1].pos;
            Vec3 c2 = network->cell.pbc_position(c1, network->nodes[mergenode2].pos);
            Vec3 newpos = 0.5 * (c1 + c2);
            newpos = AdjustCollisionPoint(system, network, mergenode1, mergenode2, vmn1, vmn2, newpos);
            newpos = network->cell.pbc_fold(newpos);
            
            //printf("collision %d %d\n",i,k);
            
            // If it looks like the node will be thrown a significant distance
            // don't do the collision.
            Vec3 d1 = network->nodes[mergenode1].pos - network->cell.pbc_position(network->nodes[mergenode1].pos, newpos);
            Vec3 d2 = network->nodes[mergenode2].pos - network->cell.pbc_position(network->nodes[mergenode2].pos, newpos);
            if ((d1.norm2() > 16 * mindist2) && (d2.norm2() > 16 * mindist2)) {
                continue;
            }
            
            // Test for glide plane violations
            if (system->crystal.enforce_glide_planes) {
                if (!test_collision_glide_planes(system, network, mergenode1, newpos, coplanar) ||
                    !test_collision_glide_planes(system, network, mergenode2, newpos, coplanar))
                    continue;
            }
            
            /*
            // Do not let junctions form
            if (!network->form_junctions) {
                if (merge_nodes_junction(network, conn, mergenode1, mergenode2)) continue;
            }
            */
            // Last check...
            //int gpv = merge_nodes_plane_violation(network, conn, mergenode1, mergenode2, newpos);
            //if (gpv) {
            //    printf("Warning: glide plane violation during merge collision\n");
            //}
            
            // Flag all connected links for subsequent collisions
            for (int l = 0; l < network->conn[mergenode1].num; l++) {
                if (network->conn[mergenode1].seg[l] < nsegs)
                    skipseg[network->conn[mergenode1].seg[l]] = 1;
            }
            for (int l = 0; l < network->conn[mergenode2].num; l++) {
                if (network->conn[mergenode2].seg[l] < nsegs)
                    skipseg[network->conn[mergenode2].seg[l]] = 1;
            }
            skipseg[i] = 1;
            skipseg[k] = 1;
            
            // Check glide plane violations
            //check_node_plane_violation(network, conn, mergenode1, "before merge collision");
            //check_node_plane_violation(network, conn, mergenode2, "before merge collision");
            
            // Merge nodes
            bool merge_error = network->merge_nodes_position(mergenode1, mergenode2, newpos, system->dEp);
            nmerge++;
            
            // Attempt to fix glide plane violations that may have been
            // introduced during the merge operation.
            if (!merge_error && system->crystal.use_glide_planes) {
                
                system->crystal.reset_node_glide_planes(network, mergenode1);
                
                Vec3 newpos0 = newpos;
                newpos = AdjustMergePoint(system, network, mergenode1, vmn1, newpos);
                
                if ((newpos-newpos0).norm2() <= 16 * mindist2)
                    network->move_node(mergenode1, newpos, system->dEp);
            }
            
            // Check glide plane violations
            //std::string msg = "after merge collision step "+std::to_string(step);
            //check_node_plane_violation(network, conn, mergenode1, msg);
            
            break;
        }
    }
    
    delete neighbor;
    
  
    // Now we have to loop for collisions on hinge joints (i.e zipping)
    for (int i = 0; i < nnodes; i++) {
        
        Vec3 p1 = network->nodes[i].pos;
        
        for (int j = 0; j < network->conn[i].num; j++) {
            int l3 = network->conn[i].seg[j];
            //if (l3 >= nsegs) continue;
            //if (skipseg[l3]) continue;
            
            int n3 = network->conn[i].node[j];
            if (n3 >= nnodes) continue;
            Vec3 p3 = network->cell.pbc_position(p1, network->nodes[n3].pos);
            Vec3 l13 = p1-p3;
            if (l13.norm2() < 1e-20) continue;
            
            for (int k = j+1; k < network->conn[i].num; k++) {
                int l4 = network->conn[i].seg[k];
                //if (l4 >= nsegs) continue;
                //if (skipseg[l4]) continue;
                
                int n4 = network->conn[i].node[k];
                if (n4 >= nnodes) continue;
                Vec3 p4 = network->cell.pbc_position(p3, network->nodes[n4].pos);
                Vec3 l14 = p1-p4;
                if (l14.norm2() < 1e-20) continue;
                
                bool tri = (network->find_connection(n3, n4) != -1);
                
                /* First interval considered : [t-delta t, t] */
                //double dist2 = 0.0;
                double L1;/*, L2;*/
                /*
                int collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,mindist,
                                                                 &pold1[0],&p1[0],&pold4[0],&p4[0],
                                                                 &pold3[0],&p3[0],&pold3[0],&p3[0]);
                */
                int collisionConditionIsMet = HingeCollisionCriterion(&L1,&p1[0],&p3[0],&p4[0]);
                
                if (!collisionConditionIsMet) {
                    // No collision in the past, look for a possible collision in the future....
                    Vec3 pnext1 = p1 + dt * network->nodes[i].v;
                    Vec3 pnext3 = p3 + dt * network->nodes[n3].v;
                    Vec3 pnext4 = p4 + dt * network->nodes[n4].v;
                    /*
                    collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,1.e-6,
                                                                 &p1[0],&pnext1[0],&p4[0],&pnext4[0],
                                                                 &p3[0],&pnext3[0],&p3[0],&pnext3[0]);
                    */
                    collisionConditionIsMet = HingeCollisionCriterion(&L1,&pnext1[0],&pnext3[0],&pnext4[0]);
                }
                if (!collisionConditionIsMet) continue;
                
                
                // Use end point on shortest segment as one of the collision points
                int mergenode1, mergenode2;
                Vec3 vmn1, vmn2;
                
                if (l14.norm2() < l13.norm2()) {
                    // 14 is the shortest segment, use node4 as one of the collision points
                    mergenode1 = n4;
                    vmn1 = network->nodes[n4].v;
                    
                    L1 = sqrt(l14.norm2() / l13.norm2());
                    int close2node1 = ( ((L1*L1) * l13.norm2()) <mindist2);
                    int close2node3 = ( ((1.0-L1)*(1.0-L1) * l13.norm2())<mindist2);
                    if (close2node1) {
                        mergenode2 = i;
                        //continue; // no need to merge 4 with 1
                    } else if (close2node3) {
                        mergenode2 = n3;
                        vmn2 = network->nodes[n3].v;
                    } else {
                        Vec3 pnew = (1.0-L1) * p1 + L1 * p3;
                        pnew = network->cell.pbc_fold(pnew);
                        mergenode2 = network->split_seg(l3, pnew);
                        vmn2 = (1.0-L1) * network->nodes[i].v + L1 * network->nodes[n3].v;
                    }
                } else {
                    // 13 is the shortest segment, use node3 as one of the collision points
                    mergenode1 = n3;
                    vmn1 = network->nodes[n3].v;
                    
                    L1 = sqrt(l13.norm2() / l14.norm2());
                    int close2node1 = ( ((L1*L1) * l14.norm2()) <mindist2);
                    int close2node4 = ( ((1.0-L1)*(1.0-L1) * l14.norm2())<mindist2);
                    if (close2node1) {
                        mergenode2 = i;
                        //continue; // no need to merge 3 with 1
                    } else if (close2node4) {
                        mergenode2 = n4;
                        vmn2 = network->nodes[n4].v;
                    } else {
                        Vec3 pnew = (1.0-L1) * p1 + L1 * p4;
                        pnew = network->cell.pbc_fold(pnew);
                        mergenode2 = network->split_seg(l4, pnew);
                        vmn2 = (1.0-L1) * network->nodes[i].v + L1 * network->nodes[n4].v;
                    }
                }
                
                /*
                if (network->use_glide_planes) {
                    // Compute mathematical intersection between all planes
                    std::vector<SlipPlane> plist;
                    for (int m : {mergenode1, mergenode2}) {
                        for (int l = 0; l < conn[m].size(); l++)
                            plist.push_back(network->links[conn[m][l].link].plane);
                    }
                    GeomObject inter = glide_planes_intersection(plist);
                    // Skip if the intersection does not exist
                    if (inter.null()) continue;
                }
                */
                
                Vec3 c1 = network->nodes[mergenode1].pos;
                Vec3 c2 = network->cell.pbc_position(c1, network->nodes[mergenode2].pos);
                Vec3 newpos = 0.5 * (c1 + c2);
                newpos = AdjustCollisionPoint(system, network, mergenode1, mergenode2, vmn1, vmn2, newpos);
                newpos = network->cell.pbc_fold(newpos);
                
                // If it looks like the node will be thrown a significant distance
                // don't do the collision.
                Vec3 d1 = network->nodes[mergenode1].pos - network->cell.pbc_position(network->nodes[mergenode1].pos, newpos);
                if (d1.norm2() > 16 * mindist2) {
                    continue;
                }
                
                // Test for glide plane violations
                if (system->crystal.enforce_glide_planes && !tri) {
                    if (!test_collision_glide_planes(system, network, mergenode1, newpos, 0) ||
                        !test_collision_glide_planes(system, network, mergenode2, newpos, 0))
                        continue;
                }
                
                /*
                // Do not let junctions form
                if (!network->form_junctions) {
                    if (merge_nodes_junction(network, conn, mergenode1, mergenode2)) continue;
                }
                */
                
                // Last check...
                //int gpv = merge_nodes_plane_violation(network, conn, mergenode1, mergenode2, newpos);
                //if (gpv) {
                //    printf("Warning: glide plane violation during merge hinge collision\n");
                //}
                
                //check_node_plane_violation(network, conn, mergenode1, "before merge hinge collision");
                //check_node_plane_violation(network, conn, mergenode2, "before merge hinge collision");
                
                bool merge_error = network->merge_nodes_position(mergenode1, mergenode2, newpos, system->dEp);
                nmerge++;
                
                // Attempt to fix glide plane violations that may have been
                // introduced during the merge operation.
                if (!merge_error && system->crystal.use_glide_planes) {
                    
                    system->crystal.reset_node_glide_planes(network, mergenode1);
                    
                    Vec3 newpos0 = newpos;
                    newpos = AdjustMergePoint(system, network, mergenode1, vmn1, newpos);
                    
                    if ((newpos-newpos0).norm2() <= 16 * mindist2)
                        network->move_node(mergenode1, newpos, system->dEp);
                }
                
                //check_node_plane_violation(network, conn, mergenode1, "after merge hinge collision");
            }
        }
    }
    
    if (nmerge > 0)
        network->purge_network();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     CollisionRetroactive::retroactive_collision_parallel()
 *
 *-------------------------------------------------------------------------*/
void CollisionRetroactive::retroactive_collision_parallel(System* system)
{
    double dt = system->realdt;
    double rann = system->params.rann;
    double mindist = rann;
    double mindist2 = mindist * mindist;

    T_x& xold = system->xold;
    DeviceDisNet* net = system->get_device_network();
    
    // Determine the maximum cutoff distance for collisions
    auto nodes = net->get_nodes();
    auto segs = net->get_segs();
    auto cell = net->cell;
    
    double dr2max = 0.0;
    Kokkos::parallel_reduce(net->Nnodes_local, KOKKOS_LAMBDA(const int i, double& dr2val) {
        Vec3 r = nodes[i].pos;
        Vec3 rold = cell.pbc_position(r, xold(i));
        double dr2 = (rold-r).norm2();
        if (dr2 > dr2val) dr2val = dr2;
    }, Kokkos::Max<double>(dr2max));
    
    double l2max = 0.0;
    Kokkos::parallel_reduce(net->Nsegs_local, KOKKOS_LAMBDA(const int i, double& l2val) {
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        Vec3 p1 = nodes[n1].pos;
        Vec3 p2 = cell.pbc_position(p1, nodes[n2].pos);
        double l2 = (p2-p1).norm2();
        if (l2 > l2val) l2val = l2;
    }, Kokkos::Max<double>(l2max));
    Kokkos::fence();
    
    double cutoff = sqrt(4.0*dr2max+0.5*l2max) + rann;
    generate_neighbor_list(system, net, neilist, cutoff, Neighbor::NeiSeg);
    NeighborList* d_neilist = neilist;
    
    // Look for collisions between segments
    int max_collisions = 2 * net->Nsegs_local;
    Kokkos::View<int, T_memory_shared> ncollisions("ncollisions");
    Kokkos::View<int**, T_memory_shared> collisions("collisions", max_collisions, 2);
    Kokkos::View<double**, T_memory_shared> Lcollisions("Lcollisions", max_collisions, 2);
    
    Kokkos::parallel_for(net->Nsegs_local, KOKKOS_LAMBDA(const int& i) {
        int n1 = segs[i].n1;
        int n2 = segs[i].n2;
        Vec3 p1 = nodes[n1].pos;
        Vec3 p2 = cell.pbc_position(p1, nodes[n2].pos);
        
        Vec3 l12 = p1-p2;
        if (l12.norm2() < 1e-20) return;
        
        Vec3 pold1 = cell.pbc_position(p1, xold(n1));
        Vec3 pold2 = cell.pbc_position(p2, xold(n2));
        
        auto count = d_neilist->get_count();
        auto nei = d_neilist->get_nei();
        
        int Nnei = count[i];
        for (int j = 0; j < Nnei; j++) {
            int k = nei(i,j); // neighbor seg
            if (i <= k) continue; // collision with segments k>i
            
            int n3 = segs[k].n1;
            int n4 = segs[k].n2;
            
            // Hinges
            if (n3 == n1 || n3 == n2 || n4 == n1 || n4 == n2) continue;
            
            Vec3 p3 = cell.pbc_position(p1, nodes[n3].pos);
            Vec3 p4 = cell.pbc_position(p3, nodes[n4].pos);
            
            Vec3 l34 = p3-p4;
            if (l34.norm2() < 1e-20) continue;
            
            Vec3 pold3 = cell.pbc_position(p3, xold(n3));
            Vec3 pold4 = cell.pbc_position(p4, xold(n4));
            
            /* First interval considered : [t-delta t, t] */
            double dist2 = 0.0;
            double L1, L2;
            int collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,mindist,
                                                             &pold1[0],&p1[0],&pold2[0],&p2[0],
                                                             &pold3[0],&p3[0],&pold4[0],&p4[0]);
            //printf("try collision %d %d\n",i,k);
            if (!collisionConditionIsMet) {
                // No collision in the past, look for a possible collision in the future....
                // Determine where the points will be at the next time step (estimate only).
                // Approximation : v_{n+1} equal to v_n
                /* Second trial interval considered is : [t, t + deltat] */
                Vec3 pnext1 = p1 + dt * nodes[n1].v;
                Vec3 pnext2 = p2 + dt * nodes[n2].v;
                Vec3 pnext3 = p3 + dt * nodes[n3].v;
                Vec3 pnext4 = p4 + dt * nodes[n4].v;
                
                collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,1.e-6,
                                                             &p1[0],&pnext1[0],&p2[0],&pnext2[0],
                                                             &p3[0],&pnext3[0],&p4[0],&pnext4[0]);
            }
            if (!collisionConditionIsMet) continue;
            
            // Store the segment pair for collision
            int idx = Kokkos::atomic_fetch_add(&ncollisions(), 1);
            if (idx < max_collisions) {
                collisions(idx,0) = i;
                collisions(idx,1) = k;
                Lcollisions(idx,0) = L1;
                Lcollisions(idx,1) = L2;
            }
        }
    });
    Kokkos::fence();
    
    if (max_collisions > 0 && ncollisions() >= max_collisions) {
        ncollisions() = max_collisions;
        ExaDiS_log("Warning: max collisions have been reached. Some collisions have been ignored\n");
    }
    
    
    SerialDisNet* network = system->get_serial_network();
    
    int nmerge = 0;
    int nnodes = network->Nnodes_local;
    int nsegs = network->Nsegs_local;
    std::vector<int> skipseg(nsegs, 0);
    
    // Now execute collisions found in the previous step
    for (int c = 0; c < ncollisions(); c++) {
        
        int i = collisions(c,0);
        int k = collisions(c,1);
        
        if (skipseg[i] || skipseg[k]) continue;
        
        int n1 = network->segs[i].n1;
        int n2 = network->segs[i].n2;
        if (n1 >= nnodes || n2 >= nnodes) continue;
        
        int n3 = network->segs[k].n1;
        int n4 = network->segs[k].n2;
        if (n3 >= nnodes || n4 >= nnodes) continue;
        
        Vec3 p1 = network->nodes[n1].pos;
        Vec3 p2 = network->cell.pbc_position(p1, network->nodes[n2].pos);
        Vec3 l12 = p1-p2;
            
        Vec3 p3 = network->cell.pbc_position(p1, network->nodes[n3].pos);
        Vec3 p4 = network->cell.pbc_position(p3, network->nodes[n4].pos);
        Vec3 l34 = p3-p4;
        
        double L1 = Lcollisions(c,0);
        double L2 = Lcollisions(c,1);
        
        
        // Coplanar segments check
        bool coplanar = 0;
        if (system->crystal.use_glide_planes)
            coplanar = is_collinear(network->segs[i].plane, network->segs[k].plane);
        
        // Identify the first node to be merged
        int close2node1 = ( ((L1 * L1)         * l12.norm2()) <mindist2);
        int close2node2 = ( ((1.0-L1)*(1.0-L1) * l12.norm2()) <mindist2);
        int mergenode1;
        Vec3 vmn1;
        if (close2node1) {
            mergenode1 = n1;
            vmn1 = network->nodes[n1].v;
        } else if (close2node2) {
            mergenode1 = n2;
            vmn1 = network->nodes[n2].v;
        } else {
            Vec3 pnew = (1.0-L1) * p1 + L1 * p2;
            pnew = network->cell.pbc_fold(pnew);
            mergenode1 = network->split_seg(i, pnew);
            vmn1 = (1.0-L1) * network->nodes[n1].v + L1 * network->nodes[n2].v;
            skipseg[i] = 1;
        }
        
        // Identify the second node to be merged
        int close2node3 = ( ((L2 * L2)         * l34.norm2()) <mindist2);
        int close2node4 = ( ((1.0-L2)*(1.0-L2) * l34.norm2()) <mindist2);
        int mergenode2;
        Vec3 vmn2;
        if (close2node3) {
            mergenode2 = n3;
            vmn2 = network->nodes[n3].v;
        } else if (close2node4) {
            mergenode2 = n4;
            vmn2 = network->nodes[n4].v;
        } else {
            Vec3 pnew = (1.0-L2) * p3 + L2 * p4;
            pnew = network->cell.pbc_fold(pnew);
            mergenode2 = network->split_seg(k, pnew);
            vmn2 = (1.0-L2) * network->nodes[n3].v + L2 * network->nodes[n4].v;
            skipseg[k] = 1;
        }
        
        /*
        if (network->use_glide_planes) {
            // Compute mathematical intersection between all planes
            std::vector<SlipPlane> plist;
            for (int m : {mergenode1, mergenode2}) {
                for (int l = 0; l < conn[m].size(); l++)
                    plist.push_back(network->links[conn[m][l].link].plane);
            }
            GeomObject inter = glide_planes_intersection(plist);
            // Skip if the intersection does not exist
            if (inter.null()) continue;
        }
        */
        
        // Find newPos: the position of the collision point.
        Vec3 c1 = network->nodes[mergenode1].pos;
        Vec3 c2 = network->cell.pbc_position(c1, network->nodes[mergenode2].pos);
        Vec3 newpos = 0.5 * (c1 + c2);
        newpos = AdjustCollisionPoint(system, network, mergenode1, mergenode2, vmn1, vmn2, newpos);
        newpos = network->cell.pbc_fold(newpos);
        
        //printf("collision %d %d\n",i,k);
        
        // If it looks like the node will be thrown a significant distance
        // don't do the collision.
        Vec3 d1 = network->nodes[mergenode1].pos - network->cell.pbc_position(network->nodes[mergenode1].pos, newpos);
        Vec3 d2 = network->nodes[mergenode2].pos - network->cell.pbc_position(network->nodes[mergenode2].pos, newpos);
        if ((d1.norm2() > 16 * mindist2) && (d2.norm2() > 16 * mindist2)) {
            continue;
        }
        
        // Test for glide plane violations
        if (system->crystal.enforce_glide_planes) {
            if (!test_collision_glide_planes(system, network, mergenode1, newpos, coplanar) ||
                !test_collision_glide_planes(system, network, mergenode2, newpos, coplanar))
                continue;
        }
        
        /*
        // Do not let junctions form
        if (!network->form_junctions) {
            if (merge_nodes_junction(network, conn, mergenode1, mergenode2)) continue;
        }
        */
        // Last check...
        //int gpv = merge_nodes_plane_violation(network, conn, mergenode1, mergenode2, newpos);
        //if (gpv) {
        //    printf("Warning: glide plane violation during merge collision\n");
        //}
        
        // Flag all connected links for subsequent collisions
        for (int l = 0; l < network->conn[mergenode1].num; l++) {
            if (network->conn[mergenode1].seg[l] < nsegs)
                skipseg[network->conn[mergenode1].seg[l]] = 1;
        }
        for (int l = 0; l < network->conn[mergenode2].num; l++) {
            if (network->conn[mergenode2].seg[l] < nsegs)
                skipseg[network->conn[mergenode2].seg[l]] = 1;
        }
        skipseg[i] = 1;
        skipseg[k] = 1;
        
        // Check glide plane violations
        //check_node_plane_violation(network, conn, mergenode1, "before merge collision");
        //check_node_plane_violation(network, conn, mergenode2, "before merge collision");
        
        // Merge nodes
        bool merge_error = network->merge_nodes_position(mergenode1, mergenode2, newpos, system->dEp);
        nmerge++;
        
        // Attempt to fix glide plane violations that may have been
        // introduced during the merge operation.
        if (!merge_error && system->crystal.use_glide_planes) {
            
            system->crystal.reset_node_glide_planes(network, mergenode1);
            
            Vec3 newpos0 = newpos;
            newpos = AdjustMergePoint(system, network, mergenode1, vmn1, newpos);
            
            if ((newpos-newpos0).norm2() <= 16 * mindist2)
                network->move_node(mergenode1, newpos, system->dEp);
        }
        
        // Check glide plane violations
        //std::string msg = "after merge collision step "+std::to_string(step);
        //check_node_plane_violation(network, conn, mergenode1, msg);
    }
    
/*
#if EXADIS_FULL_UNIFIED_MEMORY
    T_x& h_xold = system->xold;
#else
    T_x::HostMirror h_xold = Kokkos::create_mirror_view(system->xold);
    Kokkos::deep_copy(h_xold, system->xold);
#endif
*/
    
    // Now we have to loop for collisions on hinge joints (i.e zipping)
    for (int i = 0; i < nnodes; i++) {
        
        Vec3 p1 = network->nodes[i].pos;
        
        for (int j = 0; j < network->conn[i].num; j++) {
            int l3 = network->conn[i].seg[j];
            //if (l3 >= nsegs) continue;
            //if (skipseg[l3]) continue;
            
            int n3 = network->conn[i].node[j];
            if (n3 >= nnodes) continue;
            Vec3 p3 = network->cell.pbc_position(p1, network->nodes[n3].pos);
            Vec3 l13 = p1-p3;
            if (l13.norm2() < 1e-20) continue;
            
            for (int k = j+1; k < network->conn[i].num; k++) {
                int l4 = network->conn[i].seg[k];
                //if (l4 >= nsegs) continue;
                //if (skipseg[l4]) continue;
                
                int n4 = network->conn[i].node[k];
                if (n4 >= nnodes) continue;
                Vec3 p4 = network->cell.pbc_position(p3, network->nodes[n4].pos);
                Vec3 l14 = p1-p4;
                if (l14.norm2() < 1e-20) continue;
                
                bool tri = (network->find_connection(n3, n4) != -1);
                
                /* First interval considered : [t-delta t, t] */
                //double dist2 = 0.0;
                double L1;/*, L2;*/
                /*
                int collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,mindist,
                                                                 &pold1[0],&p1[0],&pold4[0],&p4[0],
                                                                 &pold3[0],&p3[0],&pold3[0],&p3[0]);
                */
                int collisionConditionIsMet = HingeCollisionCriterion(&L1,&p1[0],&p3[0],&p4[0],tri);
                if (!collisionConditionIsMet) {
                    // No collision in the past, look for a possible collision in the future....
                    Vec3 pnext1 = p1 + dt * network->nodes[i].v;
                    Vec3 pnext3 = p3 + dt * network->nodes[n3].v;
                    Vec3 pnext4 = p4 + dt * network->nodes[n4].v;
                    /*
                    collisionConditionIsMet = CollisionCriterion(&dist2,&L1,&L2,1.e-6,
                                                                 &p1[0],&pnext1[0],&p4[0],&pnext4[0],
                                                                 &p3[0],&pnext3[0],&p3[0],&pnext3[0]);
                    */
                    collisionConditionIsMet = HingeCollisionCriterion(&L1,&pnext1[0],&pnext3[0],&pnext4[0],tri);
                }
                if (!collisionConditionIsMet) continue;
                
                
                // Use end point on shortest segment as one of the collision points
                int mergenode1, mergenode2;
                Vec3 vmn1, vmn2;
                
                if (l14.norm2() < l13.norm2()) {
                    // 14 is the shortest segment, use node4 as one of the collision points
                    mergenode1 = n4;
                    vmn1 = network->nodes[n4].v;
                    
                    L1 = sqrt(l14.norm2() / l13.norm2());
                    int close2node1 = ( ((L1*L1) * l13.norm2()) <mindist2);
                    int close2node3 = ( ((1.0-L1)*(1.0-L1) * l13.norm2())<mindist2);
                    if (close2node1) {
                        mergenode2 = i;
                        //continue; // no need to merge 4 with 1
                    } else if (close2node3) {
                        mergenode2 = n3;
                        vmn2 = network->nodes[n3].v;
                    } else {
                        Vec3 pnew = (1.0-L1) * p1 + L1 * p3;
                        pnew = network->cell.pbc_fold(pnew);
                        mergenode2 = network->split_seg(l3, pnew);
                        vmn2 = (1.0-L1) * network->nodes[i].v + L1 * network->nodes[n3].v;
                    }
                } else {
                    // 13 is the shortest segment, use node3 as one of the collision points
                    mergenode1 = n3;
                    vmn1 = network->nodes[n3].v;
                    
                    L1 = sqrt(l13.norm2() / l14.norm2());
                    int close2node1 = ( ((L1*L1) * l14.norm2()) <mindist2);
                    int close2node4 = ( ((1.0-L1)*(1.0-L1) * l14.norm2())<mindist2);
                    if (close2node1) {
                        mergenode2 = i;
                        //continue; // no need to merge 3 with 1
                    } else if (close2node4) {
                        mergenode2 = n4;
                        vmn2 = network->nodes[n4].v;
                    } else {
                        Vec3 pnew = (1.0-L1) * p1 + L1 * p4;
                        pnew = network->cell.pbc_fold(pnew);
                        mergenode2 = network->split_seg(l4, pnew);
                        vmn2 = (1.0-L1) * network->nodes[i].v + L1 * network->nodes[n4].v;
                    }
                }
                
                /*
                if (network->use_glide_planes) {
                    // Compute mathematical intersection between all planes
                    std::vector<SlipPlane> plist;
                    for (int m : {mergenode1, mergenode2}) {
                        for (int l = 0; l < conn[m].size(); l++)
                            plist.push_back(network->links[conn[m][l].link].plane);
                    }
                    GeomObject inter = glide_planes_intersection(plist);
                    // Skip if the intersection does not exist
                    if (inter.null()) continue;
                }
                */
                
                Vec3 c1 = network->nodes[mergenode1].pos;
                Vec3 c2 = network->cell.pbc_position(c1, network->nodes[mergenode2].pos);
                Vec3 newpos = 0.5 * (c1 + c2);
                newpos = AdjustCollisionPoint(system, network, mergenode1, mergenode2, vmn1, vmn2, newpos);
                newpos = network->cell.pbc_fold(newpos);
                
                // If it looks like the node will be thrown a significant distance
                // don't do the collision.
                Vec3 d1 = network->nodes[mergenode1].pos - network->cell.pbc_position(network->nodes[mergenode1].pos, newpos);
                if (d1.norm2() > 16 * mindist2) {
                    continue;
                }
                
                // Test for glide plane violations
                if (system->crystal.enforce_glide_planes && !tri) {
                    if (!test_collision_glide_planes(system, network, mergenode1, newpos, 0) ||
                        !test_collision_glide_planes(system, network, mergenode2, newpos, 0))
                        continue;
                }
                
                /*
                // Do not let junctions form
                if (!network->form_junctions) {
                    if (merge_nodes_junction(network, conn, mergenode1, mergenode2)) continue;
                }
                */
                
                // Last check...
                //int gpv = merge_nodes_plane_violation(network, conn, mergenode1, mergenode2, newpos);
                //if (gpv) {
                //    printf("Warning: glide plane violation during merge hinge collision\n");
                //}
                
                //check_node_plane_violation(network, conn, mergenode1, "before merge hinge collision");
                //check_node_plane_violation(network, conn, mergenode2, "before merge hinge collision");
                
                bool merge_error = network->merge_nodes_position(mergenode1, mergenode2, newpos, system->dEp);
                nmerge++;
                
                // Attempt to fix glide plane violations that may have been
                // introduced during the merge operation.
                if (!merge_error && system->crystal.use_glide_planes) {
                    
                    system->crystal.reset_node_glide_planes(network, mergenode1);
                    
                    Vec3 newpos0 = newpos;
                    newpos = AdjustMergePoint(system, network, mergenode1, vmn1, newpos);
                    
                    if ((newpos-newpos0).norm2() <= 16 * mindist2)
                        network->move_node(mergenode1, newpos, system->dEp);
                }
                
                //check_node_plane_violation(network, conn, mergenode1, "after merge hinge collision");
            }
        }
    }
    
    if (nmerge > 0)
        network->purge_network();
}

} // namespace ExaDiS
